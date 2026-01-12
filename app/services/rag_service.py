"""RAG (Retrieval-Augmented Generation) service.

This service provides:
1. Query validation and transformation (AI Agent 2)
2. Vector similarity search
3. Context-aware result construction
"""

import json
import logging
from typing import Any, Literal
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_session_context
from app.exceptions import QueryGuardrailError
from app.models import ProcessingStatus
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


# Query guardrail system prompt
QUERY_GUARDRAIL_PROMPT = """Analyze the following search query and determine if it's valid for semantic search.

Rules:
1. GARBAGE: Random characters, gibberish, nonsensical input (e.g., "Mytj", "asdfgh", "!!!???") → Return INVALID
2. INCOMPLETE: Sentence fragments without clear intent (e.g., "I want to know about", "How to", "What is the") → Return INCOMPLETE  
3. VAGUE: Single words or very short unclear phrases (e.g., "death", "success", "money") → Rewrite to a full semantic sentence that captures likely user intent
4. GOOD: Clear, complete questions or statements with searchable meaning → Pass through unchanged

Examples of transformations for VAGUE queries:
- "death" → "What are the philosophical perspectives on death and mortality?"
- "success" → "What advice and strategies are shared for achieving success?"
- "money" → "What financial advice and perspectives on money are discussed?"

Respond ONLY with valid JSON (no markdown, no code blocks):
{
  "status": "OK" | "INVALID" | "INCOMPLETE",
  "transformed_query": "<semantic query>",
  "error_message": "<helpful message if status is not OK, otherwise null>"
}

Important:
- If status is "OK", transformed_query must contain the query (original or rewritten)
- If status is "INVALID" or "INCOMPLETE", error_message must explain the issue
- Always return valid JSON only, no other text"""


class QueryGuardrailResult(BaseModel):
    """Result from query guardrail validation."""

    status: Literal["OK", "INVALID", "INCOMPLETE"]
    transformed_query: str | None = None
    error_message: str | None = None


class ChunkContext(BaseModel):
    """Context segments surrounding a matched chunk."""

    previous_segment: str | None = Field(None, description="Content from chunk_index - 1")
    current_segment: str = Field(..., description="The matched chunk content")
    next_segment: str | None = Field(None, description="Content from chunk_index + 1")


class SearchResultItem(BaseModel):
    """A single search result with context."""

    video_id: str
    similarity_score: float = Field(
        ..., description="Cosine similarity score (0-1, higher is better)"
    )
    context: ChunkContext
    video_title: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    duration_seconds: int | None = None


class SearchResult(BaseModel):
    """Complete search response."""

    query: str = Field(..., description="Original user query")
    transformed_query: str = Field(
        ..., description="Query after guardrail transformation"
    )
    results: list[SearchResultItem] = Field(
        ..., description="Ranked list of relevant video segments"
    )
    total_results: int = Field(..., description="Number of results returned")


class RAGService:
    """Service for RAG-based semantic search.

    Provides query validation, vector search, and context retrieval
    for finding relevant video segments.
    """

    def __init__(
        self,
        guardrail_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """Initialize the RAG service.

        Args:
            guardrail_model: LLM model for query validation. Defaults to settings.
            api_key: OpenRouter API key. Defaults to settings.
            api_base: OpenRouter API base URL. Defaults to settings.
        """
        self.guardrail_model = guardrail_model or settings.query_guardrail_model
        self.api_key = api_key or settings.openrouter_api_key
        self.api_base = api_base or settings.openrouter_api_base
        self.embedding_service = get_embedding_service()

        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance for guardrail."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.guardrail_model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                timeout=30.0,
            )
        return self._llm

    async def validate_query(self, query: str) -> QueryGuardrailResult:
        """Validate and potentially transform a user query.

        Args:
            query: The user's search query.

        Returns:
            QueryGuardrailResult: Validation result with status and transformed query.

        Raises:
            QueryGuardrailError: If the query is invalid or incomplete.
        """
        if not query or not query.strip():
            raise QueryGuardrailError(
                message="Query cannot be empty",
                status="INVALID",
            )

        messages = [
            SystemMessage(content=QUERY_GUARDRAIL_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()

            # Try to parse JSON response
            try:
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                result_dict = json.loads(response_text)
                result = QueryGuardrailResult(**result_dict)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse guardrail response: {e}")
                # Fallback: treat as valid query if we can't parse
                result = QueryGuardrailResult(
                    status="OK",
                    transformed_query=query,
                )

            # Handle validation results
            if result.status == "INVALID":
                raise QueryGuardrailError(
                    message=result.error_message or "Query is not valid for search",
                    status="INVALID",
                )

            if result.status == "INCOMPLETE":
                raise QueryGuardrailError(
                    message=result.error_message
                    or "Query is incomplete, please provide more detail",
                    status="INCOMPLETE",
                )

            # Ensure we have a transformed query
            if not result.transformed_query:
                result.transformed_query = query

            logger.info(f"Query validated: '{query}' → '{result.transformed_query}'")
            return result

        except QueryGuardrailError:
            raise
        except Exception as e:
            logger.error(f"Query guardrail failed: {e}")
            # On error, pass through the original query
            return QueryGuardrailResult(
                status="OK",
                transformed_query=query,
            )

    async def _vector_search(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        top_k: int,
        similarity_threshold: float,
        channel_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            session: Database session.
            query_embedding: Query embedding vector.
            top_k: Maximum results to return.
            similarity_threshold: Minimum similarity (0-1).
            channel_id: Optional channel filter.

        Returns:
            list: Raw search results from database.
        """
        # Convert similarity threshold to distance threshold
        # PGVector uses cosine distance = 1 - cosine similarity
        distance_threshold = 1 - similarity_threshold

        logger.info(
            f"Vector search params: top_k={top_k}, "
            f"similarity_threshold={similarity_threshold}, "
            f"distance_threshold={distance_threshold}, "
            f"channel_id={channel_id}"
        )

        # Build the query using raw SQL for PGVector operations
        # Format the vector as a PostgreSQL array string with proper precision
        query_vector_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

        sql = text("""
            SELECT 
                vc.id as chunk_id,
                vc.content,
                vc.chunk_index,
                vc.document_id,
                vd.source_video_id,
                vd.source_channel_id,
                vd.meta_data,
                (vc.embedding <=> CAST(:query_vector AS vector)) as distance
            FROM video_chunks vc
            JOIN video_documents vd ON vc.document_id = vd.id
            WHERE vd.status = :status
              AND (vc.embedding <=> CAST(:query_vector AS vector)) < :distance_threshold
              AND (CAST(:channel_id AS TEXT) IS NULL OR vd.source_channel_id = :channel_id)
            ORDER BY distance ASC
            LIMIT :top_k
        """)

        result = await session.execute(
            sql,
            {
                "query_vector": query_vector_str,
                "status": ProcessingStatus.COMPLETED.value,
                "distance_threshold": distance_threshold,
                "channel_id": channel_id,
                "top_k": top_k,
            },
        )

        rows = result.fetchall()
        logger.info(f"Vector search returned {len(rows)} results before filtering")

        if len(rows) == 0:
            logger.warning(
                f"No results found! Check if: "
                f"1) Embeddings exist in DB, "
                f"2) Status is COMPLETED, "
                f"3) Distance threshold ({distance_threshold}) is too strict"
            )

        return [
            {
                "chunk_id": row.chunk_id,
                "content": row.content,
                "chunk_index": row.chunk_index,
                "document_id": row.document_id,
                "source_video_id": row.source_video_id,
                "source_channel_id": row.source_channel_id,
                "meta_data": row.meta_data,
                "similarity_score": 1 - row.distance,  # Convert distance to similarity
            }
            for row in rows
        ]

    async def _fetch_context(
        self,
        session: AsyncSession,
        document_id: UUID,
        chunk_index: int,
    ) -> ChunkContext:
        """Fetch neighboring chunks for context.

        Args:
            session: Database session.
            document_id: Parent document ID.
            chunk_index: Current chunk index.

        Returns:
            ChunkContext: Current chunk with neighbors.
        """
        sql = text("""
            SELECT content, chunk_index
            FROM video_chunks
            WHERE document_id = :document_id
              AND chunk_index IN (:prev_idx, :curr_idx, :next_idx)
            ORDER BY chunk_index ASC
        """)

        result = await session.execute(
            sql,
            {
                "document_id": str(document_id),
                "prev_idx": chunk_index - 1,
                "curr_idx": chunk_index,
                "next_idx": chunk_index + 1,
            },
        )

        chunks_map = {row.chunk_index: row.content for row in result.fetchall()}

        return ChunkContext(
            previous_segment=chunks_map.get(chunk_index - 1),
            current_segment=chunks_map.get(chunk_index, ""),
            next_segment=chunks_map.get(chunk_index + 1),
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        channel_id: str | None = None,
    ) -> SearchResult:
        """Perform a full RAG search.

        Args:
            query: User search query.
            top_k: Maximum results (default from settings).
            similarity_threshold: Minimum similarity (default from settings).
            channel_id: Optional channel filter.

        Returns:
            SearchResult: Complete search results with context.

        Raises:
            QueryGuardrailError: If query validation fails.
            EmbeddingError: If query embedding fails.
        """
        # Apply defaults
        top_k = top_k or settings.rag_top_k_default
        similarity_threshold = (
            similarity_threshold or settings.rag_similarity_threshold_default
        )

        # Step 1: Validate and transform query
        guardrail_result = await self.validate_query(query)
        transformed_query = guardrail_result.transformed_query or query

        # Step 2: Generate query embedding
        query_embedding = await self.embedding_service.embed_query(transformed_query)

        # Step 3 & 4: Vector search and fetch context
        async with get_session_context() as session:
            raw_results = await self._vector_search(
                session=session,
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                channel_id=channel_id,
            )

            # Step 5: Build results with context
            results: list[SearchResultItem] = []

            for raw in raw_results:
                # Fetch context chunks
                context = await self._fetch_context(
                    session=session,
                    document_id=raw["document_id"],
                    chunk_index=raw["chunk_index"],
                )

                # Extract metadata
                meta = raw.get("meta_data", {}) or {}

                result_item = SearchResultItem(
                    video_id=raw["source_video_id"],
                    similarity_score=raw["similarity_score"],
                    context=context,
                    video_title=meta.get("video_title"),
                    channel_id=raw["source_channel_id"],
                    channel_name=meta.get("channel_name"),
                    published_at=meta.get("published_at"),
                    duration_seconds=meta.get("duration_seconds"),
                )
                results.append(result_item)

        logger.info(
            f"Search completed: '{query}' → {len(results)} results "
            f"(threshold: {similarity_threshold})"
        )

        return SearchResult(
            query=query,
            transformed_query=transformed_query,
            results=results,
            total_results=len(results),
        )


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

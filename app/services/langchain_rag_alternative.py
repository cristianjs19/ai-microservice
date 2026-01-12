"""Alternative RAG implementation using LangChain's PGVector.

This is an alternative to the raw SQL approach. Use this if you prefer
LangChain's abstractions over direct SQL queries.

Pros:
- Simpler API, less boilerplate
- Handles vector serialization automatically
- Built-in retry logic
- Easier to swap embedding models

Cons:
- Less control over query optimization
- Harder to debug complex queries
- May not support all pgvector features
- Additional dependency layer

To use this approach:
1. Install: pip install langchain-postgres
2. Replace _vector_search method in rag_service.py
3. Initialize PGVector store in __init__
"""

from typing import Any
from uuid import UUID

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.embedding_service import get_embedding_service


class LangChainRAGService:
    """Alternative RAG service using LangChain's PGVector abstraction."""

    def __init__(self):
        """Initialize with LangChain PGVector."""
        self.embedding_service = get_embedding_service()

        # Initialize PGVector store
        self.vector_store = PGVector(
            embeddings=self.embedding_service.embeddings,
            collection_name="video_chunks",
            connection=settings.database_url,
            distance_strategy=DistanceStrategy.COSINE,
            use_jsonb=True,
        )

    async def search_with_langchain(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        filter_dict: dict[str, Any] | None = None,
    ):
        """
        Perform vector search using LangChain's abstraction.

        Args:
            query: Search query text
            top_k: Number of results
            similarity_threshold: Minimum similarity score
            filter_dict: Optional metadata filters

        Returns:
            List of (Document, similarity_score) tuples
        """
        # LangChain handles embedding generation automatically
        results = await self.vector_store.asimilarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            score_threshold=similarity_threshold,
            filter=filter_dict,
        )

        return results


# Example usage in the main RAG service:
"""
class RAGService:
    def __init__(self):
        # Option 1: Raw SQL (current approach - more control)
        self.use_langchain = False
        
        # Option 2: LangChain PGVector (simpler, less control)
        # self.use_langchain = True
        # self.langchain_rag = LangChainRAGService()
    
    async def _vector_search(self, ...):
        if self.use_langchain:
            # Use LangChain's abstraction
            results = await self.langchain_rag.search_with_langchain(...)
            # Transform results to match expected format
            return self._transform_langchain_results(results)
        else:
            # Use raw SQL (current implementation)
            ...
"""

"""Search API endpoint."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user_optional, get_rag, get_search_history_repository
from app.api.v1.schemas import (
    ChunkContext,
    ErrorResponse,
    SearchQueryRequest,
    SearchResponse,
    SearchResultItem,
)
from app.exceptions import EmbeddingError, QueryGuardrailError
from app.models.users import User
from app.repositories.users import SearchHistoryRepository
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or incomplete query"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Semantic search across video transcripts",
    description="""
    Perform a RAG-based semantic search to find relevant video segments.
    
    The query goes through validation and optional transformation before
    being embedded and compared against stored video chunks.
    
    Results include context (previous and next segments) for each match.
    
    **Authentication is OPTIONAL** - the endpoint works for both authenticated
    and anonymous users. If authenticated, the search is logged to the user's
    search history for future reference.
    """,
)
async def search(
    request: SearchQueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag)],
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
    search_history_repo: Annotated[
        SearchHistoryRepository, Depends(get_search_history_repository)
    ] = None,
) -> SearchResponse:
    """Search for relevant video segments.

    Args:
        request: Search query parameters.
        rag_service: Injected RAG service.
        current_user: Optional authenticated user (for history tracking).
        search_history_repo: Search history repository (for tracking).

    Returns:
        SearchResponse: Search results with context.

    Raises:
        HTTPException: 400 if query is invalid/incomplete, 500 on server error.
    """
    start_time = time.time()

    try:
        result = await rag_service.search(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            channel_id=request.channel_id,
        )

        # Convert internal models to API schemas
        response_items = [
            SearchResultItem(
                video_id=item.video_id,
                similarity_score=item.similarity_score,
                context=ChunkContext(
                    previous_segment=item.context.previous_segment,
                    current_segment=item.context.current_segment,
                    next_segment=item.context.next_segment,
                ),
                video_title=item.video_title,
                channel_id=item.channel_id,
                channel_name=item.channel_name,
                published_at=item.published_at,
                duration_seconds=item.duration_seconds,
            )
            for item in result.results
        ]

        response = SearchResponse(
            query=result.query,
            transformed_query=result.transformed_query,
            results=response_items,
            total_results=result.total_results,
        )

        # Track search history for authenticated users
        if current_user and search_history_repo:
            processing_time_ms = int((time.time() - start_time) * 1000)
            try:
                await search_history_repo.create_search_record(
                    user_id=current_user.id,
                    query=request.query,
                    transformed_query=result.transformed_query,
                    channel_id=request.channel_id,
                    top_k=request.top_k,
                    similarity_threshold=request.similarity_threshold,
                    results_count=result.total_results,
                    processing_time_ms=processing_time_ms,
                )
                logger.info(
                    f"Search history recorded for user {current_user.id}: "
                    f"{request.query[:50]}..."
                )
            except Exception as e:
                # Don't fail the request if history tracking fails
                logger.error(f"Failed to record search history: {e}")

        return response

    except QueryGuardrailError as e:
        logger.warning(f"Query validation failed: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": e.status,
                    "message": e.message,
                    "details": e.details,
                }
            },
        )

    except EmbeddingError as e:
        logger.error(f"Embedding error: {e.message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "EMBEDDING_ERROR",
                    "message": "Failed to process query",
                    "details": e.details,
                }
            },
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": str(e),
                }
            },
        )

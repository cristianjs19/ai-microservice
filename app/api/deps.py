"""Dependency injection for API endpoints."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.services.rag_service import RAGService, get_rag_service
from app.services.processing_pipeline import (
    VideoProcessingPipeline,
    get_processing_pipeline,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for request handling.

    Yields:
        AsyncSession: Database session that auto-commits on success.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_rag() -> RAGService:
    """Get the RAG service instance.

    Returns:
        RAGService: The singleton RAG service.
    """
    return get_rag_service()


def get_pipeline() -> VideoProcessingPipeline:
    """Get the video processing pipeline instance.

    Returns:
        VideoProcessingPipeline: The singleton processing pipeline.
    """
    return get_processing_pipeline()

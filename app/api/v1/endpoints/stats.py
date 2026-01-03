"""Stats API endpoint."""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.api.v1.schemas import StatsResponse
from app.models import ProcessingStatus, VideoChunk, VideoDocument

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get service statistics",
    description="Returns counts of documents and chunks by processing status.",
)
async def get_stats(
    session: AsyncSession = Depends(get_db_session),
) -> StatsResponse:
    """Get service statistics.

    Args:
        session: Database session.

    Returns:
        StatsResponse: Document and chunk counts.
    """
    # Get document counts by status
    status_query = select(
        VideoDocument.status,
        func.count(VideoDocument.id).label("count"),
    ).group_by(VideoDocument.status)

    status_result = await session.execute(status_query)
    status_counts = {row.status: row.count for row in status_result}

    # Get total chunk count
    chunk_query = select(func.count(VideoChunk.id))
    chunk_result = await session.execute(chunk_query)
    total_chunks = chunk_result.scalar() or 0

    # Calculate totals
    total_documents = sum(status_counts.values())

    return StatsResponse(
        total_documents=total_documents,
        completed_documents=status_counts.get(ProcessingStatus.COMPLETED, 0),
        pending_documents=status_counts.get(ProcessingStatus.PENDING, 0),
        processing_documents=status_counts.get(ProcessingStatus.PROCESSING, 0),
        failed_documents=status_counts.get(ProcessingStatus.FAILED, 0),
        total_chunks=total_chunks,
    )

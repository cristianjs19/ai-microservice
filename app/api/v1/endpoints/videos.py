"""Videos API endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db_session
from app.api.v1.schemas import VideoListItem, VideoListResponse
from app.repositories.videos import VideoRepository

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/videos",
    response_model=VideoListResponse,
    summary="List all processed videos",
    description="""
    Retrieve a paginated list of all processed videos in the system.
    
    Each video includes its processing status, metadata (title, channel, publication date),
    and timestamps for creation and last update.
    
    Results are ordered by creation date (most recent first).
    """,
)
async def list_videos(
    skip: Annotated[int, Query(ge=0, description="Number of videos to skip")] = 0,
    limit: Annotated[
        int, Query(gt=0, le=100, description="Maximum videos to return")
    ] = 20,
    session: AsyncSession = Depends(get_db_session),
) -> VideoListResponse:
    """List all videos with pagination.

    Args:
        skip: Pagination offset (number of items to skip)
        limit: Maximum number of items to return (max 100)
        session: Database session

    Returns:
        VideoListResponse: Paginated list of videos with metadata
    """
    video_repo = VideoRepository(session)
    videos, total_count = await video_repo.get_all_videos(skip=skip, limit=limit)

    # Convert to response items
    video_items = [
        VideoListItem(
            id=str(video.id),
            source_video_id=video.source_video_id,
            source_channel_id=video.source_channel_id,
            status=video.status.value,  # Convert enum to string
            title=video.meta_data.get("title") if video.meta_data else None,
            channel_name=video.meta_data.get("channel_name") if video.meta_data else None,
            published_at=video.meta_data.get("published_at") if video.meta_data else None,
            created_at=video.created_at.isoformat(),
            updated_at=video.updated_at.isoformat(),
        )
        for video in videos
    ]

    return VideoListResponse(
        videos=video_items,
        total=total_count,
        skip=skip,
        limit=limit,
        count=len(video_items),
    )

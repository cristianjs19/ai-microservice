"""Video repository for database operations."""

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.videos import VideoDocument


class VideoRepository:
    """Repository for video-related database operations.

    Follows repository pattern for clean separation of concerns
    and testability.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def get_all_videos(
        self,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[Sequence[VideoDocument], int]:
        """Retrieve all videos with pagination.

        Args:
            skip: Number of videos to skip (pagination offset)
            limit: Maximum number of videos to return

        Returns:
            Tuple of (videos, total_count)
        """
        # Get total count
        count_query = select(len(VideoDocument.__table__.columns))
        total_count = await self.session.execute(
            select(len(VideoDocument.__table__.columns)).select_from(VideoDocument)
        )

        # Use a simpler count approach
        from sqlalchemy import func

        count_result = await self.session.execute(select(func.count(VideoDocument.id)))
        total_count = count_result.scalar() or 0

        # Get paginated results, ordered by creation date (most recent first)
        query = (
            select(VideoDocument)
            .order_by(VideoDocument.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        videos = result.scalars().all()

        return videos, total_count

    async def get_video_by_id(self, video_id: str) -> VideoDocument | None:
        """Get a single video by source_video_id.

        Args:
            video_id: The source video ID

        Returns:
            VideoDocument if found, None otherwise
        """
        query = select(VideoDocument).where(VideoDocument.source_video_id == video_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

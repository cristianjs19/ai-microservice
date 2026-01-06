"""Video processing pipeline service.

This service orchestrates the full video processing pipeline:
1. Fetch video data from Fetching Service
2. Format transcript using LLM
3. Chunk formatted text
4. Generate embeddings
5. Store in database
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_context
from app.exceptions import FetchingServiceError, ProcessingError
from app.models import ProcessingStatus, VideoChunk, VideoDocument
from app.services.chunking_service import get_chunking_service
from app.services.embedding_service import get_embedding_service
from app.services.fetching_client import FetchingServiceClient
from app.services.formatting_service import get_formatting_service

logger = logging.getLogger(__name__)


class VideoProcessingPipeline:
    """Orchestrates the complete video processing pipeline.

    This class coordinates all the steps needed to process a video:
    fetching data, formatting, chunking, embedding, and storage.
    """

    def __init__(self):
        """Initialize the processing pipeline with required services."""
        self.fetching_client = FetchingServiceClient()
        self.formatting_service = get_formatting_service()
        self.chunking_service = get_chunking_service()
        self.embedding_service = get_embedding_service()

    async def _fetch_video_data(
        self,
        video_id: str,
    ) -> tuple[dict[str, Any], str]:
        """Fetch video metadata and caption from Fetching Service.

        Args:
            video_id: The YouTube video ID.

        Returns:
            tuple: (metadata dict, raw transcript string)

        Raises:
            ProcessingError: If fetching fails.
        """
        try:
            async with self.fetching_client as client:
                metadata = await client.get_video_metadata(video_id)
                caption = await client.get_video_caption(video_id)

            if not caption or not caption.strip():
                raise ProcessingError(
                    message="Video has no transcript",
                    details=f"Video {video_id} returned empty caption",
                )

            return metadata, caption

        except FetchingServiceError as e:
            raise ProcessingError(
                message=f"Failed to fetch video data: {e.message}",
                details=e.details,
            )

    def _extract_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant metadata for storage.

        Args:
            metadata: Raw metadata from Fetching Service.

        Returns:
            dict: Extracted metadata for JSONB storage.
        """
        return {
            "title": metadata.get("title", ""),
            "channel_name": metadata.get("channel_name", ""),
            "channel_id": metadata.get("channel_id", ""),
            "published_at": metadata.get("upload_date"),
            "duration_seconds": metadata.get("duration"),
            "view_count": metadata.get("view_count"),
            "url": metadata.get("url", ""),
        }

    async def _create_video_document(
        self,
        session: AsyncSession,
        video_id: str,
        channel_id: str | None,
        metadata: dict[str, Any],
    ) -> VideoDocument:
        """Create initial VideoDocument record with PENDING status.

        Args:
            session: Database session.
            video_id: Source video ID.
            channel_id: Source channel ID.
            metadata: Extracted metadata.

        Returns:
            VideoDocument: The created document.
        """
        document = VideoDocument(
            source_video_id=video_id,
            source_channel_id=channel_id,
            formatted_content="",  # Will be updated after formatting
            status=ProcessingStatus.PENDING,
            meta_data=metadata,
        )
        session.add(document)
        await session.flush()  # Get the ID without committing

        logger.info(f"Created VideoDocument {document.id} for video {video_id}")
        return document

    async def _update_document_status(
        self,
        session: AsyncSession,
        document: VideoDocument,
        status: ProcessingStatus,
        formatted_content: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update document status and optional fields.

        Args:
            session: Database session.
            document: The document to update.
            status: New processing status.
            formatted_content: Optional formatted content.
            error_message: Optional error message.
        """
        document.status = status
        document.updated_at = datetime.utcnow()

        if formatted_content is not None:
            document.formatted_content = formatted_content

        if error_message is not None:
            document.error_message = error_message

        await session.flush()

    async def _create_video_chunks(
        self,
        session: AsyncSession,
        document: VideoDocument,
        chunks: list[tuple[int, str]],
        embeddings: list[list[float]],
    ) -> list[VideoChunk]:
        """Create VideoChunk records with embeddings.

        Args:
            session: Database session.
            document: Parent VideoDocument.
            chunks: List of (index, content) tuples.
            embeddings: List of embedding vectors.

        Returns:
            list[VideoChunk]: Created chunk records.
        """
        video_chunks = []

        for (chunk_index, content), embedding in zip(chunks, embeddings):
            chunk = VideoChunk(
                document_id=document.id,
                content=content,
                chunk_index=chunk_index,
                embedding=embedding,
            )
            session.add(chunk)
            video_chunks.append(chunk)

        await session.flush()

        logger.info(f"Created {len(video_chunks)} chunks for document {document.id}")
        return video_chunks

    async def process_video(self, video_id: str) -> VideoDocument:
        """Execute the full video processing pipeline.

        Args:
            video_id: The YouTube video ID to process.

        Returns:
            VideoDocument: The completed document record.

        Raises:
            ProcessingError: If any step fails.
        """
        logger.info(f"Starting processing pipeline for video: {video_id}")

        async with get_session_context() as session:
            document: VideoDocument | None = None

            try:
                # Step 1: Fetch video data
                logger.info(f"[{video_id}] Fetching video data...")
                metadata, raw_transcript = await self._fetch_video_data(video_id)

                channel_id = metadata.get("channel_id")
                extracted_metadata = self._extract_metadata(metadata)

                # Step 2: Create document record (PENDING)
                logger.info(f"[{video_id}] Creating document record...")
                document = await self._create_video_document(
                    session=session,
                    video_id=video_id,
                    channel_id=channel_id,
                    metadata=extracted_metadata,
                )

                # Step 3: Format transcript
                logger.info(f"[{video_id}] Formatting transcript...")
                formatted_content = await self.formatting_service.format_transcript(
                    raw_transcript
                )

                # Update status to PROCESSING
                await self._update_document_status(
                    session=session,
                    document=document,
                    status=ProcessingStatus.PROCESSING,
                    formatted_content=formatted_content,
                )

                # Step 4: Chunk formatted text
                logger.info(f"[{video_id}] Chunking text...")
                chunks = self.chunking_service.chunk_text(formatted_content)

                # Step 5: Generate embeddings
                logger.info(
                    f"[{video_id}] Generating embeddings for {len(chunks)} chunks..."
                )
                chunk_contents = [content for _, content in chunks]
                embeddings = await self.embedding_service.embed_documents(chunk_contents)

                # Step 6: Create chunk records
                logger.info(f"[{video_id}] Storing chunks...")
                await self._create_video_chunks(
                    session=session,
                    document=document,
                    chunks=chunks,
                    embeddings=embeddings,
                )

                # Step 7: Update status to COMPLETED
                await self._update_document_status(
                    session=session,
                    document=document,
                    status=ProcessingStatus.COMPLETED,
                )

                logger.info(
                    f"Successfully processed video {video_id}: "
                    f"{len(chunks)} chunks created"
                )

                # Commit happens automatically via context manager
                return document

            except Exception as e:
                logger.error(f"Processing failed for video {video_id}: {e}")

                # Update document status to FAILED if it was created
                if document is not None:
                    try:
                        await self._update_document_status(
                            session=session,
                            document=document,
                            status=ProcessingStatus.FAILED,
                            error_message=str(e),
                        )
                        await session.commit()
                    except Exception as update_error:
                        logger.error(f"Failed to update document status: {update_error}")

                # Re-raise as ProcessingError
                if isinstance(e, ProcessingError):
                    raise
                raise ProcessingError(
                    message=f"Video processing failed: {e}",
                    details=str(e),
                )


# Singleton instance
_pipeline: VideoProcessingPipeline | None = None


def get_processing_pipeline() -> VideoProcessingPipeline:
    """Get the global processing pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VideoProcessingPipeline()
    return _pipeline


async def process_video(video_id: str) -> VideoDocument:
    """Process a video using the global pipeline instance.

    Convenience function for use as a callback.

    Args:
        video_id: The YouTube video ID to process.

    Returns:
        VideoDocument: The processed document.
    """
    pipeline = get_processing_pipeline()
    return await pipeline.process_video(video_id)

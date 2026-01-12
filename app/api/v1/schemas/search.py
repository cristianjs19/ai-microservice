"""Request and response schemas for the search API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# Request Schemas
# ============================================================


class SearchQueryRequest(BaseModel):
    """Request schema for search endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user's search query.",
    )
    channel_id: Optional[str] = Field(
        None,
        description="Optional. Filter results to a specific YouTube channel ID.",
    )
    top_k: int = Field(
        5,
        gt=0,
        le=20,
        description="Maximum number of video candidates to return.",
    )
    similarity_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.7 recommended for BGE models).",
    )


class ProcessVideoRequest(BaseModel):
    """Request schema for manual video processing."""

    source_video_id: str = Field(
        ...,
        min_length=1,
        description="The YouTube video ID to process.",
    )


# ============================================================
# Response Schemas
# ============================================================


class QueryTransformationResult(BaseModel):
    """Result from Query Guardrail (AI Agent 2)."""

    status: Literal["OK", "INVALID", "INCOMPLETE"]
    transformed_query: Optional[str] = None
    error_message: Optional[str] = None


class ChunkContext(BaseModel):
    """Context segments surrounding the matched chunk."""

    previous_segment: Optional[str] = Field(
        None, description="Content from chunk_index - 1"
    )
    current_segment: str = Field(..., description="The matched chunk content")
    next_segment: Optional[str] = Field(None, description="Content from chunk_index + 1")


class SearchResultItem(BaseModel):
    """A single search result with metadata and context."""

    video_id: str
    similarity_score: float = Field(
        ..., description="Cosine similarity score (0-1, higher is better)"
    )

    # Context segments
    context: ChunkContext

    # Metadata from VideoDocument.meta_data JSONB field
    video_title: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    published_at: Optional[str] = None
    duration_seconds: Optional[int] = None


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""

    query: str = Field(..., description="Original user query")
    transformed_query: str = Field(
        ..., description="Query after guardrail transformation"
    )
    results: list[SearchResultItem] = Field(
        ..., description="Ranked list of relevant video segments"
    )
    total_results: int = Field(..., description="Number of results returned")


class ProcessStatusResponse(BaseModel):
    """Response schema for video processing status."""

    source_video_id: str
    status: str  # ProcessingStatus enum as string
    message: Optional[str] = None


class StatsResponse(BaseModel):
    """Response schema for service statistics."""

    total_documents: int
    completed_documents: int
    pending_documents: int
    processing_documents: int
    failed_documents: int
    total_chunks: int


class ErrorDetail(BaseModel):
    """Error detail schema."""

    code: str
    message: str
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    service: str


class ReadyResponse(BaseModel):
    """Readiness check response schema."""

    ready: bool
    checks: dict[str, str]


# ============================================================
# Video List Schemas
# ============================================================


class VideoListItem(BaseModel):
    """A single video item in the list response."""

    id: str = Field(..., description="Video document UUID")
    source_video_id: str = Field(..., description="YouTube video ID")
    source_channel_id: Optional[str] = Field(None, description="YouTube channel ID")
    status: str = Field(
        ..., description="Processing status (pending/processing/completed/failed)"
    )
    title: Optional[str] = Field(None, description="Video title from metadata")
    channel_name: Optional[str] = Field(None, description="Channel name from metadata")
    published_at: Optional[str] = Field(
        None, description="Publication date from metadata"
    )
    created_at: str = Field(..., description="When the document was created")
    updated_at: str = Field(..., description="When the document was last updated")


class VideoListResponse(BaseModel):
    """Response schema for list videos endpoint."""

    videos: list[VideoListItem] = Field(..., description="List of videos")
    total: int = Field(..., description="Total number of videos in database")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Maximum items returned")
    count: int = Field(..., description="Number of items in this response")

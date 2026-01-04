"""User authentication models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class User(Base):
    """User model with phone-based authentication.

    Follows industry best practices for user management:
    - UUID primary keys for security
    - Phone number as unique identifier
    - Secure password hashing (handled at service layer)
    - Audit timestamps
    - Optional email and full name fields
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="Email address - primary authentication identifier",
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Bcrypt hashed password",
    )

    phone: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        unique=True,
        index=True,
        comment="Optional phone number",
    )

    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Account active status",
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Admin/superuser flag",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    search_history: Mapped[list["SearchHistory"]] = relationship(
        "SearchHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<User(id={self.id}, email={self.email}, active={self.is_active})>"


class SearchHistory(Base):
    """Search history tracking for authenticated users.

    Tracks all searches performed by authenticated users for:
    - Analytics and insights
    - User behavior analysis
    - Personalization opportunities
    """

    __tablename__ = "search_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to user who performed the search",
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original search query",
    )

    transformed_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Query after AI transformation/validation",
    )

    channel_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Optional channel filter applied",
    )

    top_k: Mapped[int] = mapped_column(
        nullable=False,
        default=5,
        comment="Number of results requested",
    )

    similarity_threshold: Mapped[float] = mapped_column(
        nullable=False,
        default=0.7,
        comment="Similarity threshold used",
    )

    results_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        comment="Number of results returned",
    )

    processing_time_ms: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        comment="Time taken to process the search in milliseconds",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="search_history",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<SearchHistory(id={self.id}, user_id={self.user_id}, query={self.query[:50]})>"

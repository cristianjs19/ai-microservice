"""User authentication schemas."""

import re
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserBase(BaseModel):
    """Base user schema with common fields."""

    email: Annotated[
        EmailStr,
        Field(
            description="Email address - primary authentication identifier",
            examples=["user@example.com"],
        ),
    ]
    phone: Annotated[
        str | None,
        Field(
            None,
            min_length=5,
            max_length=20,
            pattern=r"^\d+$",
            description="Optional phone number (digits only, minimum 5 digits)",
            examples=["1234567890"],
        ),
    ] = None
    full_name: str | None = Field(
        None,
        max_length=255,
        description="Optional full name",
    )

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        """Validate phone number format if provided."""
        if v is None:
            return v
        if not v.isdigit():
            raise ValueError("Phone number must contain only digits")
        if len(v) < 5:
            raise ValueError("Phone number must be at least 5 digits long")
        return v


class UserRegister(UserBase):
    """Schema for user registration."""

    password: Annotated[
        str,
        Field(
            min_length=8,
            max_length=128,
            description="Password (minimum 8 characters)",
        ),
    ]

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """Schema for user login."""

    email: Annotated[
        EmailStr,
        Field(
            description="Email address",
            examples=["user@example.com"],
        ),
    ]
    password: Annotated[
        str,
        Field(
            min_length=1,
            max_length=128,
            description="Password",
        ),
    ]


class UserUpdate(BaseModel):
    """Schema for user profile updates."""

    email: EmailStr | None = None
    full_name: str | None = Field(None, max_length=255)
    password: str | None = Field(None, min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str | None) -> str | None:
        """Validate password strength if provided."""
        if v is None:
            return v
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserResponse(UserBase):
    """Schema for user responses (excludes sensitive data)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime


class TokenPayload(BaseModel):
    """JWT token payload schema."""

    sub: str = Field(..., description="Subject (user_id)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    type: str = Field(..., description="Token type (access or refresh)")


class TokenResponse(BaseModel):
    """JWT token response schema."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""

    refresh_token: str = Field(..., description="Refresh token")


class UserWithTokens(UserResponse):
    """User response with JWT tokens (used after registration/login)."""

    tokens: TokenResponse


class SearchHistoryResponse(BaseModel):
    """Schema for search history response."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    query: str
    transformed_query: str | None
    channel_id: str | None
    top_k: int
    similarity_threshold: float
    results_count: int
    processing_time_ms: int
    created_at: datetime


class SearchHistoryListResponse(BaseModel):
    """Schema for paginated search history list."""

    items: list[SearchHistoryResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

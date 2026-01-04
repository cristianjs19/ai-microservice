"""Dependency injection for API endpoints."""

import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.database import async_session_factory
from app.core.security import SecurityUtils
from app.exceptions import AuthenticationError, UserNotFoundError
from app.models.users import User
from app.repositories.users import SearchHistoryRepository, UserRepository
from app.services.processing_pipeline import (
    VideoProcessingPipeline,
    get_processing_pipeline,
)
from app.services.rag_service import RAGService, get_rag_service

# Security scheme for bearer token
security_scheme = HTTPBearer(auto_error=False)


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


def get_security_utils(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SecurityUtils:
    """Get security utilities instance.

    Args:
        settings: Application settings

    Returns:
        SecurityUtils: Security utilities for JWT and password operations
    """
    return SecurityUtils(settings)


def get_user_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserRepository:
    """Get user repository instance.

    Args:
        session: Database session

    Returns:
        UserRepository: Repository for user operations
    """
    return UserRepository(session)


def get_search_history_repository(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SearchHistoryRepository:
    """Get search history repository instance.

    Args:
        session: Database session

    Returns:
        SearchHistoryRepository: Repository for search history operations
    """
    return SearchHistoryRepository(session)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_scheme)],
    security_utils: Annotated[SecurityUtils, Depends(get_security_utils)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
) -> User:
    """Get current authenticated user (required authentication).

    This dependency REQUIRES authentication. If no token is provided or token
    is invalid, it raises a 401 error.

    Args:
        credentials: HTTP Bearer token credentials
        security_utils: Security utilities for token verification
        user_repo: User repository

    Returns:
        User: Authenticated user instance

    Raises:
        HTTPException: 401 if not authenticated or user not found
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Verify token and extract payload
        payload = security_utils.verify_token(
            credentials.credentials,
            expected_type="access",
        )

        user_id = uuid.UUID(payload["sub"])

        # Get user from database
        user = await user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user",
            )

        return user

    except (AuthenticationError, UserNotFoundError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)],
    security_utils: Annotated[SecurityUtils, Depends(get_security_utils)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
) -> User | None:
    """Get current authenticated user (optional authentication).

    This dependency allows OPTIONAL authentication. If no token is provided,
    it returns None. If token is provided but invalid, it raises 401.

    This is perfect for endpoints that work differently for authenticated vs
    anonymous users (like the search endpoint).

    Args:
        credentials: HTTP Bearer token credentials (optional)
        security_utils: Security utilities for token verification
        user_repo: User repository

    Returns:
        User: Authenticated user instance, or None if no credentials provided

    Raises:
        HTTPException: 401 if token is provided but invalid
    """
    # No credentials provided - that's ok for optional auth
    if not credentials:
        return None

    try:
        # Verify token and extract payload
        payload = security_utils.verify_token(
            credentials.credentials,
            expected_type="access",
        )

        user_id = uuid.UUID(payload["sub"])

        # Get user from database
        user = await user_repo.get_by_id(user_id)
        if not user:
            # Token was provided but user doesn't exist
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user",
            )

        return user

    except (AuthenticationError, UserNotFoundError, ValueError) as e:
        # Token was provided but invalid - this is an error
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_active_superuser(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Verify that current user is a superuser.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Authenticated superuser

    Raises:
        HTTPException: 403 if user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user

"""Authentication endpoints for user registration, login, and token management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    get_current_user,
    get_search_history_repository,
    get_security_utils,
    get_user_repository,
)
from app.core.security import SecurityUtils
from app.exceptions import AuthenticationError, DuplicateUserError
from app.models.users import User
from app.repositories.users import SearchHistoryRepository, UserRepository
from app.schemas.users import (
    SearchHistoryListResponse,
    SearchHistoryResponse,
    TokenRefresh,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    UserUpdate,
    UserWithTokens,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserWithTokens,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="""
    Register a new user with email and password.
    
    Requirements:
    - Email: valid email address (required)
    - Password: minimum 8 characters, must contain uppercase, lowercase, and digit
    - Phone: optional, minimum 5 digits if provided
    - Full name: optional
    
    Returns user data with access and refresh tokens.
    """,
)
async def register(
    user_data: UserRegister,
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    security_utils: Annotated[SecurityUtils, Depends(get_security_utils)],
) -> UserWithTokens:
    """Register a new user and return tokens."""
    try:
        # Hash the password
        hashed_password = security_utils.hash_password(user_data.password)

        # Create user
        user = await user_repo.create_user(
            email=user_data.email,
            hashed_password=hashed_password,
            phone=user_data.phone,
            full_name=user_data.full_name,
        )

        # Generate tokens
        tokens = security_utils.create_token_pair(str(user.id))

        # Return user with tokens
        return UserWithTokens(
            **user.__dict__,
            tokens=TokenResponse(**tokens),
        )

    except DuplicateUserError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login user",
    description="""
    Authenticate user with email and password.
    
    Returns access and refresh tokens on successful authentication.
    """,
)
async def login(
    credentials: UserLogin,
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    security_utils: Annotated[SecurityUtils, Depends(get_security_utils)],
) -> TokenResponse:
    """Login user and return tokens."""
    # Get user by email
    user = await user_repo.get_by_email(credentials.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if not security_utils.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )

    # Generate tokens
    tokens = security_utils.create_token_pair(str(user.id))

    # Return tokens only
    return TokenResponse(**tokens)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="""
    Use a refresh token to obtain a new access token.
    
    Refresh tokens have longer expiration time than access tokens,
    allowing users to stay logged in without re-entering credentials.
    """,
)
async def refresh_token(
    token_data: TokenRefresh,
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    security_utils: Annotated[SecurityUtils, Depends(get_security_utils)],
) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = security_utils.verify_token(
            token_data.refresh_token,
            expected_type="refresh",
        )

        # Verify user still exists and is active
        import uuid

        user_id = uuid.UUID(payload["sub"])
        user = await user_repo.get_by_id(user_id)

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        # Generate new token pair
        tokens = security_utils.create_token_pair(str(user.id))

        return TokenResponse(**tokens)

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="""
    Get the profile of the currently authenticated user.
    
    Requires valid access token in Authorization header.
    """,
)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Get current user profile."""
    return UserResponse(**current_user.__dict__)


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user",
    description="""
    Update the profile of the currently authenticated user.
    
    Can update: email, full_name, password
    All fields are optional - only provided fields will be updated.
    """,
)
async def update_me(
    user_update: UserUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    security_utils: Annotated[SecurityUtils, Depends(get_security_utils)],
) -> UserResponse:
    """Update current user profile."""
    update_data = user_update.model_dump(exclude_unset=True)

    # Hash password if provided
    if "password" in update_data:
        update_data["hashed_password"] = security_utils.hash_password(
            update_data.pop("password")
        )

    # Update user
    updated_user = await user_repo.update_user(current_user.id, **update_data)

    return UserResponse(**updated_user.__dict__)


@router.get(
    "/me/search-history",
    response_model=SearchHistoryListResponse,
    summary="Get my search history",
    description="""
    Get paginated search history for the currently authenticated user.
    
    Returns all searches performed by the user, ordered by most recent first.
    """,
)
async def get_my_search_history(
    current_user: Annotated[User, Depends(get_current_user)],
    search_repo: Annotated[
        SearchHistoryRepository, Depends(get_search_history_repository)
    ],
    page: int = 1,
    page_size: int = 50,
) -> SearchHistoryListResponse:
    """Get current user's search history."""
    # Validate pagination
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be >= 1",
        )
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be between 1 and 100",
        )

    skip = (page - 1) * page_size

    # Get search history
    items, total = await search_repo.get_user_search_history(
        user_id=current_user.id,
        skip=skip,
        limit=page_size,
    )

    total_pages = (total + page_size - 1) // page_size  # Ceiling division

    return SearchHistoryListResponse(
        items=[SearchHistoryResponse(**item.__dict__) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )

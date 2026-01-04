"""User repository for database operations."""

import uuid
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import DuplicateUserError, UserNotFoundError
from app.models.users import SearchHistory, User


class UserRepository:
    """Repository for user-related database operations.

    Follows repository pattern for clean separation of concerns
    and testability.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create_user(
        self,
        email: str,
        hashed_password: str,
        phone: str | None = None,
        full_name: str | None = None,
    ) -> User:
        """Create a new user.

        Args:
            email: Email address (unique identifier)
            hashed_password: Pre-hashed password
            phone: Optional phone number
            full_name: Optional full name

        Returns:
            Created user instance

        Raises:
            DuplicateUserError: If email or phone already exists
        """
        # Check if user already exists
        existing = await self.get_by_email(email)
        if existing:
            raise DuplicateUserError(f"User with email {email} already exists")

        if phone:
            existing_phone = await self.get_by_phone(phone)
            if existing_phone:
                raise DuplicateUserError(f"User with phone {phone} already exists")

        user = User(
            email=email,
            hashed_password=hashed_password,
            phone=phone,
            full_name=full_name,
        )

        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)

        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Get user by ID.

        Args:
            user_id: User's UUID

        Returns:
            User instance or None if not found
        """
        stmt = select(User).where(User.id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_phone(self, phone: str) -> User | None:
        """Get user by phone number.

        Args:
            phone: Phone number

        Returns:
            User instance or None if not found
        """
        stmt = select(User).where(User.phone == phone)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email.

        Args:
            email: Email address

        Returns:
            User instance or None if not found
        """
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_user(
        self,
        user_id: uuid.UUID,
        **kwargs,
    ) -> User:
        """Update user fields.

        Args:
            user_id: User's UUID
            **kwargs: Fields to update

        Returns:
            Updated user instance

        Raises:
            UserNotFoundError: If user doesn't exist
        """
        user = await self.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        for key, value in kwargs.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)

        await self.session.commit()
        await self.session.refresh(user)

        return user

    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """Delete a user (soft delete by setting is_active=False).

        Args:
            user_id: User's UUID

        Returns:
            True if deleted successfully

        Raises:
            UserNotFoundError: If user doesn't exist
        """
        user = await self.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        user.is_active = False
        await self.session.commit()

        return True

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: bool | None = None,
    ) -> Sequence[User]:
        """List users with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            is_active: Optional filter by active status

        Returns:
            List of users
        """
        stmt = select(User)

        if is_active is not None:
            stmt = stmt.where(User.is_active == is_active)

        stmt = stmt.offset(skip).limit(limit).order_by(User.created_at.desc())

        result = await self.session.execute(stmt)
        return result.scalars().all()


class SearchHistoryRepository:
    """Repository for search history operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session

    async def create_search_record(
        self,
        user_id: uuid.UUID,
        query: str,
        transformed_query: str | None,
        channel_id: str | None,
        top_k: int,
        similarity_threshold: float,
        results_count: int,
        processing_time_ms: int,
    ) -> SearchHistory:
        """Create a search history record.

        Args:
            user_id: User who performed the search
            query: Original search query
            transformed_query: AI-transformed query
            channel_id: Optional channel filter
            top_k: Number of results requested
            similarity_threshold: Similarity threshold used
            results_count: Number of results returned
            processing_time_ms: Processing time in milliseconds

        Returns:
            Created search history instance
        """
        search_record = SearchHistory(
            user_id=user_id,
            query=query,
            transformed_query=transformed_query,
            channel_id=channel_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            results_count=results_count,
            processing_time_ms=processing_time_ms,
        )

        self.session.add(search_record)
        await self.session.commit()
        await self.session.refresh(search_record)

        return search_record

    async def get_user_search_history(
        self,
        user_id: uuid.UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[Sequence[SearchHistory], int]:
        """Get paginated search history for a user.

        Args:
            user_id: User's UUID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (search history list, total count)
        """
        # Get total count
        count_stmt = (
            select(func.count())
            .select_from(SearchHistory)
            .where(SearchHistory.user_id == user_id)
        )
        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar_one()

        # Get paginated results
        stmt = (
            select(SearchHistory)
            .where(SearchHistory.user_id == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(SearchHistory.created_at.desc())
        )

        result = await self.session.execute(stmt)
        items = result.scalars().all()

        return items, total

    async def get_search_by_id(
        self,
        search_id: uuid.UUID,
    ) -> SearchHistory | None:
        """Get a specific search record by ID.

        Args:
            search_id: Search history UUID

        Returns:
            Search history instance or None if not found
        """
        stmt = select(SearchHistory).where(SearchHistory.id == search_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_search_record(
        self,
        search_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """Delete a search record (must belong to the user).

        Args:
            search_id: Search history UUID
            user_id: User's UUID (for authorization)

        Returns:
            True if deleted successfully

        Raises:
            UserNotFoundError: If search record doesn't exist or doesn't belong to user
        """
        search = await self.get_search_by_id(search_id)
        if not search or search.user_id != user_id:
            raise UserNotFoundError("Search record not found")

        await self.session.delete(search)
        await self.session.commit()

        return True

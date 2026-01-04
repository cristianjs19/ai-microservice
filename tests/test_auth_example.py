"""Example tests for authentication system.

These tests demonstrate how to test the authentication system.
Run with: pytest tests/test_auth_example.py -v
"""

import pytest
from fastapi import status
from httpx import AsyncClient

from app.core.security import SecurityUtils
from app.config import Settings


# ============================================================================
# UNIT TESTS - Security Utils
# ============================================================================


class TestSecurityUtils:
    """Test security utilities (password hashing, JWT tokens)."""

    @pytest.fixture
    def security_utils(self):
        """Create security utils instance for testing."""
        settings = Settings(
            jwt_secret_key="test-secret-key-do-not-use-in-production",
            jwt_algorithm="HS256",
            jwt_access_token_expire_minutes=30,
            jwt_refresh_token_expire_days=7,
        )
        return SecurityUtils(settings)

    def test_password_hashing(self, security_utils):
        """Test password hashing and verification."""
        password = "SecurePass123"
        hashed = security_utils.hash_password(password)

        # Hash should be different from original
        assert hashed != password

        # Should verify correct password
        assert security_utils.verify_password(password, hashed)

        # Should not verify wrong password
        assert not security_utils.verify_password("WrongPassword", hashed)

    def test_access_token_creation(self, security_utils):
        """Test JWT access token creation."""
        user_id = "123e4567-e89b-12d3-a456-426614174000"

        # Create token
        token = security_utils.create_access_token(user_id)

        # Should be a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0

        # Should be verifiable
        payload = security_utils.verify_token(token, expected_type="access")
        assert payload["sub"] == user_id
        assert payload["type"] == "access"

    def test_refresh_token_creation(self, security_utils):
        """Test JWT refresh token creation."""
        user_id = "123e4567-e89b-12d3-a456-426614174000"

        # Create token
        token = security_utils.create_refresh_token(user_id)

        # Should be a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0

        # Should be verifiable
        payload = security_utils.verify_token(token, expected_type="refresh")
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"

    def test_token_pair_creation(self, security_utils):
        """Test creating both access and refresh tokens."""
        user_id = "123e4567-e89b-12d3-a456-426614174000"

        # Create token pair
        tokens = security_utils.create_token_pair(user_id)

        # Should have both tokens
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        # Both should be verifiable
        access_payload = security_utils.verify_token(
            tokens["access_token"], expected_type="access"
        )
        refresh_payload = security_utils.verify_token(
            tokens["refresh_token"], expected_type="refresh"
        )

        assert access_payload["sub"] == user_id
        assert refresh_payload["sub"] == user_id

    def test_invalid_token_verification(self, security_utils):
        """Test that invalid tokens are rejected."""
        from app.exceptions import AuthenticationError

        # Invalid token should raise error
        with pytest.raises(AuthenticationError):
            security_utils.verify_token("invalid-token", expected_type="access")

    def test_wrong_token_type(self, security_utils):
        """Test that using wrong token type is rejected."""
        from app.exceptions import AuthenticationError

        user_id = "123e4567-e89b-12d3-a456-426614174000"
        access_token = security_utils.create_access_token(user_id)

        # Using access token where refresh token expected should fail
        with pytest.raises(AuthenticationError):
            security_utils.verify_token(access_token, expected_type="refresh")


# ============================================================================
# INTEGRATION TESTS - API Endpoints
# ============================================================================


@pytest.mark.asyncio
class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""

    async def test_register_user(self, client: AsyncClient):
        """Test user registration."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "1234567890",
                "password": "SecurePass123",
                "email": "test@example.com",
                "full_name": "Test User",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Check user data
        assert data["phone"] == "1234567890"
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"
        assert data["is_active"] is True
        assert data["is_superuser"] is False

        # Check tokens are present
        assert "tokens" in data
        assert "access_token" in data["tokens"]
        assert "refresh_token" in data["tokens"]
        assert data["tokens"]["token_type"] == "bearer"

    async def test_register_duplicate_phone(self, client: AsyncClient):
        """Test that registering with duplicate phone fails."""
        user_data = {
            "phone": "9876543210",
            "password": "SecurePass123",
            "email": "user1@example.com",
        }

        # First registration should succeed
        response1 = await client.post("/api/v1/auth/register", json=user_data)
        assert response1.status_code == status.HTTP_201_CREATED

        # Second registration with same phone should fail
        user_data["email"] = "user2@example.com"  # Different email
        response2 = await client.post("/api/v1/auth/register", json=user_data)
        assert response2.status_code == status.HTTP_409_CONFLICT

    async def test_register_weak_password(self, client: AsyncClient):
        """Test that weak passwords are rejected."""
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "5555555555",
                "password": "weak",  # Too short, no uppercase, no digit
                "email": "test@example.com",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_login_success(self, client: AsyncClient):
        """Test successful login."""
        # First register a user
        await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "1111111111",
                "password": "TestPass123",
                "email": "login@example.com",
            },
        )

        # Then try to login
        response = await client.post(
            "/api/v1/auth/login",
            json={"phone": "1111111111", "password": "TestPass123"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "tokens" in data
        assert "access_token" in data["tokens"]
        assert "refresh_token" in data["tokens"]

    async def test_login_wrong_password(self, client: AsyncClient):
        """Test login with wrong password."""
        # First register a user
        await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "2222222222",
                "password": "CorrectPass123",
                "email": "wrong@example.com",
            },
        )

        # Try to login with wrong password
        response = await client.post(
            "/api/v1/auth/login",
            json={"phone": "2222222222", "password": "WrongPass123"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_me(self, client: AsyncClient):
        """Test getting current user profile."""
        # Register and get token
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "3333333333",
                "password": "TestPass123",
                "email": "me@example.com",
                "full_name": "Me User",
            },
        )
        access_token = register_response.json()["tokens"]["access_token"]

        # Get profile
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["phone"] == "3333333333"
        assert data["email"] == "me@example.com"
        assert data["full_name"] == "Me User"

    async def test_get_me_unauthorized(self, client: AsyncClient):
        """Test that accessing /me without token fails."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_update_profile(self, client: AsyncClient):
        """Test updating user profile."""
        # Register and get token
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "4444444444",
                "password": "TestPass123",
                "email": "old@example.com",
            },
        )
        access_token = register_response.json()["tokens"]["access_token"]

        # Update profile
        response = await client.patch(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "email": "new@example.com",
                "full_name": "New Name",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "new@example.com"
        assert data["full_name"] == "New Name"

    async def test_refresh_token(self, client: AsyncClient):
        """Test refreshing access token."""
        # Register and get tokens
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "5555555555",
                "password": "TestPass123",
            },
        )
        refresh_token = register_response.json()["tokens"]["refresh_token"]

        # Refresh token
        response = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    async def test_search_with_authentication(self, client: AsyncClient):
        """Test that authenticated search is tracked."""
        # Register and get token
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "phone": "6666666666",
                "password": "TestPass123",
            },
        )
        access_token = register_response.json()["tokens"]["access_token"]

        # Perform authenticated search
        search_response = await client.post(
            "/api/v1/search",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"query": "What is machine learning?"},
        )

        # Should work (might return empty results if no data)
        assert search_response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
        ]

        # Check search history
        history_response = await client.get(
            "/api/v1/auth/me/search-history",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert history_response.status_code == status.HTTP_200_OK
        history = history_response.json()
        assert history["total"] >= 1  # At least one search recorded

    async def test_search_without_authentication(self, client: AsyncClient):
        """Test that anonymous search works but isn't tracked."""
        # Perform anonymous search
        response = await client.post(
            "/api/v1/search",
            json={"query": "What is AI?"},
        )

        # Should work (might return empty results if no data)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
async def client():
    """Create test client.

    Note: You'll need to set up proper test fixtures for your app.
    This is just an example structure.
    """
    from httpx import AsyncClient
    from app.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

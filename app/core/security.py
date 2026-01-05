"""JWT token utilities for authentication.

Industry-standard JWT implementation using python-jose with:
- RS256 algorithm for asymmetric signing (production-ready)
- Separate access and refresh tokens
- Token expiration and validation
- Type-safe token generation and verification
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import Settings
from app.exceptions import AuthenticationError

# Password hashing context using argon2 (more secure and compatible than bcrypt)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class SecurityUtils:
    """Security utilities for password hashing and JWT operations."""

    def __init__(self, settings: Settings):
        """Initialize security utilities with settings.

        Args:
            settings: Application settings containing JWT configuration
        """
        self.settings = settings
        self.algorithm = settings.jwt_algorithm
        self.secret_key = settings.jwt_secret_key
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = settings.jwt_refresh_token_expire_days

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to compare against

        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self,
        subject: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a JWT access token.

        Args:
            subject: Subject (typically user_id as string)
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode: dict[str, Any] = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm,
        )
        return encoded_jwt

    def create_refresh_token(
        self,
        subject: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a JWT refresh token.

        Args:
            subject: Subject (typically user_id as string)
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.refresh_token_expire_days
            )

        to_encode: dict[str, Any] = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
        }

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm,
        )
        return encoded_jwt

    def create_token_pair(self, subject: str) -> dict[str, str]:
        """Create both access and refresh tokens.

        Args:
            subject: Subject (typically user_id as string)

        Returns:
            Dictionary with access_token and refresh_token
        """
        return {
            "access_token": self.create_access_token(subject),
            "refresh_token": self.create_refresh_token(subject),
        }

    def verify_token(self, token: str, expected_type: str = "access") -> dict[str, Any]:
        """Verify and decode a JWT token.

        Args:
            token: JWT token to verify
            expected_type: Expected token type ("access" or "refresh")

        Returns:
            Decoded token payload

        Raises:
            AuthenticationError: If token is invalid, expired, or wrong type
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )

            # Validate token type
            token_type = payload.get("type")
            if token_type != expected_type:
                raise AuthenticationError(
                    f"Invalid token type. Expected '{expected_type}', got '{token_type}'"
                )

            # Validate subject exists
            if not payload.get("sub"):
                raise AuthenticationError("Token missing subject (sub) claim")

            return payload

        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}") from e

    def decode_token(self, token: str) -> dict[str, Any]:
        """Decode a token without verification (for inspection only).

        Args:
            token: JWT token to decode

        Returns:
            Decoded token payload

        Raises:
            AuthenticationError: If token cannot be decoded
        """
        try:
            return jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False},
            )
        except JWTError as e:
            raise AuthenticationError(f"Cannot decode token: {str(e)}") from e

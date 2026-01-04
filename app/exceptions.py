"""Custom exception classes for the AI Processing Service."""


class AIServiceError(Exception):
    """Base exception for all AI Service errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ProcessingError(AIServiceError):
    """Raised when video processing fails."""

    pass


class FetchingServiceError(AIServiceError):
    """Raised when communication with the Fetching Service fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: str | None = None,
    ):
        self.status_code = status_code
        super().__init__(message, details)


class EmbeddingError(AIServiceError):
    """Raised when embedding generation fails."""

    pass


class QueryGuardrailError(AIServiceError):
    """Raised when query validation/transformation fails."""

    def __init__(
        self,
        message: str,
        status: str = "INVALID",
        details: str | None = None,
    ):
        self.status = status  # INVALID, INCOMPLETE
        super().__init__(message, details)


class FormattingError(AIServiceError):
    """Raised when LLM text formatting fails."""

    pass


class ChunkingError(AIServiceError):
    """Raised when text chunking fails."""

    pass


class RabbitMQError(AIServiceError):
    """Raised when RabbitMQ operations fail."""

    pass


class DatabaseError(AIServiceError):
    """Raised when database operations fail."""

    pass


class AuthenticationError(AIServiceError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(AIServiceError):
    """Raised when authorization/permission check fails."""

    pass


class UserNotFoundError(AIServiceError):
    """Raised when a user is not found."""

    pass


class DuplicateUserError(AIServiceError):
    """Raised when attempting to create a user that already exists."""

    pass

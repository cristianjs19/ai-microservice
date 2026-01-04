"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Uses pydantic-settings to validate and load configuration from
    environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Video AI Processor"
    env: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    service_name: str = "ai-processing-service"

    # Database
    database_url: str

    # RabbitMQ
    rabbitmq_url: str
    rabbitmq_queue_name: str = "transcript.fetched"
    rabbitmq_dlq_name: str = "transcript.failed"
    rabbitmq_exchange: str = "video.events"

    # Fetching Service
    fetching_service_url: str

    # OpenRouter
    openrouter_api_key: str
    openrouter_api_base: str = "https://openrouter.ai/api/v1"

    # Models
    embedding_model_name: str = "baai/bge-base-en-v1.5"
    embedding_dimensions: int = 768
    llm_model_name: str = "openai/gpt-oss-120b"
    query_guardrail_model: str = "openai/gpt-oss-120b"

    # Chunking
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # RAG
    rag_top_k_default: int = 5
    rag_similarity_threshold_default: float = 0.7

    # API
    api_v1_prefix: str = "/api/v1"
    api_title: str = "AI Processing Service"
    api_version: str = "2.1.0"

    # JWT Authentication
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


# Singleton instance for direct import
settings = get_settings()

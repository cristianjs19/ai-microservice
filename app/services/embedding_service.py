"""Embedding generation service.

This service generates vector embeddings for text chunks
using OpenRouter's embedding API.
"""

import asyncio
import logging

from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings.

    Uses OpenRouter API via LangChain to generate embeddings
    compatible with BAAI/bge-base-en-v1.5 (768 dimensions).

    Attributes:
        model_name: The embedding model to use.
        dimensions: Expected embedding dimensions.
        max_retries: Maximum retry attempts on rate limits.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
        max_retries: int = 3,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """Initialize the embedding service.

        Args:
            model_name: Embedding model name. Defaults to settings.
            dimensions: Expected embedding dimensions. Defaults to settings.
            max_retries: Max retries on rate limit errors.
            api_key: OpenRouter API key. Defaults to settings.
            api_base: OpenRouter API base URL. Defaults to settings.
        """
        self.model_name = model_name or settings.embedding_model_name
        self.dimensions = dimensions or settings.embedding_dimensions
        self.max_retries = max_retries
        self.api_key = api_key or settings.openrouter_api_key
        self.api_base = api_base or settings.openrouter_api_base

        self._embeddings: OpenAIEmbeddings | None = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Get or create the embeddings instance."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
            )
        return self._embeddings

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
        expected_count: int,
    ) -> None:
        """Validate embedding dimensions and count.

        Args:
            embeddings: List of embedding vectors.
            expected_count: Expected number of embeddings.

        Raises:
            EmbeddingError: If validation fails.
        """
        if len(embeddings) != expected_count:
            raise EmbeddingError(
                message="Embedding count mismatch",
                details=f"Expected {expected_count} embeddings, got {len(embeddings)}",
            )

        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dimensions:
                raise EmbeddingError(
                    message=f"Invalid embedding dimensions for chunk {i}",
                    details=f"Expected {self.dimensions}, got {len(embedding)}",
                )

    async def embed_documents(
        self,
        chunks: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for a list of text chunks.

        Args:
            chunks: List of text chunks to embed.

        Returns:
            list[list[float]]: List of embedding vectors (768-dimensional).

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not chunks:
            raise EmbeddingError(
                message="Cannot embed empty chunk list",
                details="No chunks provided for embedding",
            )

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Generating embeddings for {len(chunks)} chunks "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

                # Generate embeddings
                embeddings = await self.embeddings.aembed_documents(chunks)

                # Validate results
                self._validate_embeddings(embeddings, len(chunks))

                logger.info(
                    f"Successfully generated {len(embeddings)} embeddings "
                    f"({self.dimensions} dimensions each)"
                )
                return embeddings

            except EmbeddingError:
                raise
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limit errors
                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    delay = 2 ** (attempt + 1)  # Exponential backoff
                    logger.warning(
                        f"Rate limited on attempt {attempt + 1}, retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable error
                logger.error(f"Embedding generation failed: {e}")
                raise EmbeddingError(
                    message="Failed to generate embeddings",
                    details=str(e),
                )

        # All retries exhausted
        raise EmbeddingError(
            message=f"Failed to generate embeddings after {self.max_retries} attempts",
            details=str(last_error) if last_error else "Unknown error",
        )

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string.

        Args:
            query: The query text to embed.

        Returns:
            list[float]: The embedding vector (768-dimensional).

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not query or not query.strip():
            raise EmbeddingError(
                message="Cannot embed empty query",
                details="Query is empty or whitespace only",
            )

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Generating query embedding (attempt {attempt + 1})")

                embedding = await self.embeddings.aembed_query(query)

                if len(embedding) != self.dimensions:
                    raise EmbeddingError(
                        message="Invalid query embedding dimensions",
                        details=f"Expected {self.dimensions}, got {len(embedding)}",
                    )

                logger.debug("Successfully generated query embedding")
                return embedding

            except EmbeddingError:
                raise
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "rate" in error_str or "429" in error_str:
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue

                logger.error(f"Query embedding failed: {e}")
                raise EmbeddingError(
                    message="Failed to generate query embedding",
                    details=str(e),
                )

        raise EmbeddingError(
            message=f"Failed to generate query embedding after {self.max_retries} attempts",
            details=str(last_error) if last_error else "Unknown error",
        )


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

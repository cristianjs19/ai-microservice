"""Token-based text chunking service.

This service splits formatted transcripts into semantic chunks
suitable for embedding generation.
"""

import logging

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.exceptions import ChunkingError

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for splitting text into token-sized chunks.

    Uses RecursiveCharacterTextSplitter with tiktoken for accurate
    token counting, ensuring chunks fit within embedding model limits.

    Attributes:
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        encoding_name: Tiktoken encoding to use for token counting.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize the chunking service.

        Args:
            chunk_size: Target chunk size in tokens. Defaults to settings.
            chunk_overlap: Overlap between chunks in tokens. Defaults to settings.
            encoding_name: Tiktoken encoding name for token counting.
        """
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_tokens
        self.encoding_name = encoding_name

        # Initialize tiktoken encoder
        self._encoding = tiktoken.get_encoding(self.encoding_name)

        # Create the text splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

    def _token_length(self, text: str) -> int:
        """Calculate the token length of a text string.

        Args:
            text: The text to measure.

        Returns:
            int: Number of tokens in the text.
        """
        return len(self._encoding.encode(text))

    def chunk_text(self, text: str) -> list[tuple[int, str]]:
        """Split text into chunks with sequential indices.

        Args:
            text: The text to split into chunks.

        Returns:
            list[tuple[int, str]]: List of (chunk_index, content) tuples.

        Raises:
            ChunkingError: If chunking fails.
        """
        if not text or not text.strip():
            raise ChunkingError(
                message="Cannot chunk empty text",
                details="The provided text is empty or whitespace only",
            )

        try:
            # Split the text
            chunks = self._splitter.split_text(text)

            if not chunks:
                raise ChunkingError(
                    message="Text splitting produced no chunks",
                    details="The text could not be split into chunks",
                )

            # Create indexed tuples
            indexed_chunks = [(i, chunk) for i, chunk in enumerate(chunks)]

            logger.info(
                f"Split text into {len(indexed_chunks)} chunks "
                f"(avg {len(text) // len(indexed_chunks)} chars/chunk)"
            )

            # Log token counts for debugging
            for idx, content in indexed_chunks:
                token_count = self._token_length(content)
                logger.debug(f"Chunk {idx}: {token_count} tokens, {len(content)} chars")

            return indexed_chunks

        except ChunkingError:
            raise
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise ChunkingError(
                message="Failed to chunk text",
                details=str(e),
            )

    def get_token_count(self, text: str) -> int:
        """Get the token count for a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            int: Number of tokens.
        """
        return self._token_length(text)


# Singleton instance
_chunking_service: ChunkingService | None = None


def get_chunking_service() -> ChunkingService:
    """Get the global chunking service instance."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service

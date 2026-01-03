"""LLM-based transcript formatting service (AI Agent 1).

This service uses an LLM to format raw video transcripts into
human-readable articles without summarization.
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.exceptions import FormattingError

logger = logging.getLogger(__name__)

# System prompt for transcript formatting
FORMATTING_SYSTEM_PROMPT = """You are an expert transcript editor. Your task is to format raw video captions into a highly readable, engaging article format.

Follow these rules strictly:

1. NO SUMMARIZATION: Retain all original information. Do not omit or condense any content.
2. PARAGRAPH BREAKS: Insert logical paragraph breaks (\\n\\n) where topics shift or after natural pauses in thought.
3. PUNCTUATION: Fix run-on sentences, add proper commas, periods, and other punctuation.
4. CLEAN UP: Remove verbal fillers ("uh", "um", "you know", "like") if they disrupt readability.
5. FORMATTING: Keep as prose paragraphs only. Do not add headers, bullet points, or numbered lists.
6. PRESERVE MEANING: Maintain the speaker's original voice, tone, and intended meaning.
7. SPEAKER LABELS: If multiple speakers are identified, preserve speaker distinctions.

Output only the formatted transcript text without any preamble or explanation."""


class FormattingService:
    """Service for formatting raw transcripts using LLM.

    Uses OpenRouter API via LangChain to transform raw captions
    into well-formatted, readable text.

    Attributes:
        model_name: The LLM model to use for formatting.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on rate limits.
    """

    def __init__(
        self,
        model_name: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """Initialize the formatting service.

        Args:
            model_name: LLM model name. Defaults to settings.
            timeout: Request timeout in seconds.
            max_retries: Max retries on rate limit errors.
            api_key: OpenRouter API key. Defaults to settings.
            api_base: OpenRouter API base URL. Defaults to settings.
        """
        self.model_name = model_name or settings.llm_model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key or settings.openrouter_api_key
        self.api_base = api_base or settings.openrouter_api_base

        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                timeout=self.timeout,
                max_retries=0,  # We handle retries ourselves
            )
        return self._llm

    async def format_transcript(self, raw_transcript: str) -> str:
        """Format a raw transcript into readable text.

        Args:
            raw_transcript: The raw, unformatted transcript text.

        Returns:
            str: The formatted transcript text.

        Raises:
            FormattingError: If formatting fails after all retries.
        """
        if not raw_transcript or not raw_transcript.strip():
            raise FormattingError(
                message="Cannot format empty transcript",
                details="The provided transcript is empty or whitespace only",
            )

        messages = [
            SystemMessage(content=FORMATTING_SYSTEM_PROMPT),
            HumanMessage(content=raw_transcript),
        ]

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Formatting transcript (attempt {attempt + 1}/{self.max_retries}), "
                    f"length: {len(raw_transcript)} chars"
                )

                response = await self.llm.ainvoke(messages)
                formatted_text = response.content

                if not formatted_text or not formatted_text.strip():
                    raise FormattingError(
                        message="LLM returned empty response",
                        details="The formatting model returned no content",
                    )

                logger.info(
                    f"Successfully formatted transcript: "
                    f"{len(raw_transcript)} → {len(formatted_text)} chars"
                )
                return formatted_text.strip()

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limit errors
                if "rate" in error_str or "429" in error_str or "limit" in error_str:
                    delay = 2 ** (attempt + 1)  # Exponential backoff: 2, 4, 8
                    logger.warning(
                        f"Rate limited on attempt {attempt + 1}, "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                # Check for timeout
                if "timeout" in error_str:
                    logger.warning(
                        f"Timeout on attempt {attempt + 1}, retrying..."
                    )
                    await asyncio.sleep(1)
                    continue

                # Non-retryable error
                logger.error(f"Formatting failed: {e}")
                raise FormattingError(
                    message="Failed to format transcript",
                    details=str(e),
                )

        # All retries exhausted
        raise FormattingError(
            message=f"Failed to format transcript after {self.max_retries} attempts",
            details=str(last_error) if last_error else "Unknown error",
        )


# Singleton instance
_formatting_service: FormattingService | None = None


def get_formatting_service() -> FormattingService:
    """Get the global formatting service instance."""
    global _formatting_service
    if _formatting_service is None:
        _formatting_service = FormattingService()
    return _formatting_service

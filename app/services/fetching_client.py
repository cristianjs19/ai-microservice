"""HTTP client for communicating with the Fetching Service (yt-scraper)."""

import asyncio
import logging
from typing import Any

import httpx

from app.config import settings
from app.exceptions import FetchingServiceError

logger = logging.getLogger(__name__)


class FetchingServiceClient:
    """HTTP client wrapper for calling Fetching Service REST API.

    Implements exponential backoff retry logic for transient failures.

    Attributes:
        base_url: The base URL of the Fetching Service.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        max_retries: int = 4,
        base_delay: float = 2.0,
        timeout: float = 30.0,
    ):
        """Initialize the Fetching Service client.

        Args:
            base_url: Base URL of the Fetching Service. Defaults to settings.
            max_retries: Maximum retry attempts for transient failures.
            base_delay: Base delay for exponential backoff.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url or settings.fetching_service_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client instance."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def __aenter__(self) -> "FetchingServiceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path.
            **kwargs: Additional arguments for httpx request.

        Returns:
            httpx.Response: The successful response.

        Raises:
            FetchingServiceError: If all retries fail or non-retryable error occurs.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, endpoint, **kwargs)

                # Don't retry on 4xx errors (client errors)
                if 400 <= response.status_code < 500:
                    if response.status_code == 404:
                        raise FetchingServiceError(
                            message=f"Resource not found: {endpoint}",
                            status_code=404,
                            details="The requested video does not exist in the Fetching Service",
                        )
                    raise FetchingServiceError(
                        message=f"Client error: {response.status_code}",
                        status_code=response.status_code,
                        details=response.text,
                    )

                # Raise for 5xx errors to trigger retry
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500:
                    delay = self.base_delay * (2**attempt)
                    logger.warning(
                        f"Server error {e.response.status_code} on attempt {attempt + 1}, "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    raise FetchingServiceError(
                        message=f"HTTP error: {e.response.status_code}",
                        status_code=e.response.status_code,
                        details=str(e),
                    )

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                delay = self.base_delay * (2**attempt)
                logger.warning(
                    f"Connection error on attempt {attempt + 1}: {e}, "
                    f"retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Unexpected error calling Fetching Service: {e}")
                raise FetchingServiceError(
                    message="Unexpected error communicating with Fetching Service",
                    details=str(e),
                )

        # All retries exhausted
        raise FetchingServiceError(
            message=f"Failed after {self.max_retries} attempts",
            details=str(last_exception) if last_exception else "Unknown error",
        )

    async def get_video_metadata(self, video_id: str) -> dict[str, Any]:
        """Fetch complete video metadata including channel info.

        Args:
            video_id: The YouTube video ID.

        Returns:
            dict: Video metadata including title, channel info, duration, etc.

        Raises:
            FetchingServiceError: If the request fails.
        """
        logger.debug(f"Fetching metadata for video: {video_id}")
        response = await self._request_with_retry("GET", f"/api/v1/videos/{video_id}")
        data = response.json()
        logger.info(f"Successfully fetched metadata for video: {video_id}")
        return data

    async def get_video_caption(self, video_id: str) -> str:
        """Fetch raw transcript text only.

        Args:
            video_id: The YouTube video ID.

        Returns:
            str: The raw transcript/caption text.

        Raises:
            FetchingServiceError: If the request fails.
        """
        logger.debug(f"Fetching caption for video: {video_id}")
        response = await self._request_with_retry(
            "GET", f"/api/v1/videos/{video_id}/caption"
        )
        caption = response.text
        logger.info(
            f"Successfully fetched caption for video: {video_id} "
            f"({len(caption)} characters)"
        )
        return caption

    async def check_health(self) -> bool:
        """Check if the Fetching Service is healthy.

        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Fetching Service health check failed: {e}")
            return False

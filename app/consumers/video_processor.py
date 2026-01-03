"""RabbitMQ consumer for processing video transcript events."""

import asyncio
import json
import logging
from typing import Any, Callable

import aio_pika
from aio_pika import Message
from aio_pika.abc import AbstractChannel, AbstractConnection, AbstractQueue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_session_context
from app.exceptions import RabbitMQError
from app.models import ProcessingStatus, VideoDocument

logger = logging.getLogger(__name__)


class VideoProcessorConsumer:
    """RabbitMQ consumer that listens for transcript.fetched events.

    Processes incoming messages containing video IDs and triggers
    the video processing pipeline.

    Attributes:
        rabbitmq_url: The RabbitMQ connection URL.
        queue_name: Name of the queue to consume from.
        dlq_name: Name of the dead-letter queue.
        exchange_name: Name of the exchange.
    """

    def __init__(
        self,
        rabbitmq_url: str | None = None,
        queue_name: str | None = None,
        dlq_name: str | None = None,
        exchange_name: str | None = None,
    ):
        """Initialize the video processor consumer.

        Args:
            rabbitmq_url: RabbitMQ connection URL. Defaults to settings.
            queue_name: Queue name to consume from. Defaults to settings.
            dlq_name: Dead-letter queue name. Defaults to settings.
            exchange_name: Exchange name. Defaults to settings.
        """
        self.rabbitmq_url = rabbitmq_url or settings.rabbitmq_url
        self.queue_name = queue_name or settings.rabbitmq_queue_name
        self.dlq_name = dlq_name or settings.rabbitmq_dlq_name
        self.exchange_name = exchange_name or settings.rabbitmq_exchange

        self._connection: AbstractConnection | None = None
        self._channel: AbstractChannel | None = None
        self._queue: AbstractQueue | None = None
        self._dlq: AbstractQueue | None = None
        self._is_running = False
        self._process_callback: Callable[[str], Any] | None = None

    async def connect(self) -> None:
        """Establish connection to RabbitMQ and set up queues.

        Creates the main queue, dead-letter queue, and exchange bindings.

        Raises:
            RabbitMQError: If connection or setup fails.
        """
        try:
            logger.info(f"Connecting to RabbitMQ at {self.rabbitmq_url}")
            self._connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                client_properties={"connection_name": "ai-service-consumer"},
            )
            self._channel = await self._connection.channel()

            # Set prefetch count for fair dispatch
            await self._channel.set_qos(prefetch_count=1)

            # Declare the exchange
            exchange = await self._channel.declare_exchange(
                self.exchange_name,
                aio_pika.ExchangeType.TOPIC,
                durable=True,
            )

            # Declare dead-letter queue first
            self._dlq = await self._channel.declare_queue(
                self.dlq_name,
                durable=True,
            )

            # Declare main queue with dead-letter configuration
            self._queue = await self._channel.declare_queue(
                self.queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": self.dlq_name,
                },
            )

            # Bind queue to exchange with routing key
            await self._queue.bind(exchange, routing_key=self.queue_name)

            logger.info(
                f"Connected to RabbitMQ. Queue: {self.queue_name}, "
                f"DLQ: {self.dlq_name}, Exchange: {self.exchange_name}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise RabbitMQError(
                message="Failed to connect to RabbitMQ",
                details=str(e),
            )

    async def disconnect(self) -> None:
        """Close the RabbitMQ connection gracefully."""
        self._is_running = False

        if self._channel is not None:
            await self._channel.close()
            self._channel = None

        if self._connection is not None:
            await self._connection.close()
            self._connection = None

        logger.info("Disconnected from RabbitMQ")

    async def _check_duplicate(
        self,
        session: AsyncSession,
        video_id: str,
    ) -> bool:
        """Check if a video has already been processed.

        Args:
            session: Database session.
            video_id: The source video ID to check.

        Returns:
            bool: True if video already exists, False otherwise.
        """
        result = await session.execute(
            select(VideoDocument.id).where(VideoDocument.source_video_id == video_id)
        )
        return result.scalar_one_or_none() is not None

    async def _publish_to_dlq(
        self,
        video_id: str,
        error_message: str,
        attempt_count: int,
    ) -> None:
        """Publish a failed message to the dead-letter queue.

        Args:
            video_id: The video ID that failed processing.
            error_message: Description of the failure.
            attempt_count: Number of processing attempts made.
        """
        if self._channel is None:
            logger.error("Cannot publish to DLQ: channel is None")
            return

        try:
            message_body = json.dumps({
                "video_id": video_id,
                "error_message": error_message,
                "attempt_count": attempt_count,
            })

            await self._channel.default_exchange.publish(
                Message(
                    body=message_body.encode(),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self.dlq_name,
            )
            logger.info(f"Published failed message for {video_id} to DLQ")

        except Exception as e:
            logger.error(f"Failed to publish to DLQ: {e}")

    async def _handle_message(
        self,
        message: aio_pika.abc.AbstractIncomingMessage,
    ) -> None:
        """Handle an incoming RabbitMQ message.

        Parses the message, checks for duplicates, and triggers processing.

        Args:
            message: The incoming RabbitMQ message.
        """
        async with message.process(requeue=False):
            try:
                # Parse message body
                body = json.loads(message.body.decode())
                video_id = body.get("video_id")

                if not video_id:
                    logger.warning("Received message without video_id, skipping")
                    return

                logger.info(f"Received message for video: {video_id}")

                # Check for duplicates
                async with get_session_context() as session:
                    if await self._check_duplicate(session, video_id):
                        logger.info(f"Video {video_id} already processed, skipping")
                        return

                # Trigger processing pipeline
                if self._process_callback is not None:
                    await self._process_callback(video_id)
                else:
                    # Placeholder: Log the video_id for now
                    # Full processing pipeline will be implemented in Phase 6
                    logger.info(
                        f"Would process video: {video_id} "
                        "(processing pipeline not yet implemented)"
                    )

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}")
                await self._publish_to_dlq(
                    video_id="unknown",
                    error_message=f"Invalid JSON: {e}",
                    attempt_count=1,
                )

            except Exception as e:
                video_id_str = video_id if "video_id" in locals() else "unknown"
                logger.error(f"Error processing message for {video_id_str}: {e}")
                await self._publish_to_dlq(
                    video_id=video_id_str,
                    error_message=str(e),
                    attempt_count=1,
                )

    def set_process_callback(
        self,
        callback: Callable[[str], Any],
    ) -> None:
        """Set the callback function for video processing.

        Args:
            callback: Async function that takes a video_id and processes it.
        """
        self._process_callback = callback

    async def start_consuming(self) -> None:
        """Start consuming messages from the queue.

        This method blocks and continues consuming until stop() is called.
        """
        if self._queue is None:
            raise RabbitMQError(
                message="Cannot start consuming: not connected",
                details="Call connect() before start_consuming()",
            )

        self._is_running = True
        logger.info(f"Starting to consume from queue: {self.queue_name}")

        async with self._queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self._is_running:
                    break
                await self._handle_message(message)

    async def run(self) -> None:
        """Connect and start consuming messages.

        Convenience method that combines connect() and start_consuming().
        """
        await self.connect()
        try:
            await self.start_consuming()
        finally:
            await self.disconnect()

    def stop(self) -> None:
        """Signal the consumer to stop processing messages."""
        self._is_running = False
        logger.info("Consumer stop requested")

    async def check_connection(self) -> bool:
        """Check if the RabbitMQ connection is healthy.

        Returns:
            bool: True if connected, False otherwise.
        """
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
        )


# Consumer singleton for use in FastAPI lifespan
_consumer: VideoProcessorConsumer | None = None


def get_consumer() -> VideoProcessorConsumer:
    """Get the global consumer instance."""
    global _consumer
    if _consumer is None:
        _consumer = VideoProcessorConsumer()
    return _consumer


async def start_consumer_background() -> asyncio.Task:
    """Start the consumer in the background.

    Returns:
        asyncio.Task: The background task running the consumer.
    """
    consumer = get_consumer()
    await consumer.connect()

    task = asyncio.create_task(consumer.start_consuming())
    logger.info("Consumer background task started")
    return task


async def stop_consumer() -> None:
    """Stop the global consumer instance."""
    global _consumer
    if _consumer is not None:
        _consumer.stop()
        await _consumer.disconnect()
        _consumer = None

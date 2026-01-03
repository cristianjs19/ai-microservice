"""RabbitMQ message consumers."""

from app.consumers.video_processor import VideoProcessorConsumer

__all__ = ["VideoProcessorConsumer"]

"""FastAPI application initialization and lifespan events."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.config import settings
from app.consumers.video_processor import (
    get_consumer,
    start_consumer_background,
    stop_consumer,
)
from app.core.database import (
    check_database_connection,
    close_database,
    init_pgvector_extension,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if settings.log_format != "json"
    else '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Background task for consumer
_consumer_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Handles startup and shutdown events including:
    - Database connection initialization
    - PGVector extension setup
    - RabbitMQ consumer startup
    """
    global _consumer_task

    logger.info(f"Starting {settings.app_name} v{settings.api_version}")

    # Startup
    try:
        # Initialize PGVector extension
        await init_pgvector_extension()
        logger.info("Database initialized")

        # Start RabbitMQ consumer in background
        _consumer_task = await start_consumer_background()
        logger.info("RabbitMQ consumer started")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Stop consumer
    if _consumer_task is not None:
        await stop_consumer()
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass
        logger.info("RabbitMQ consumer stopped")

    # Close database connections
    await close_database()
    logger.info("Database connections closed")

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns basic health status of the service.
    """
    return {"status": "healthy", "service": settings.service_name}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint.

    Verifies that all dependencies (database, RabbitMQ) are connected.
    """
    # Check database connection
    db_healthy = await check_database_connection()

    # Check RabbitMQ connection
    consumer = get_consumer()
    rabbitmq_healthy = await consumer.check_connection()

    is_ready = db_healthy and rabbitmq_healthy

    return {
        "ready": is_ready,
        "checks": {
            "database": "connected" if db_healthy else "disconnected",
            "rabbitmq": "connected" if rabbitmq_healthy else "disconnected",
        },
    }


# Register API routers
app.include_router(api_router)

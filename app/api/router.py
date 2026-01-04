"""Main API router aggregating all routes."""

from fastapi import APIRouter

from app.api.v1 import auth
from app.api.v1.endpoints import search, stats
from app.config import settings

# Create the main API router
api_router = APIRouter()

# Include authentication endpoints
api_router.include_router(
    auth.router,
    prefix=settings.api_v1_prefix,
)

# Include v1 endpoints
api_router.include_router(
    search.router,
    prefix=settings.api_v1_prefix,
    tags=["search"],
)

api_router.include_router(
    stats.router,
    prefix=settings.api_v1_prefix,
    tags=["stats"],
)

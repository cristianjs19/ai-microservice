"""API v1 module."""

from app.api.v1 import auth
from app.api.v1.endpoints import search, stats, videos

__all__ = ["auth", "search", "stats", "videos"]

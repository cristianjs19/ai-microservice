"""SQLAlchemy declarative base class for AI Service."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models.

    Includes common configuration for Alembic and metadata.
    """

    pass

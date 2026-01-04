"""Logging configuration helpers for the FastAPI application."""

from __future__ import annotations

import logging
from logging.config import dictConfig
from pathlib import Path

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structured console and file logging.

    The configuration mirrors the Django-style dictConfig the project previously
    used, minus ANSI color formatting. Console output follows the standard
    formatter and the file handler uses a timed rotation to keep up to five
    backups. Uvicorn loggers are aligned with the application formatters to keep
    a single, consistent log format.
    """

    log_level = settings.log_level.upper()
    file_log_level = "DEBUG" if settings.debug else log_level

    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "format": "[%(asctime)s] %(levelname)s [uvicorn.access] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": file_log_level,
                "formatter": "standard",
                "filename": str(log_file),
                "when": "midnight",
                "interval": 1,
                "backupCount": 5,
                "utc": True,
            },
            "uvicorn_access": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "access",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "handlers": ["console", "file"],
                "level": "WARNING",
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["uvicorn_access", "file"],
                "level": log_level,
                "propagate": False,
            },
        },
    }

    dictConfig(logging_config)
    logging.captureWarnings(True)

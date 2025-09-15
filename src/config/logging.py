from __future__ import annotations

import logging
from logging.config import dictConfig

from src.config.settings import settings


def setup_logging() -> None:
    level = settings.log_level

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
        }
    )

    logging.getLogger(__name__).debug("Logging configured", extra={"level": level})

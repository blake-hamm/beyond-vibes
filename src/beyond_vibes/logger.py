"""Logging configuration."""

import logging
import sys

from beyond_vibes.settings import settings


def configure_logging() -> None:
    """Configure the root logger for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        root_logger.addHandler(handler)

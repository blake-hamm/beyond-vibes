"""Logging configuration."""

import logging
import sys

logger = logging.getLogger("beyond_vibes")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger.addHandler(handler)
logger.setLevel(logging.INFO)

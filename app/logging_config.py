from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(name: str = "xtrader") -> logging.Logger:
    """
    Global app logger.
    - LOG_LEVEL in env: DEBUG/INFO/WARNING/ERROR
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers during uvicorn reload
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Create a single global logger instance for import anywhere
logger: logging.Logger = setup_logging("xtrader")

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False

def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=lvl, format=fmt)
    _CONFIGURED = True

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging(os.getenv("NWM_LOG_LEVEL", "INFO"))
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

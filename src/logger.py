"""
src/logger.py
=============
Centralised logging configuration for the entire project.

Usage in any module:
    from .logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
    logger.warning("something odd")
    logger.error("something failed")

Log files are written to logs/ directory with a timestamp in the filename.
Console output mirrors the file output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Single session log file (shared across all modules) ───────────────────────
_SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"run_{_SESSION_TIMESTAMP}.log"

# ── Format ────────────────────────────────────────────────────────────────────
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Root logger — configure once ─────────────────────────────────────────────
_configured = False

def _configure_root():
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    # File handler — DEBUG and above (captures everything)
    file_h = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(file_h)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring handlers on first call."""
    _configure_root()
    return logging.getLogger(name)
"""
paper_trading/logger.py — Rotating file + console logger for the paper trader.

Usage
-----
    from paper_trading.logger import get_logger
    log = get_logger()
    log.info("Trade opened: AVAXUSDT long @ 22.45")

The log rotates at 10 MB, keeps 5 backup files.
Console handler shows INFO and above; file handler shows DEBUG and above.
"""

import logging
import logging.handlers
import os

from paper_trading.config_live import LOG_FILE


def get_logger(name: str = "paper_trader") -> logging.Logger:
    """
    Return a named logger with rotating file + console handlers.

    Safe to call multiple times with the same name — handlers are added only
    once (idempotent via hasHandlers() check).
    """
    log = logging.getLogger(name)

    if log.hasHandlers():
        return log   # already configured in this process

    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Rotating file handler ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)

    return log

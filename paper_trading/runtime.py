"""
paper_trading/runtime.py — Shared runtime flags for graceful shutdown.

The poll loop checks `is_shutdown_requested()` between ticks; signal handlers
installed in paper_trader.py call `request_shutdown()` on SIGINT / SIGTERM.

This module is intentionally tiny and dependency-free so it can be imported
from both the top-level launcher and the polling loop without circular refs.
"""

import logging

log = logging.getLogger(__name__)

_shutdown_requested = False
_shutdown_reason: str = ""


def request_shutdown(reason: str = "external") -> None:
    """
    Signal the poll loop to stop after the current tick.

    Safe to call from inside a signal handler — only sets a module-level flag,
    does no I/O.  The flag is checked by `is_shutdown_requested()` from the
    main thread between iterations.
    """
    global _shutdown_requested, _shutdown_reason
    if not _shutdown_requested:
        _shutdown_requested = True
        _shutdown_reason    = reason
        # Use print + log here: signal handlers may run before logging is fully
        # initialised, and we want at least *some* trace if log.info fails.
        try:
            log.warning("Shutdown requested (reason=%s)", reason)
        except Exception:
            pass


def is_shutdown_requested() -> bool:
    """Returns True once request_shutdown() has been called."""
    return _shutdown_requested


def shutdown_reason() -> str:
    """Returns the reason string passed to the first request_shutdown() call."""
    return _shutdown_reason


def reset() -> None:
    """Reset the shutdown flag.  Used only by tests; not for normal operation."""
    global _shutdown_requested, _shutdown_reason
    _shutdown_requested = False
    _shutdown_reason    = ""

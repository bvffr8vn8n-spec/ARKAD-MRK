"""
paper_trading/scheduler.py — Bar-close-aligned wake-up scheduler.

Replaces the old "sleep POLL_INTERVAL_S seconds and check" polling loop.  Both
1H and 15m bars close on the 15-minute grid (HH:00, :15, :30, :45 UTC), so a
single scheduler aimed at the 15m grid covers both.

Cadence per hour
----------------
  Old (POLL_INTERVAL_S=90s):  ~40 wake-ups/hour, ~960 API hits/day across 4 assets
  New (15m grid):              4 wake-ups/hour, ~96 idle wake-ups + per-bar work

The scheduler injects a randomised grace period (5–15s) after each boundary so
that Bybit has time to publish the just-closed bar, and a small jitter (0–5s)
on top to de-align this instance from other clients waking at the same mark
(the original cause of `retCode 10006` bursts at HH:00).

A `MAX_WAKE_SECONDS` heartbeat cap prevents pathological long sleeps if the
clock or grid math drifts; in normal operation the cap is never hit because
boundaries are at most 900s apart.
"""

import random
from datetime import datetime, timedelta

# Grace window: seconds AFTER a bar boundary before we wake to fetch.
# Bybit usually publishes the closed bar within ~1s; 5s is conservative.
GRACE_MIN_S = 5.0
GRACE_MAX_S = 15.0

# Jitter on top of grace: spreads concurrent clients across a 5s window.
JITTER_MAX_S = 5.0

# Heartbeat safety cap.  Boundaries are 900s apart (15min), so this should
# never bind in practice — it only protects against a clock anomaly.
MAX_WAKE_SECONDS = 1000.0

# Minimum sleep — if we're already past the boundary + grace (e.g., tick took
# longer than expected), still yield for at least this many seconds before
# the next iteration to avoid spinning.
MIN_WAKE_SECONDS = 1.0


def seconds_until_next_bar_wake(now: datetime) -> float:
    """
    Return the number of seconds to sleep until the next 15m bar boundary
    plus a randomised grace + jitter offset.

    Example: now = 14:07:23 → next boundary = 14:15:00, grace = 8.4s,
    jitter = 2.1s → sleep ~488 seconds.
    """
    minute_floor   = (now.minute // 15) * 15
    current_anchor = now.replace(minute=minute_floor, second=0, microsecond=0)
    next_boundary  = current_anchor + timedelta(minutes=15)

    grace  = random.uniform(GRACE_MIN_S, GRACE_MAX_S)
    jitter = random.uniform(0.0, JITTER_MAX_S)
    wake_at = next_boundary + timedelta(seconds=grace + jitter)

    seconds = (wake_at - now).total_seconds()
    return max(MIN_WAKE_SECONDS, min(seconds, MAX_WAKE_SECONDS))


def next_wake_label(now: datetime) -> str:
    """Human-readable label for the next bar boundary (no jitter)."""
    minute_floor   = (now.minute // 15) * 15
    current_anchor = now.replace(minute=minute_floor, second=0, microsecond=0)
    next_boundary  = current_anchor + timedelta(minutes=15)
    return next_boundary.isoformat(timespec="seconds")

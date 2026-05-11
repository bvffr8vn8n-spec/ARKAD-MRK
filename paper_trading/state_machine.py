"""
paper_trading/state_machine.py — 15m A+B execution state machine per asset.

Mirrors the logic in features/execution_15m.py but operates bar-by-bar in real
time rather than in a batch vectorised pass over historical data.

State transitions
-----------------
IDLE
  → 1H signal fires
  → A_WATCHING  (start counting 15m bars for Approach A filter)

A_WATCHING (up to A_FILTER_BARS=4 bars)
  → per 15m bar: increment seen / aligned counts
  → after A_FILTER_BARS bars:
      aligned >= A_FILTER_MIN_ALIGNED  → B_WATCHING  (A passed)
      aligned <  A_FILTER_MIN_ALIGNED  → CANCELLED   (A blocked)

B_WATCHING (waiting for pullback, up to B_WAIT_BARS=8 bars)
  → per 15m bar:
      if counter-move past ref_price → B_PULLBACK
      if B_WAIT_BARS bars elapsed without pullback → ENTERED (fallback at signal close)

B_PULLBACK (pullback seen, waiting for resumption)
  → per 15m bar:
      if bar closes in signal direction → ENTERED (pullback entry)
      if B_WAIT_BARS bars elapsed       → ENTERED (fallback)

ENTERED  — trade is open; state machine is done
CANCELLED — A blocked; no trade

Persistence
-----------
MonitorState is a plain dict that can be serialised to / from JSON without any
special encoding.  Datetimes are stored as ISO strings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from paper_trading import config_live
from paper_trading.logger import get_logger

log = get_logger()


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class MonitorState:
    """
    Full state of the 15m A+B execution tracker for one asset.

    Stored and reloaded as a JSON dict between restarts.
    Timestamps are ISO strings (naive UTC) for JSON portability.
    """
    asset:              str
    signal:             int         # +1 long, -1 short
    signal_ts:          str         # ISO timestamp of 1H signal bar open (naive UTC)
    signal_close:       float       # 1H close price at signal time (ref_price for B)
    atr_1h:             float       # ATR in dollar terms at signal bar (atr_pct × close)

    phase: str = "A_WATCHING"       # A_WATCHING | B_WATCHING | B_PULLBACK | ENTERED | CANCELLED

    # A filter counters
    a_bars_seen:  int   = 0
    a_aligned:    int   = 0

    # B filter counters
    b_bars_seen:  int   = 0
    b_pullback_seen: bool = False

    # A filter result (set when A window closes)
    a_filter_result: str = ""       # "pass" | "fail" | "" (not yet evaluated)
    a_aligned_bars:  int = 0        # final aligned count recorded for the trade log

    # Entry (set when ENTERED)
    entry_price: Optional[float] = None
    entry_ts:    Optional[str]   = None   # ISO timestamp

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorState":
        """
        Rebuild a MonitorState from its JSON dict.

        Unknown keys are ignored (forward compat) and missing keys fall back
        to dataclass defaults (backward compat).  This makes restart recovery
        robust across schema migrations.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── State machine update ──────────────────────────────────────────────────────

def advance_monitor(state: MonitorState, bar: pd.Series) -> MonitorState:
    """
    Process one 15m bar through the state machine and return the updated state.

    `bar` must be a pd.Series with open, high, low, close fields.
    The bar's index (name) is a pd.Timestamp (naive UTC).

    This function is pure: it returns a modified copy.  The caller is responsible
    for persisting the returned state.
    """
    import copy
    state = copy.copy(state)   # shallow copy is fine; all fields are primitives

    phase = state.phase
    if phase in ("ENTERED", "CANCELLED"):
        return state   # terminal state

    bar_ts    = bar.name.isoformat() if hasattr(bar.name, "isoformat") else str(bar.name)
    bar_close = float(bar["close"])
    bar_open  = float(bar["open"])
    direction = state.signal   # +1 or -1

    # ── A_WATCHING ────────────────────────────────────────────────────────────
    if phase == "A_WATCHING":
        state.a_bars_seen += 1

        # Aligned = bar closes in signal direction
        if direction == 1 and bar_close > bar_open:
            state.a_aligned += 1
        elif direction == -1 and bar_close < bar_open:
            state.a_aligned += 1

        if state.a_bars_seen >= config_live.A_FILTER_BARS:
            # A window is complete — evaluate
            if state.a_aligned >= config_live.A_FILTER_MIN_ALIGNED:
                state.a_filter_result = "pass"
                state.a_aligned_bars  = state.a_aligned
                state.phase           = "B_WATCHING"
                log.debug(
                    "%s A-filter PASS  aligned=%d/%d  → B_WATCHING",
                    state.asset, state.a_aligned, state.a_bars_seen,
                )
            else:
                state.a_filter_result = "fail"
                state.a_aligned_bars  = state.a_aligned
                state.phase           = "CANCELLED"
                log.info(
                    "%s A-filter FAIL  aligned=%d/%d  → CANCELLED",
                    state.asset, state.a_aligned, state.a_bars_seen,
                )
        return state

    # ── B_WATCHING ────────────────────────────────────────────────────────────
    if phase == "B_WATCHING":
        state.b_bars_seen += 1
        ref = state.signal_close

        # Detect counter-move past ref_price → transition to B_PULLBACK
        if direction == 1 and bar_close < ref:
            state.b_pullback_seen = True
            state.phase           = "B_PULLBACK"
            log.debug("%s pullback detected @ %.6f  → B_PULLBACK", state.asset, bar_close)
            # The pullback bar itself is not a resumption candidate — wait for next bar
            return state

        if direction == -1 and bar_close > ref:
            state.b_pullback_seen = True
            state.phase           = "B_PULLBACK"
            log.debug("%s pullback detected @ %.6f  → B_PULLBACK", state.asset, bar_close)
            return state

        # No pullback — check B timeout
        if state.b_bars_seen >= config_live.B_WAIT_BARS:
            _enter(state, price=state.signal_close, ts=bar_ts, mode="fallback")
        return state

    # ── B_PULLBACK ────────────────────────────────────────────────────────────
    if phase == "B_PULLBACK":
        state.b_bars_seen += 1

        # Detect resumption: bar closes in signal direction
        if direction == 1 and bar_close > bar_open:
            _enter(state, price=bar_close, ts=bar_ts, mode="pullback")
        elif direction == -1 and bar_close < bar_open:
            _enter(state, price=bar_close, ts=bar_ts, mode="pullback")
        elif state.b_bars_seen >= config_live.B_WAIT_BARS:
            _enter(state, price=state.signal_close, ts=bar_ts, mode="fallback")
        return state

    return state


# ── Helper ────────────────────────────────────────────────────────────────────

def _enter(state: MonitorState, price: float, ts: str, mode: str) -> None:
    """Set the state to ENTERED with the given entry price and timestamp."""
    state.entry_price = price
    state.entry_ts    = ts
    state.phase       = "ENTERED"
    log.info(
        "%s ENTERED  dir=%+d  price=%.6f  mode=%s",
        state.asset, state.signal, price, mode,
    )

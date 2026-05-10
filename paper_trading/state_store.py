"""
paper_trading/state_store.py — Persistent JSON state store for the paper trader.

Stores
------
  processed_1h_bars   : last processed 1H bar timestamp per asset (ISO string)
  processed_15m_bars  : last processed 15m bar timestamp per asset (ISO string)
  monitors            : active MonitorState dicts per asset (may be absent if no monitor)
  open_trades         : open TradeState dicts per asset (may be absent if no trade)
  trade_id_counter    : monotonically increasing integer for unique trade IDs
  equity              : current paper-account equity ($)

Atomicity
---------
Writes go to STATE_FILE + ".tmp", then os.replace() to the final path.
A partial write therefore never corrupts the state file.

Usage
-----
    store = StateStore()
    state = store.load()            # returns dict; creates default if missing
    state["equity"] = 9_800.0
    store.save(state)               # atomic write
"""

import json
import os
from copy import deepcopy
from typing import Any

from paper_trading import config_live
from paper_trading.logger import get_logger

log = get_logger()

_DEFAULT_STATE: dict[str, Any] = {
    "version":           1,
    "trade_id_counter":  1,
    "equity":            config_live.INITIAL_CAPITAL,
    "processed_1h_bars": {},    # asset → ISO timestamp string
    "processed_15m_bars": {},   # asset → ISO timestamp string
    "monitors":          {},    # asset → MonitorState.to_dict()
    "open_trades":       {},    # asset → TradeState.to_dict()
}


class StateStore:
    """Thin wrapper around a single JSON file with atomic write semantics."""

    def __init__(self, path: str = config_live.STATE_FILE) -> None:
        self._path     = path
        self._tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load(self) -> dict:
        """
        Read and return the state dict.

        If the file does not exist or is corrupt, return a fresh default state.

        Fresh-start guard: when no state file exists, seed processed_1h_bars and
        processed_15m_bars to the current UTC hour so the poller does NOT replay
        historical candles as if they were live events.  Without this guard every
        startup processes hundreds of historical bars, opens trades at historical
        candle prices, and logs them as live paper trades.
        """
        if not os.path.exists(self._path):
            log.info("State file not found - starting fresh: %s", self._path)
            state = deepcopy(_DEFAULT_STATE)

            # Seed bar timestamps to "now" — prevents historical-bar replay
            from datetime import datetime
            now_floor = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            now_iso   = now_floor.isoformat()
            from paper_trading.config_live import ASSETS
            for asset in ASSETS:
                state["processed_1h_bars"][asset]  = now_iso
                state["processed_15m_bars"][asset] = now_iso
            log.info(
                "Fresh-start: bar timestamps seeded to %s  (historical replay suppressed)",
                now_iso,
            )
            return state

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
            # Fill in any keys missing from older state files (forward compat)
            for key, default_val in _DEFAULT_STATE.items():
                if key not in state:
                    state[key] = deepcopy(default_val)
            log.debug("State loaded from %s  (trade_id=%d, equity=%.2f)",
                      self._path, state["trade_id_counter"], state["equity"])
            return state
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.error("State file corrupt (%s) — starting fresh.  Error: %s",
                      self._path, exc)
            return deepcopy(_DEFAULT_STATE)

    def save(self, state: dict) -> None:
        """
        Atomically write `state` to disk.

        Writes to a .tmp file first, then uses os.replace() so a mid-write
        crash leaves the previous state file intact.
        """
        try:
            with open(self._tmp_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, ensure_ascii=False)
            os.replace(self._tmp_path, self._path)
        except Exception as exc:
            log.error("Failed to save state: %s", exc, exc_info=True)
            # Clean up partial tmp file
            if os.path.exists(self._tmp_path):
                try:
                    os.remove(self._tmp_path)
                except OSError:
                    pass

    def next_trade_id(self, state: dict) -> int:
        """
        Return the next trade ID and increment the counter in `state` in-place.

        Call this inside the same critical section as state.save() to keep the
        counter consistent.
        """
        tid = int(state["trade_id_counter"])
        state["trade_id_counter"] = tid + 1
        return tid

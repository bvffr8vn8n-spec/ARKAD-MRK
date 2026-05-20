"""
paper_trading/signal_log.py — Append-only CSV log of every scored 1H signal.

Purpose
-------
The trade log (`paper_trades_tier1.csv`) only contains bars that became real
trades.  For live↔backtest parity diagnostics we need to record what the model
saw on EVERY post-cutoff 1H bar, regardless of whether a trade followed.

Schema
------
    bar_ts            ISO timestamp of the 1H bar that was scored (UTC)
    asset             symbol
    signal            -1 (short) | 0 (flat) | +1 (long), after all filters
    buy_prob          float, from the calibrated RF
    sell_prob         float, from the calibrated RF
    atr_pct           float (ATR / close)
    close             float, bar's close price
    scored_at         ISO wall-clock timestamp when this row was written
    had_open_trade    "1" if a trade was already open at score time, else "0"
    had_active_monitor "1" if an A/B monitor was active at score time, else "0"

The trailing two flags help interpret missing trades in the parity diff: a bar
where the model emitted a signal but `had_open_trade=1` cannot have produced a
new trade even if the model and BT replay agree.

Usage
-----
    log = SignalLog()
    log.append(
        bar_ts="2026-05-18T14:00:00",
        asset="AVAXUSDT",
        signal=1, buy_prob=0.612, sell_prob=0.388,
        atr_pct=0.0123, close=22.45,
        had_open_trade=False, had_active_monitor=False,
    )
"""

import csv
import os
from datetime import datetime
from typing import Optional

from paper_trading import config_live
from paper_trading.logger import get_logger

log = get_logger()

_COLUMNS = [
    "bar_ts",
    "asset",
    "signal",
    "buy_prob",
    "sell_prob",
    "atr_pct",
    "close",
    "scored_at",
    "had_open_trade",
    "had_active_monitor",
]


class SignalLog:
    """Append-only writer for paper_trading/signal_log.csv."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or config_live.SIGNAL_LOG_FILE
        self._ensure_header()

    def append(
        self,
        bar_ts: str,
        asset: str,
        signal: int,
        buy_prob: float,
        sell_prob: float,
        atr_pct: float,
        close: float,
        had_open_trade: bool,
        had_active_monitor: bool,
    ) -> None:
        row = {
            "bar_ts":             bar_ts,
            "asset":              asset,
            "signal":             int(signal),
            "buy_prob":           f"{float(buy_prob):.8f}",
            "sell_prob":          f"{float(sell_prob):.8f}",
            "atr_pct":            f"{float(atr_pct):.10f}",
            "close":              f"{float(close):.8f}",
            "scored_at":          datetime.utcnow().isoformat(timespec="seconds"),
            "had_open_trade":     "1" if had_open_trade else "0",
            "had_active_monitor": "1" if had_active_monitor else "0",
        }
        try:
            with open(self._path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writerow(row)
        except Exception as exc:
            log.error("Failed to write signal_log row for %s @ %s: %s",
                      asset, bar_ts, exc, exc_info=True)

    def _ensure_header(self) -> None:
        if os.path.exists(self._path):
            return
        try:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writeheader()
            log.info("Created new signal log: %s", self._path)
        except Exception as exc:
            log.error("Failed to create signal log: %s", exc, exc_info=True)

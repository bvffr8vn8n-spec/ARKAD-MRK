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
    open              float, bar's open price       (added for parity diagnostics)
    high              float, bar's high price       (added for parity diagnostics)
    low               float, bar's low price        (added for parity diagnostics)
    close             float, bar's close price
    volume            float, bar's volume           (added for parity diagnostics)
    scored_at         ISO wall-clock timestamp when this row was written
    had_open_trade    "1" if a trade was already open at score time, else "0"
    had_active_monitor "1" if an A/B monitor was active at score time, else "0"

The OHLCV columns were added after the 2026-06-01 parity run showed atr_pct
diff > 1e-8 on 44/44 overlap bars at max delta 0.005 — symptomatic of small
OHLC value differences between live Bybit fetches and the CSV-sourced replay.
With raw OHLCV logged, the parity test can locate exactly which field drifts
on which bar.

The trailing two flags help interpret missing trades in the parity diff: a bar
where the model emitted a signal but `had_open_trade=1` cannot have produced a
new trade even if the model and BT replay agree.

Schema drift
------------
On startup, the existing signal_log.csv (if any) is inspected.  If its header
matches the current `_COLUMNS` list it is appended to; otherwise it is renamed
to `signal_log.csv.old[.N]` and a fresh file is created with the new schema.
This avoids a silent failure when columns are added or removed.

Usage
-----
    log = SignalLog()
    log.append(
        bar_ts="2026-05-18T14:00:00",
        asset="AVAXUSDT",
        signal=1, buy_prob=0.612, sell_prob=0.388,
        atr_pct=0.0123,
        open_=22.40, high=22.55, low=22.31, close=22.45, volume=12345.6,
        had_open_trade=False, had_active_monitor=False,
    )

`open_` (trailing underscore) is used because `open` shadows the Python
builtin used internally for file I/O.  The CSV column name is still `open`.
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
    "open",
    "high",
    "low",
    "close",
    "volume",
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
        open_: float,    # `open` is a Python builtin — trailing underscore avoids shadowing
        high: float,
        low: float,
        close: float,
        volume: float,
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
            "open":               f"{float(open_):.8f}",
            "high":               f"{float(high):.8f}",
            "low":                f"{float(low):.8f}",
            "close":              f"{float(close):.8f}",
            "volume":             f"{float(volume):.8f}",
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
        """
        Create the CSV with header row if it does not exist.  If a file is
        already present but its header doesn't match the current schema,
        archive it (rename to .old[.N]) and start fresh — silent extras-key
        failures during append would otherwise burn data.
        """
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", newline="", encoding="utf-8") as fh:
                    existing_header = next(csv.reader(fh), None)
            except Exception as exc:
                log.error("Could not read signal log header: %s", exc)
                return

            if existing_header == _COLUMNS:
                return   # schema matches, will be appended to

            # Schema drift — archive the old file so we don't crash on extra keys.
            archive = self._path + ".old"
            i = 1
            while os.path.exists(archive):
                archive = self._path + f".old.{i}"
                i += 1
            try:
                os.rename(self._path, archive)
                log.info(
                    "Signal log schema changed (cols=%d → %d).  Old file "
                    "archived as %s; starting fresh.",
                    len(existing_header) if existing_header else 0,
                    len(_COLUMNS), archive,
                )
            except Exception as exc:
                log.error("Could not archive old signal log: %s", exc)
                return

        try:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writeheader()
            log.info("Created new signal log: %s", self._path)
        except Exception as exc:
            log.error("Failed to create signal log: %s", exc, exc_info=True)

"""
paper_trading/csv_writer.py — Append closed trades to paper_trades_tier1.csv.

Each completed TradeState is written as one row to the CSV log.
If the file does not exist it is created with the correct header row.
If it already exists rows are appended without re-writing the header.

Original columns (unchanged):
    trade_id, symbol, signal_time, direction,
    a_filter_result, a_aligned_bars, b_pullback_found,
    entry_time, entry_price_fill, atr_1h,
    stop_price, tp_price,
    exit_time, exit_price, exit_reason,
    net_pnl_usd, r_multiple, mfe_r, mae_r, notes

Scaled-exit columns (populated only when exit_mode == "scaled", else empty):
    exit_mode, tp1_hit, tp2_hit, tp3_hit, be_hit,
    tp1_time, tp2_time, tp3_time, be_time,
    realized_r_scaled, partial_pnl_usd, remaining_fraction
"""

import csv
import os
from typing import Optional

from paper_trading import config_live
from paper_trading.logger import get_logger
from paper_trading.trade_manager import TradeState

log = get_logger()

_COLUMNS = [
    # ── Original columns ──────────────────────────────────────────────────────
    "trade_id",
    "symbol",
    "signal_time",
    "direction",
    "a_filter_result",
    "a_aligned_bars",
    "b_pullback_found",
    "entry_time",
    "entry_price_fill",
    "atr_1h",
    "stop_price",
    "tp_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "net_pnl_usd",
    "r_multiple",
    "mfe_r",
    "mae_r",
    # ── MFE/MAE checkpoints (added 2026-06-XX for early-kill BT sweep) ─────────
    "mfe_r_h6",
    "mae_r_h6",
    "mfe_r_h12",
    "mae_r_h12",
    "mfe_r_h18",
    "mae_r_h18",
    "notes",
    # ── Scaled exit columns ───────────────────────────────────────────────────
    "exit_mode",
    "tp1_hit",
    "tp2_hit",
    "tp3_hit",
    "be_hit",
    "tp1_time",
    "tp2_time",
    "tp3_time",
    "be_time",
    "realized_r_scaled",
    "partial_pnl_usd",
    "remaining_fraction",
]


class CsvWriter:
    """Append-only writer for paper_trades_tier1.csv."""

    def __init__(self, path: str = config_live.CSV_FILE) -> None:
        self._path = path
        self._ensure_header()

    def append(self, trade: TradeState) -> None:
        """Write one closed trade as a new row."""
        is_scaled = (getattr(trade, "exit_mode", "original") == "scaled")

        row = {
            # Original columns
            "trade_id":         trade.trade_id,
            "symbol":           trade.asset,
            "signal_time":      trade.signal_ts,
            "direction":        trade.direction,
            "a_filter_result":  trade.a_filter_result,
            "a_aligned_bars":   trade.a_aligned_bars,
            "b_pullback_found": trade.b_pullback_found,
            "entry_time":       trade.entry_ts,
            "entry_price_fill": _fmt(trade.entry_price),
            "atr_1h":           _fmt(trade.atr_1h),
            "stop_price":       _fmt(trade.stop_price),
            "tp_price":         _fmt(trade.tp_price),
            "exit_time":        trade.exit_ts or "",
            "exit_price":       _fmt(trade.exit_price),
            "exit_reason":      trade.exit_reason or "",
            "net_pnl_usd":      _fmt(trade.net_pnl_usd),
            "r_multiple":       _fmt(trade.r_multiple),
            "mfe_r":            _fmt(trade.mfe_r),
            "mae_r":            _fmt(trade.mae_r),
            "mfe_r_h6":         _fmt(getattr(trade, "mfe_r_h6", None)),
            "mae_r_h6":         _fmt(getattr(trade, "mae_r_h6", None)),
            "mfe_r_h12":        _fmt(getattr(trade, "mfe_r_h12", None)),
            "mae_r_h12":        _fmt(getattr(trade, "mae_r_h12", None)),
            "mfe_r_h18":        _fmt(getattr(trade, "mfe_r_h18", None)),
            "mae_r_h18":        _fmt(getattr(trade, "mae_r_h18", None)),
            "notes":            trade.notes,
            # Scaled exit columns
            "exit_mode":           trade.exit_mode if is_scaled else "original",
            "tp1_hit":             str(trade.tp1_hit)  if is_scaled else "",
            "tp2_hit":             str(trade.tp2_hit)  if is_scaled else "",
            "tp3_hit":             str(trade.tp3_hit)  if is_scaled else "",
            "be_hit":              str(trade.be_hit)   if is_scaled else "",
            "tp1_time":            trade.tp1_time or "" if is_scaled else "",
            "tp2_time":            trade.tp2_time or "" if is_scaled else "",
            "tp3_time":            trade.tp3_time or "" if is_scaled else "",
            "be_time":             trade.be_time  or "" if is_scaled else "",
            "realized_r_scaled":   _fmt(trade.realized_r_scaled) if is_scaled else "",
            "partial_pnl_usd":     _fmt(trade.realized_pnl_usd)  if is_scaled else "",
            "remaining_fraction":  _fmt(trade.remaining_fraction) if is_scaled else "",
        }

        try:
            with open(self._path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writerow(row)
            log.info(
                "CSV row written: trade #%d %s %s  PnL=$%.2f  R=%.2f  mode=%s",
                trade.trade_id, trade.asset, trade.direction,
                trade.net_pnl_usd or 0.0, trade.r_multiple or 0.0,
                trade.exit_mode,
            )
        except Exception as exc:
            log.error("Failed to write CSV row for trade #%d: %s",
                      trade.trade_id, exc, exc_info=True)

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_header(self) -> None:
        """
        Create the CSV with header row if it does not exist.  If a file is
        present but its header does not match the current schema, rename it
        to .old[.N] and start fresh — appending new columns to an old file
        with csv.DictWriter would raise on extras_keys and burn rows.
        """
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", newline="", encoding="utf-8") as fh:
                    existing_header = next(csv.reader(fh), None)
            except Exception as exc:
                log.error("Could not read CSV log header: %s", exc)
                return

            if existing_header == _COLUMNS:
                return   # schema matches, will be appended to

            archive = self._path + ".old"
            i = 1
            while os.path.exists(archive):
                archive = self._path + f".old.{i}"
                i += 1
            try:
                os.rename(self._path, archive)
                log.info(
                    "Trade CSV schema changed (cols=%d → %d).  Old file "
                    "archived as %s; starting fresh.",
                    len(existing_header) if existing_header else 0,
                    len(_COLUMNS), archive,
                )
            except Exception as exc:
                log.error("Could not archive old trade CSV: %s", exc)
                return

        try:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
                writer.writeheader()
            log.info("Created new CSV log: %s", self._path)
        except Exception as exc:
            log.error("Failed to create CSV log: %s", exc, exc_info=True)


def _fmt(v) -> str:
    """Format a numeric value for CSV output; empty string for None / NaN."""
    if v is None:
        return ""
    try:
        import math
        if math.isnan(float(v)):
            return ""
    except (TypeError, ValueError):
        return str(v)
    return str(v)

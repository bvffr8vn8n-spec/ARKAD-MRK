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
    # ── MFE/MAE hourly checkpoints (extended 2026-07-XX to h1..h24) ────────────
    "mfe_r_h1",
    "mae_r_h1",
    "mfe_r_h2",
    "mae_r_h2",
    "mfe_r_h4",
    "mae_r_h4",
    "mfe_r_h6",
    "mae_r_h6",
    "mfe_r_h8",
    "mae_r_h8",
    "mfe_r_h12",
    "mae_r_h12",
    "mfe_r_h18",
    "mae_r_h18",
    "mfe_r_h24",
    "mae_r_h24",
    # ── Time-to-R crossings (minutes from entry to first crossing) ────────────
    "t_to_0_3R",
    "t_to_0_5R",
    "t_to_0_65R",
    "t_to_1R",
    "t_to_1_67R",
    # ── MFE peak dynamics ─────────────────────────────────────────────────────
    "t_to_mfe_peak",
    "mae_r_before_mfe_peak",
    "t_from_peak_to_exit",
    # ── Entry-state snapshot (setup characterisation for lifetime research) ───
    "signal_buy_prob",
    "signal_sell_prob",
    "entry_atr_pct",
    "entry_vol_ratio",
    "entry_rsi",
    "entry_bb_pos",
    "entry_macd_hist",
    "entry_sma_50_ratio",
    "entry_vol_expansion",
    "entry_volume",
    "entry_trend",
    "entry_vol_regime",
    "entry_session",
    # ── Shadow multi-horizon forecasts at signal time (log-only) ──────────────
    "signal_buy_prob_h2",
    "signal_sell_prob_h2",
    "signal_buy_prob_h4",
    "signal_sell_prob_h4",
    "signal_buy_prob_h8",
    "signal_sell_prob_h8",
    "signal_buy_prob_h12",
    "signal_sell_prob_h12",
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
            # Hourly checkpoints
            "mfe_r_h1":         _fmt(getattr(trade, "mfe_r_h1",  None)),
            "mae_r_h1":         _fmt(getattr(trade, "mae_r_h1",  None)),
            "mfe_r_h2":         _fmt(getattr(trade, "mfe_r_h2",  None)),
            "mae_r_h2":         _fmt(getattr(trade, "mae_r_h2",  None)),
            "mfe_r_h4":         _fmt(getattr(trade, "mfe_r_h4",  None)),
            "mae_r_h4":         _fmt(getattr(trade, "mae_r_h4",  None)),
            "mfe_r_h6":         _fmt(getattr(trade, "mfe_r_h6",  None)),
            "mae_r_h6":         _fmt(getattr(trade, "mae_r_h6",  None)),
            "mfe_r_h8":         _fmt(getattr(trade, "mfe_r_h8",  None)),
            "mae_r_h8":         _fmt(getattr(trade, "mae_r_h8",  None)),
            "mfe_r_h12":        _fmt(getattr(trade, "mfe_r_h12", None)),
            "mae_r_h12":        _fmt(getattr(trade, "mae_r_h12", None)),
            "mfe_r_h18":        _fmt(getattr(trade, "mfe_r_h18", None)),
            "mae_r_h18":        _fmt(getattr(trade, "mae_r_h18", None)),
            "mfe_r_h24":        _fmt(getattr(trade, "mfe_r_h24", None)),
            "mae_r_h24":        _fmt(getattr(trade, "mae_r_h24", None)),
            # Time-to-R
            "t_to_0_3R":        _fmt(getattr(trade, "t_to_0_3R",  None)),
            "t_to_0_5R":        _fmt(getattr(trade, "t_to_0_5R",  None)),
            "t_to_0_65R":       _fmt(getattr(trade, "t_to_0_65R", None)),
            "t_to_1R":          _fmt(getattr(trade, "t_to_1R",    None)),
            "t_to_1_67R":       _fmt(getattr(trade, "t_to_1_67R", None)),
            # MFE peak dynamics
            "t_to_mfe_peak":         _fmt(getattr(trade, "t_to_mfe_peak",         None)),
            "mae_r_before_mfe_peak": _fmt(getattr(trade, "mae_r_before_mfe_peak", None)),
            "t_from_peak_to_exit":   _fmt(getattr(trade, "t_from_peak_to_exit",   None)),
            # Entry-state snapshot
            "signal_buy_prob":      _fmt(getattr(trade, "signal_buy_prob",      None)),
            "signal_sell_prob":     _fmt(getattr(trade, "signal_sell_prob",     None)),
            "entry_atr_pct":        _fmt(getattr(trade, "entry_atr_pct",        None)),
            "entry_vol_ratio":      _fmt(getattr(trade, "entry_vol_ratio",      None)),
            "entry_rsi":            _fmt(getattr(trade, "entry_rsi",            None)),
            "entry_bb_pos":         _fmt(getattr(trade, "entry_bb_pos",         None)),
            "entry_macd_hist":      _fmt(getattr(trade, "entry_macd_hist",      None)),
            "entry_sma_50_ratio":   _fmt(getattr(trade, "entry_sma_50_ratio",   None)),
            "entry_vol_expansion":  _fmt(getattr(trade, "entry_vol_expansion",  None)),
            "entry_volume":         _fmt(getattr(trade, "entry_volume",         None)),
            "entry_trend":          getattr(trade, "entry_trend",       "") or "",
            "entry_vol_regime":     getattr(trade, "entry_vol_regime",  "") or "",
            "entry_session":        getattr(trade, "entry_session",     "") or "",
            # Shadow multi-horizon forecasts (observational only)
            "signal_buy_prob_h2":   _fmt(getattr(trade, "signal_buy_prob_h2",   None)),
            "signal_sell_prob_h2":  _fmt(getattr(trade, "signal_sell_prob_h2",  None)),
            "signal_buy_prob_h4":   _fmt(getattr(trade, "signal_buy_prob_h4",   None)),
            "signal_sell_prob_h4":  _fmt(getattr(trade, "signal_sell_prob_h4",  None)),
            "signal_buy_prob_h8":   _fmt(getattr(trade, "signal_buy_prob_h8",   None)),
            "signal_sell_prob_h8":  _fmt(getattr(trade, "signal_sell_prob_h8",  None)),
            "signal_buy_prob_h12":  _fmt(getattr(trade, "signal_buy_prob_h12",  None)),
            "signal_sell_prob_h12": _fmt(getattr(trade, "signal_sell_prob_h12", None)),
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

"""
paper_trading/trade_manager.py — Paper trade lifecycle management.

Supports two exit modes controlled by config_live.EXIT_MODE:

  "original"  — single TP at TAKE_PROFIT_ATR_MULT × ATR (unchanged behaviour)

  "scaled"    — three partial exits with BE stop after TP1:
                  TP1 = +0.65R → close 50 %  → move stop to breakeven
                  TP2 = +1.00R → close 25 %
                  TP3 = +1.67R → close remaining 25 %  (≈ original TP)
                BE stop: if price retraces to entry after TP1 → close remaining
                Time exit: closes remaining fraction at live bid/ask

Exit priority (original mode, matches backtest/engine.py):
    Long:   stop (low≤SL) → tp (high≥TP) → time (elapsed≥HOLD_HOURS)
    Short:  stop (high≥SL) → tp (low≤TP) → time

Scaled mode priority per bar:
    1. BE stop  (only active after TP1, before TP3)
    2. Next TP  (TP1 → TP2 → TP3, in sequence; multiple can fire same bar)
    3. Time exit for remaining fraction
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from paper_trading import config_live
from paper_trading.logger import get_logger
from paper_trading.state_machine import MonitorState

log = get_logger()


# ── TradeState ────────────────────────────────────────────────────────────────

@dataclass
class TradeState:
    """
    Complete record for one paper trade.

    All timestamps are ISO strings (naive UTC) for JSON portability.
    MFE / MAE are maintained in price units and converted to R at close.

    Scaled-exit fields are populated only when exit_mode == "scaled".
    In original mode they remain at their default values and are ignored.
    """
    trade_id:         int
    asset:            str

    # Signal metadata
    signal_ts:        str    # ISO timestamp of 1H signal bar
    direction:        str    # "long" | "short"

    # A+B execution metadata
    a_filter_result:  str
    a_aligned_bars:   int
    b_pullback_found: bool

    # Entry
    entry_ts:         str
    entry_price:      float
    atr_1h:           float    # dollar ATR at signal time
    stop_price:       float
    tp_price:         float    # original single TP (also used as TP3 in scaled)

    # Exit (filled when trade fully closes)
    exit_ts:          Optional[str]   = None
    exit_price:       Optional[float] = None
    exit_reason:      Optional[str]   = None    # "tp" | "stop" | "time"

    # PnL (filled at final close)
    net_pnl_usd:      Optional[float] = None
    r_multiple:       Optional[float] = None
    mfe_r:            Optional[float] = None
    mae_r:            Optional[float] = None

    # Running MFE / MAE trackers (price units, updated each bar)
    _mfe_price: float = 0.0
    _mae_price: float = 0.0

    # MFE / MAE snapshots at fixed post-entry hour checkpoints.  Recorded once
    # each on the first 15m bar where elapsed >= h.  Used by lifetime research
    # (time-to-MFE distributions, early-kill sweeps, adaptive TP tuning).
    # None until the checkpoint is crossed; some remain None if the trade
    # closed earlier.  Extended to h1/h2/h4/h8/h24 on 2026-07-XX for
    # Adaptive Time-Horizon Profit Capture research.
    mfe_r_h1:  Optional[float] = None
    mae_r_h1:  Optional[float] = None
    mfe_r_h2:  Optional[float] = None
    mae_r_h2:  Optional[float] = None
    mfe_r_h4:  Optional[float] = None
    mae_r_h4:  Optional[float] = None
    mfe_r_h6:  Optional[float] = None
    mae_r_h6:  Optional[float] = None
    mfe_r_h8:  Optional[float] = None
    mae_r_h8:  Optional[float] = None
    mfe_r_h12: Optional[float] = None
    mae_r_h12: Optional[float] = None
    mfe_r_h18: Optional[float] = None
    mae_r_h18: Optional[float] = None
    mfe_r_h24: Optional[float] = None
    mae_r_h24: Optional[float] = None

    # Time-to-R-level (minutes from entry to FIRST crossing of each R-level).
    # Only first crossing is recorded — a trade that touched 1R at t=90min,
    # retraced to 0.5R, then broke out to 1.67R has t_to_1R=90 (not the later
    # re-crossing).  This avoids post-hoc bias: the tracker only ever "knows"
    # what the price did up to `now`.
    t_to_0_3R:  Optional[float] = None
    t_to_0_5R:  Optional[float] = None
    t_to_0_65R: Optional[float] = None
    t_to_1R:    Optional[float] = None
    t_to_1_67R: Optional[float] = None

    # MFE peak tracking — when did the running MFE reach its all-time high
    # WITHIN THIS TRADE, and what was the running MAE just before that peak?
    # Also: how many minutes elapsed from the peak to trade close (decay).
    t_to_mfe_peak:        Optional[float] = None   # minutes from entry to peak
    mae_r_before_mfe_peak: Optional[float] = None   # worst MAE_R before the peak
    t_from_peak_to_exit:  Optional[float] = None   # minutes from peak to close

    # Entry-state snapshot — copied from MonitorState at open_trade time.
    # Observational; correlated post-hoc with time-to-MFE and outcomes to
    # inform adaptive TP / setup classification research.
    signal_buy_prob:      Optional[float] = None
    signal_sell_prob:     Optional[float] = None
    entry_atr_pct:        Optional[float] = None
    entry_vol_ratio:      Optional[float] = None
    entry_rsi:            Optional[float] = None
    entry_bb_pos:         Optional[float] = None
    entry_macd_hist:      Optional[float] = None
    entry_sma_50_ratio:   Optional[float] = None
    entry_vol_expansion:  Optional[float] = None
    entry_volume:         Optional[float] = None
    entry_trend:          Optional[str]   = None
    entry_vol_regime:     Optional[str]   = None
    entry_session:        Optional[str]   = None

    # Multi-horizon shadow forecasts at SIGNAL time (observational only).
    signal_buy_prob_h2:   Optional[float] = None
    signal_sell_prob_h2:  Optional[float] = None
    signal_buy_prob_h4:   Optional[float] = None
    signal_sell_prob_h4:  Optional[float] = None
    signal_buy_prob_h8:   Optional[float] = None
    signal_sell_prob_h8:  Optional[float] = None
    signal_buy_prob_h12:  Optional[float] = None
    signal_sell_prob_h12: Optional[float] = None

    # Status
    status: str = "OPEN"    # "OPEN" | "CLOSED"
    notes:  str = ""

    # ── Scaled exit fields ────────────────────────────────────────────────────
    exit_mode: str = "original"   # mirrors config_live.EXIT_MODE at open time

    # Pre-computed TP levels in price units (set at open for scaled mode)
    tp1_price: Optional[float] = None   # entry ± SCALED_TP1_R × sl_dist
    tp2_price: Optional[float] = None   # entry ± SCALED_TP2_R × sl_dist

    # Partial exit flags (prevent re-triggering after restart)
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    be_moved: bool = False   # True after TP1 fires: stop moved to breakeven
    be_hit:   bool = False   # True if BE stop triggered

    # Event timestamps (ISO strings, None until event fires)
    tp1_time: Optional[str] = None
    tp2_time: Optional[str] = None
    tp3_time: Optional[str] = None
    be_time:  Optional[str] = None

    # Running accumulated PnL from partial exits (scaled mode only)
    remaining_fraction:  float = 1.0    # starts 1.0, decreases as partials close
    realized_pnl_usd:    float = 0.0    # sum of all partial exit net PnL
    realized_r_scaled:   float = 0.0    # sum of all partial exit R contributions

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TradeState":
        """
        Deserialise from a JSON dict.
        Unknown keys are ignored (forward compat); missing keys use defaults
        (backward compat when loading old state.json without scaled fields).
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── TradeManager ──────────────────────────────────────────────────────────────

class TradeManager:
    """
    Manages all open paper trades (one per asset).

    Callers pass in a dict of open trades at construction time, then call
    update_bar() for each new 15m bar.  Closed trades are returned by
    update_bar() for CSV logging.
    """

    def __init__(self, open_trades: dict[str, TradeState]) -> None:
        self._trades: dict[str, TradeState] = open_trades

    @property
    def open_trades(self) -> dict[str, TradeState]:
        return self._trades

    # ── Open ──────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        monitor: MonitorState,
        trade_id: int,
        equity: float,
    ) -> TradeState:
        """
        Create and register a new paper trade from a completed MonitorState.

        Entry price is slippage-adjusted (taker fill).
        ATR-based stop/tp are calculated from the signal bar's close.
        In scaled mode, tp1_price and tp2_price are also pre-computed.
        """
        asset     = monitor.asset
        direction = monitor.signal   # +1 or -1
        dir_str   = "long" if direction == 1 else "short"

        slip        = config_live.SLIPPAGE_PCT
        raw_entry   = monitor.entry_price
        entry_price = raw_entry * (1.0 + slip) if direction == 1 else raw_entry * (1.0 - slip)

        atr = monitor.atr_1h

        if direction == 1:
            stop_price = entry_price - atr * config_live.STOP_LOSS_ATR_MULT
            tp_price   = entry_price + atr * config_live.TAKE_PROFIT_ATR_MULT
        else:
            stop_price = entry_price + atr * config_live.STOP_LOSS_ATR_MULT
            tp_price   = entry_price - atr * config_live.TAKE_PROFIT_ATR_MULT

        # Scaled TP levels
        sl_dist = abs(entry_price - stop_price)   # = 1R in price units
        tp1_price = None
        tp2_price = None
        if config_live.EXIT_MODE == "scaled":
            tp1_price = entry_price + direction * config_live.SCALED_TP1_R * sl_dist
            tp2_price = entry_price + direction * config_live.SCALED_TP2_R * sl_dist

        trade = TradeState(
            trade_id         = trade_id,
            asset            = asset,
            signal_ts        = monitor.signal_ts,
            direction        = dir_str,
            a_filter_result  = monitor.a_filter_result,
            a_aligned_bars   = monitor.a_aligned_bars,
            b_pullback_found = monitor.b_pullback_seen,
            entry_ts         = monitor.entry_ts,
            entry_price      = entry_price,
            atr_1h           = atr,
            stop_price       = stop_price,
            tp_price         = tp_price,
            exit_mode        = config_live.EXIT_MODE,
            tp1_price        = tp1_price,
            tp2_price        = tp2_price,
            # Entry-state snapshot copied from MonitorState (populated by
            # poller from signal_engine.score_bar).  Fields default to None
            # so a monitor that predates this schema still opens cleanly.
            signal_buy_prob      = getattr(monitor, "signal_buy_prob",      None),
            signal_sell_prob     = getattr(monitor, "signal_sell_prob",     None),
            entry_atr_pct        = getattr(monitor, "entry_atr_pct",        None),
            entry_vol_ratio      = getattr(monitor, "entry_vol_ratio",      None),
            entry_rsi            = getattr(monitor, "entry_rsi",            None),
            entry_bb_pos         = getattr(monitor, "entry_bb_pos",         None),
            entry_macd_hist      = getattr(monitor, "entry_macd_hist",      None),
            entry_sma_50_ratio   = getattr(monitor, "entry_sma_50_ratio",   None),
            entry_vol_expansion  = getattr(monitor, "entry_vol_expansion",  None),
            entry_volume         = getattr(monitor, "entry_volume",         None),
            entry_trend          = getattr(monitor, "entry_trend",          None),
            entry_vol_regime     = getattr(monitor, "entry_vol_regime",     None),
            entry_session        = getattr(monitor, "entry_session",        None),
            # Shadow multi-horizon forecasts (observational only)
            signal_buy_prob_h2   = getattr(monitor, "signal_buy_prob_h2",   None),
            signal_sell_prob_h2  = getattr(monitor, "signal_sell_prob_h2",  None),
            signal_buy_prob_h4   = getattr(monitor, "signal_buy_prob_h4",   None),
            signal_sell_prob_h4  = getattr(monitor, "signal_sell_prob_h4",  None),
            signal_buy_prob_h8   = getattr(monitor, "signal_buy_prob_h8",   None),
            signal_sell_prob_h8  = getattr(monitor, "signal_sell_prob_h8",  None),
            signal_buy_prob_h12  = getattr(monitor, "signal_buy_prob_h12",  None),
            signal_sell_prob_h12 = getattr(monitor, "signal_sell_prob_h12", None),
        )

        self._trades[asset] = trade

        mode_tag = f"  [scaled TP1={tp1_price:.6f} TP2={tp2_price:.6f}]" \
                   if config_live.EXIT_MODE == "scaled" else ""
        log.info(
            "TRADE OPEN  #%d  %s %s  entry=%.6f  stop=%.6f  tp=%.6f  atr=%.6f%s",
            trade_id, asset, dir_str, entry_price, stop_price, tp_price, atr, mode_tag,
        )
        return trade

    # ── Checkpoint snapshots ──────────────────────────────────────────────────

    def _maybe_snapshot_checkpoint(
        self,
        trade: TradeState,
        bar_ts: pd.Timestamp,
    ) -> None:
        """
        Snapshot MFE / MAE (in R-units) at post-entry hour checkpoints —
        h1 / h2 / h4 / h6 / h8 / h12 / h18 / h24 — each on the FIRST 15m bar
        where the boundary is crossed.

        Must be called AFTER trade._mfe_price / _mae_price have been updated
        for the current bar but BEFORE any exit fires, so the snapshot
        reflects the bar that actually crossed the boundary.
        """
        r_unit = trade.atr_1h * config_live.STOP_LOSS_ATR_MULT
        if r_unit <= 0:
            return

        entry_dt  = datetime.fromisoformat(trade.entry_ts)
        elapsed_h = (bar_ts.to_pydatetime() - entry_dt).total_seconds() / 3600

        mfe_r = trade._mfe_price / r_unit
        mae_r = trade._mae_price / r_unit

        # Same guard on each: first-crossing only, no retroactive rewrites.
        if elapsed_h >= 1  and trade.mfe_r_h1  is None:
            trade.mfe_r_h1,  trade.mae_r_h1  = mfe_r, mae_r
        if elapsed_h >= 2  and trade.mfe_r_h2  is None:
            trade.mfe_r_h2,  trade.mae_r_h2  = mfe_r, mae_r
        if elapsed_h >= 4  and trade.mfe_r_h4  is None:
            trade.mfe_r_h4,  trade.mae_r_h4  = mfe_r, mae_r
        if elapsed_h >= 6  and trade.mfe_r_h6  is None:
            trade.mfe_r_h6,  trade.mae_r_h6  = mfe_r, mae_r
        if elapsed_h >= 8  and trade.mfe_r_h8  is None:
            trade.mfe_r_h8,  trade.mae_r_h8  = mfe_r, mae_r
        if elapsed_h >= 12 and trade.mfe_r_h12 is None:
            trade.mfe_r_h12, trade.mae_r_h12 = mfe_r, mae_r
        if elapsed_h >= 18 and trade.mfe_r_h18 is None:
            trade.mfe_r_h18, trade.mae_r_h18 = mfe_r, mae_r
        if elapsed_h >= 24 and trade.mfe_r_h24 is None:
            trade.mfe_r_h24, trade.mae_r_h24 = mfe_r, mae_r

    def _track_time_to_R_and_peak(
        self,
        trade: TradeState,
        bar_ts: pd.Timestamp,
    ) -> None:
        """
        Track TWO things per bar (called after MFE/MAE update, before exits):

        1. Time-to-R crossings: minutes from entry when the current MFE
           FIRST reaches each of 0.3 / 0.5 / 0.65 / 1.0 / 1.67 R.  Once set,
           the field is never overwritten (matches live-observation semantics).

        2. MFE peak: the running MFE's all-time high FOR THIS TRADE, when it
           was reached, and the running MAE at the moment the peak was set.
           Because MAX-MFE only increases, "MAE before peak" here means:
           what MAE was recorded up to the bar where the peak was refreshed.
        """
        r_unit = trade.atr_1h * config_live.STOP_LOSS_ATR_MULT
        if r_unit <= 0:
            return

        entry_dt   = datetime.fromisoformat(trade.entry_ts)
        elapsed_m  = (bar_ts.to_pydatetime() - entry_dt).total_seconds() / 60.0
        cur_mfe_r  = trade._mfe_price / r_unit
        cur_mae_r  = trade._mae_price / r_unit

        # (1) Time-to-R (first crossing only)
        for level_r, attr in (
            (0.30, "t_to_0_3R"),
            (0.50, "t_to_0_5R"),
            (0.65, "t_to_0_65R"),
            (1.00, "t_to_1R"),
            (1.67, "t_to_1_67R"),
        ):
            if cur_mfe_r >= level_r and getattr(trade, attr) is None:
                setattr(trade, attr, elapsed_m)

        # (2) MFE peak refresh
        prev_peak = trade.__dict__.get("_prev_peak_mfe_r", 0.0)
        if cur_mfe_r > prev_peak:
            trade.__dict__["_prev_peak_mfe_r"] = cur_mfe_r
            trade.t_to_mfe_peak         = elapsed_m
            trade.mae_r_before_mfe_peak = cur_mae_r

    def _finalize_peak_decay(
        self,
        trade: TradeState,
        exit_ts: pd.Timestamp,
    ) -> None:
        """
        Compute time-from-peak-to-exit at trade close.  Called by both the
        original and scaled finalisers so the record is complete regardless
        of exit mode.
        """
        if trade.t_to_mfe_peak is None or trade.entry_ts is None:
            return
        entry_dt = datetime.fromisoformat(trade.entry_ts)
        exit_dt  = exit_ts.to_pydatetime() if hasattr(exit_ts, "to_pydatetime") else exit_ts
        total_minutes = (exit_dt - entry_dt).total_seconds() / 60.0
        trade.t_from_peak_to_exit = max(0.0, total_minutes - trade.t_to_mfe_peak)

    # ── Update ────────────────────────────────────────────────────────────────

    def update_bar(
        self,
        asset: str,
        bar: pd.Series,
        ticker: Optional[dict] = None,
    ) -> Optional[TradeState]:
        """
        Process one 15m bar for the asset's open trade.

        Parameters
        ----------
        asset  : symbol string
        bar    : pd.Series with open/high/low/close; .name = pd.Timestamp
        ticker : live bid/ask dict from LiveFeed.get_ticker_price() — used for
                 time exits in scaled mode.  Falls back to bar close if None.

        Returns
        -------
        TradeState  — trade was closed on this bar (caller writes CSV)
        None        — trade still open
        """
        trade = self._trades.get(asset)
        if trade is None or trade.status == "CLOSED":
            return None

        if trade.exit_mode == "scaled":
            return self._update_bar_scaled(trade, asset, bar, ticker)
        else:
            return self._update_bar_original(trade, asset, bar)

    # ── Original mode ─────────────────────────────────────────────────────────

    def _update_bar_original(
        self,
        trade: TradeState,
        asset: str,
        bar: pd.Series,
    ) -> Optional[TradeState]:
        """Original single-TP logic — completely unchanged."""
        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])
        bar_ts    = bar.name

        direction = 1 if trade.direction == "long" else -1

        # MFE / MAE
        if direction == 1:
            favourable = bar_high  - trade.entry_price
            adverse    = trade.entry_price - bar_low
        else:
            favourable = trade.entry_price - bar_low
            adverse    = bar_high - trade.entry_price

        trade._mfe_price = max(trade._mfe_price, favourable)
        trade._mae_price = max(trade._mae_price, adverse)

        # Hourly checkpoint snapshots + R-level time trackers + MFE peak.
        # All three must run BEFORE any exit branch so the recorded values
        # reflect the bar that actually crossed the threshold.
        self._maybe_snapshot_checkpoint(trade, bar_ts)
        self._track_time_to_R_and_peak(trade, bar_ts)

        slip = config_live.SLIPPAGE_PCT
        exit_price  = None
        exit_reason = None

        if direction == 1:
            if bar_low <= trade.stop_price:
                exit_price  = trade.stop_price * (1.0 - slip)
                exit_reason = "stop"
            elif bar_high >= trade.tp_price:
                exit_price  = trade.tp_price
                exit_reason = "tp"
        else:
            if bar_high >= trade.stop_price:
                exit_price  = trade.stop_price * (1.0 + slip)
                exit_reason = "stop"
            elif bar_low <= trade.tp_price:
                exit_price  = trade.tp_price
                exit_reason = "tp"

        if exit_price is None:
            entry_dt  = datetime.fromisoformat(trade.entry_ts)
            elapsed_h = (bar_ts.to_pydatetime() - entry_dt).total_seconds() / 3600
            if elapsed_h >= config_live.HOLD_HOURS:
                exit_price  = bar_close * (1.0 - slip if direction == 1 else 1.0 + slip)
                exit_reason = "time"

        if exit_price is None:
            return None

        return self._close_trade_original(trade, asset, exit_price, exit_reason, bar_ts)

    def _close_trade_original(
        self,
        trade: TradeState,
        asset: str,
        exit_price: float,
        exit_reason: str,
        exit_ts: pd.Timestamp,
    ) -> TradeState:
        """Finalise original-mode trade record and remove from open_trades."""
        direction = 1 if trade.direction == "long" else -1
        entry     = trade.entry_price

        position_value = config_live.INITIAL_CAPITAL * config_live.POSITION_SIZE_PCT
        shares         = position_value / entry

        gross_pnl  = direction * (exit_price - entry) * shares
        commission = (entry + exit_price) * shares * config_live.COMMISSION_PCT
        net_pnl    = gross_pnl - commission

        r_unit     = trade.atr_1h * config_live.STOP_LOSS_ATR_MULT
        r_multiple = (net_pnl / (r_unit * shares)) if r_unit > 0 and shares > 0 else 0.0
        mfe_r      = trade._mfe_price / r_unit if r_unit > 0 else 0.0
        mae_r      = trade._mae_price / r_unit if r_unit > 0 else 0.0

        trade.exit_ts     = exit_ts.isoformat() if hasattr(exit_ts, "isoformat") else str(exit_ts)
        trade.exit_price  = exit_price
        trade.exit_reason = exit_reason
        trade.net_pnl_usd = net_pnl
        trade.r_multiple  = r_multiple
        trade.mfe_r       = mfe_r
        trade.mae_r       = mae_r
        trade.status      = "CLOSED"
        self._finalize_peak_decay(trade, exit_ts)

        del self._trades[asset]

        log.info(
            "TRADE CLOSE #%d  %s %s  exit=%.6f  reason=%s  PnL=$%.2f  R=%.2f",
            trade.trade_id, asset, trade.direction,
            exit_price, exit_reason, net_pnl, r_multiple,
        )
        return trade

    # ── Scaled mode ───────────────────────────────────────────────────────────

    def _update_bar_scaled(
        self,
        trade: TradeState,
        asset: str,
        bar: pd.Series,
        ticker: Optional[dict],
    ) -> Optional[TradeState]:
        """
        Scaled-exit bar update.

        Exit priority per bar:
          1. BE stop — only if TP1 was set in a PREVIOUS bar (be_moved=True
             AND TP1 was not set this very call)
          2. Next TP (TP1 → TP2 → TP3; multiple can trigger in the same bar)
          3. Time exit for remaining fraction
        """
        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])
        bar_ts    = bar.name
        ts_iso    = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)

        direction = 1 if trade.direction == "long" else -1
        entry     = trade.entry_price
        slip      = config_live.SLIPPAGE_PCT

        # ── MFE / MAE (full-position tracking) ───────────────────────────────
        if direction == 1:
            favourable = bar_high  - entry
            adverse    = entry - bar_low
        else:
            favourable = entry - bar_low
            adverse    = bar_high - entry

        trade._mfe_price = max(trade._mfe_price, favourable)
        trade._mae_price = max(trade._mae_price, adverse)

        # Hourly checkpoints + R-level time trackers + MFE peak (same as
        # original mode — always runs before any exit branch fires).
        self._maybe_snapshot_checkpoint(trade, bar_ts)
        self._track_time_to_R_and_peak(trade, bar_ts)

        # ── Remember which TPs were already hit BEFORE this bar ───────────────
        tp1_was_hit = trade.tp1_hit   # prevents same-bar BE check after TP1 fires
        tp2_was_hit = trade.tp2_hit

        # ── 1. BE stop (only active if TP1 already hit in a prior bar) ────────
        if tp1_was_hit and not trade.tp3_hit and not trade.be_hit:
            be_px = entry   # BE = breakeven = original entry price
            be_triggered = (
                (direction == 1  and bar_low  <= be_px) or
                (direction == -1 and bar_high >= be_px)
            )
            if be_triggered:
                fill = be_px * (1.0 - slip * direction)
                pnl, r = self._calc_partial(trade, fill, trade.remaining_fraction, is_limit=False)
                trade.realized_pnl_usd  += pnl
                trade.realized_r_scaled += r
                trade.be_hit  = True
                trade.be_time = ts_iso
                trade.remaining_fraction = 0.0
                log.info(
                    "BE_EXIT     #%d  %s  fill=%.6f  partial_PnL=$%.2f  "
                    "cumR=%.3f  remaining=0%%",
                    trade.trade_id, asset, fill, pnl, trade.realized_r_scaled,
                )
                return self._finalize_scaled(trade, asset, fill, "stop", bar_ts)

        # ── 2. TP1 ────────────────────────────────────────────────────────────
        if not trade.tp1_hit:
            tp1_triggered = (
                (direction == 1  and bar_high >= trade.tp1_price) or
                (direction == -1 and bar_low  <= trade.tp1_price)
            )
            if tp1_triggered:
                fill = trade.tp1_price   # limit-like fill, no slippage
                pnl, r = self._calc_partial(trade, fill, config_live.SCALED_TP1_FRAC,
                                             is_limit=True)
                trade.realized_pnl_usd  += pnl
                trade.realized_r_scaled += r
                trade.tp1_hit            = True
                trade.be_moved           = True
                trade.tp1_time           = ts_iso
                trade.remaining_fraction = 1.0 - config_live.SCALED_TP1_FRAC
                log.info(
                    "TP1_HIT  BE_MOVED  #%d  %s  fill=%.6f  closed=%.0f%%  "
                    "partial_PnL=$%.2f  cumR=%.3f",
                    trade.trade_id, asset, fill,
                    config_live.SCALED_TP1_FRAC * 100, pnl, trade.realized_r_scaled,
                )

        # ── 3. TP2 ────────────────────────────────────────────────────────────
        if trade.tp1_hit and not trade.tp2_hit:
            tp2_triggered = (
                (direction == 1  and bar_high >= trade.tp2_price) or
                (direction == -1 and bar_low  <= trade.tp2_price)
            )
            if tp2_triggered:
                fill = trade.tp2_price
                pnl, r = self._calc_partial(trade, fill, config_live.SCALED_TP2_FRAC,
                                             is_limit=True)
                trade.realized_pnl_usd  += pnl
                trade.realized_r_scaled += r
                trade.tp2_hit            = True
                trade.tp2_time           = ts_iso
                trade.remaining_fraction = (
                    1.0 - config_live.SCALED_TP1_FRAC - config_live.SCALED_TP2_FRAC
                )
                log.info(
                    "TP2_HIT     #%d  %s  fill=%.6f  closed=%.0f%%  "
                    "partial_PnL=$%.2f  cumR=%.3f",
                    trade.trade_id, asset, fill,
                    config_live.SCALED_TP2_FRAC * 100, pnl, trade.realized_r_scaled,
                )

        # ── 4. TP3 (= original tp_price) ─────────────────────────────────────
        if trade.tp2_hit and not trade.tp3_hit:
            tp3_triggered = (
                (direction == 1  and bar_high >= trade.tp_price) or
                (direction == -1 and bar_low  <= trade.tp_price)
            )
            if tp3_triggered:
                fill = trade.tp_price
                frac = trade.remaining_fraction   # should be SCALED_TP3_FRAC
                pnl, r = self._calc_partial(trade, fill, frac, is_limit=True)
                trade.realized_pnl_usd  += pnl
                trade.realized_r_scaled += r
                trade.tp3_hit            = True
                trade.tp3_time           = ts_iso
                trade.remaining_fraction = 0.0
                log.info(
                    "TP3_HIT     #%d  %s  fill=%.6f  closed=%.0f%%  "
                    "partial_PnL=$%.2f  cumR=%.3f  TRADE COMPLETE",
                    trade.trade_id, asset, fill, frac * 100,
                    pnl, trade.realized_r_scaled,
                )
                return self._finalize_scaled(trade, asset, fill, "tp", bar_ts)

        # ── 5. Time exit — remaining fraction ─────────────────────────────────
        entry_dt  = datetime.fromisoformat(trade.entry_ts)
        elapsed_h = (bar_ts.to_pydatetime() - entry_dt).total_seconds() / 3600
        if elapsed_h >= config_live.HOLD_HOURS:
            if ticker is not None:
                # Use live bid/ask (long exits at bid, short exits at ask)
                raw_fill = ticker["bid"] if direction == 1 else ticker["ask"]
                fill = raw_fill * (1.0 - slip * direction)
                price_source = "live_ticker"
            else:
                fill = bar_close * (1.0 - slip if direction == 1 else 1.0 + slip)
                price_source = "bar_close"

            frac = trade.remaining_fraction
            pnl, r = self._calc_partial(trade, fill, frac, is_limit=False)
            trade.realized_pnl_usd  += pnl
            trade.realized_r_scaled += r
            trade.remaining_fraction = 0.0
            log.info(
                "TIME_EXIT_REMAINING  #%d  %s  fill=%.6f (%s)  "
                "closed=%.0f%%  partial_PnL=$%.2f  cumR=%.3f",
                trade.trade_id, asset, fill, price_source,
                frac * 100, pnl, trade.realized_r_scaled,
            )
            return self._finalize_scaled(trade, asset, fill, "time", bar_ts)

        return None   # trade still open

    # ── Scaled helpers ────────────────────────────────────────────────────────

    def _calc_partial(
        self,
        trade: TradeState,
        fill_price: float,
        fraction: float,
        is_limit: bool,
    ) -> tuple[float, float]:
        """
        Calculate net PnL and R contribution for a partial exit.

        Parameters
        ----------
        fill_price : executed fill price (already slippage-adjusted for taker fills)
        fraction   : fraction of the ORIGINAL full position being closed (0–1)
        is_limit   : True for TP-level exits (no extra slippage); False for
                     market/stop/time exits (slippage already baked into fill_price)

        Returns
        -------
        (net_pnl_usd, r_contribution)
        """
        direction = 1 if trade.direction == "long" else -1
        entry     = trade.entry_price

        pos_value     = config_live.INITIAL_CAPITAL * config_live.POSITION_SIZE_PCT
        total_shares  = pos_value / entry
        partial_shares = total_shares * fraction

        gross = direction * (fill_price - entry) * partial_shares
        comm  = (entry + fill_price) * partial_shares * config_live.COMMISSION_PCT
        pnl   = gross - comm

        sl_dist = abs(entry - trade.stop_price)   # 1R in price units
        r_contrib = (
            direction * (fill_price - entry) / sl_dist * fraction
            if sl_dist > 0 else 0.0
        )
        return pnl, r_contrib

    def _finalize_scaled(
        self,
        trade: TradeState,
        asset: str,
        last_fill: float,
        exit_reason: str,
        exit_ts: pd.Timestamp,
    ) -> TradeState:
        """
        Finalise a scaled-exit trade after the last partial close.

        Sets all exit fields from accumulated partial state and removes the
        trade from open_trades.
        """
        r_unit = abs(trade.entry_price - trade.stop_price)

        trade.exit_ts     = exit_ts.isoformat() if hasattr(exit_ts, "isoformat") else str(exit_ts)
        trade.exit_price  = last_fill
        trade.exit_reason = exit_reason
        trade.net_pnl_usd = trade.realized_pnl_usd
        trade.r_multiple  = trade.realized_r_scaled
        trade.mfe_r       = trade._mfe_price / r_unit if r_unit > 0 else 0.0
        trade.mae_r       = trade._mae_price / r_unit if r_unit > 0 else 0.0
        trade.status      = "CLOSED"
        self._finalize_peak_decay(trade, exit_ts)

        del self._trades[asset]

        log.info(
            "TRADE CLOSE #%d  %s %s  last_fill=%.6f  reason=%s  "
            "netPnL=$%.2f  cumR=%.3f  TP1=%s TP2=%s TP3=%s BE=%s",
            trade.trade_id, asset, trade.direction,
            last_fill, exit_reason,
            trade.net_pnl_usd, trade.r_multiple,
            trade.tp1_hit, trade.tp2_hit, trade.tp3_hit, trade.be_hit,
        )
        return trade

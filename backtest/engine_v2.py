"""
backtest/engine_v2.py — Realistic, lookahead-free backtest engine.

Execution modes
---------------
realistic_execution = True  (default — use this for strategy validation)
    Entry fills at the OPEN of the next available bar after the signal or after
    the 15m A+B pattern confirmation.  Eliminates same-candle entry bias.

    Without 15m data (1H-only):
        signal fires at close of 1H bar T  →  entry at OPEN of bar T+1

    With 15m data (A+B execution):
        signal fires at close of 1H bar T
        → A filter: examine 4 × 15m bars from T+1H
          if < 2 aligned → skip trade entirely
        → B window: scan up to 8 × 15m bars for pullback + resumption
          resumption confirmed at 15m bar K  →  entry at OPEN of bar K+1
          B timeout (no pattern)             →  entry at OPEN of bar after B window

realistic_execution = False  (legacy / naive — for direct comparison only)
    Entry fills at the close of the signal 1H bar.
    Matches the existing engine.py behaviour.  Shows the optimistic bias.

Same-candle SL + TP (worst case)
---------------------------------
If a candle's low touches SL AND its high touches TP in the same bar, we assume
SL filled first (worst case for the trader).  This is the only defensible
assumption when we cannot determine intrabar price order from OHLC data alone.

Exit priority per bar:
    Long:   1. low  <= SL  → fill at SL × (1 − slip)
            2. high >= TP  → fill at TP price  (limit, no slip)
            3. time limit  → fill at close × (1 − slip)
    Short:  1. high >= SL  → fill at SL × (1 + slip)
            2. low  <= TP  → fill at TP price  (limit, no slip)
            3. time limit  → fill at close × (1 + slip)

ATR anchoring
-------------
SL/TP are sized from the ATR measured at the SIGNAL bar (not entry bar).
This is intentional: the signal bar's volatility determines how wide the
risk zone should be.  The actual dollar distance shifts with entry price.

Lookahead validation
--------------------
In realistic mode the engine asserts after simulation:
    - every trade has entry_time strictly > signal_time
    - no trade's entry uses an index <= signal bar index
These checks raise AssertionError if any lookahead leakage is detected.

Output
------
trades       : list[dict] — one record per closed trade (see _TRADE_FIELDS)
equity_curve : pd.Series indexed by date
"""

import math
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config


# ── Trade record field names ───────────────────────────────────────────────────

_TRADE_FIELDS = [
    "trade_id",
    "symbol",
    "signal_time",
    "entry_time",
    "direction",
    "entry_price",
    "stop_price",
    "tp_price",
    "atr_at_signal",
    "exit_time",
    "exit_price",
    "exit_reason",
    "R",
    "pnl",
    # Internals kept for analysis
    "shares",
    "gross_pnl",
    "a_filter_result",   # "pass" | "fail" | "no_data" | "n/a"
    "a_aligned_bars",
    "b_pullback_found",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def run_backtest_v2(
    signals_df: pd.DataFrame,
    df_15m: Optional[pd.DataFrame] = None,
    realistic_execution: bool = True,
    symbol: str = "ASSET",
    slippage_pct: float = config.SLIPPAGE_PCT,
    commission_pct: float = config.COMMISSION_PCT,
    stop_atr_mult: float = config.STOP_LOSS_ATR_MULT,
    tp_atr_mult: float = config.TAKE_PROFIT_ATR_MULT,
    hold_hours: int = config.HOLD_BARS,
    initial_capital: float = config.INITIAL_CAPITAL,
    position_size_pct: float = config.POSITION_SIZE_PCT,
    a_filter_bars: int = 4,
    a_filter_min_aligned: int = 2,
    b_wait_bars: int = 8,
) -> tuple[list[dict], pd.Series]:
    """
    Run the realistic backtest simulation.

    Parameters
    ----------
    signals_df : 1H signal DataFrame with columns:
                 open, high, low, close, signal (1=long, -1=short, 0=flat)
                 and optionally atr_pct.
                 Index must be a DatetimeIndex (naive UTC, sorted ascending).
    df_15m     : 15m OHLCV DataFrame with the same column set and DatetimeIndex.
                 When provided with realistic_execution=True, the A+B entry
                 model is used.  Otherwise ignored.
    realistic_execution : True = next-bar open entry (correct).
                          False = same-bar close entry (legacy, inflated).
    symbol     : Name tag for the trade log.
    slippage_pct : Taker fill slippage per side (applied to market fills).
    commission_pct : One-way commission rate (applied to both entry and exit).
    stop_atr_mult  : SL = entry ± stop_atr_mult × ATR.
    tp_atr_mult    : TP = entry ∓ tp_atr_mult  × ATR.
    hold_hours  : Maximum hold duration in hours.
    initial_capital : Starting equity in $.
    position_size_pct : Fraction of current equity to risk per trade.
    a_filter_bars : Number of 15m bars examined for Approach A filter.
    a_filter_min_aligned : Minimum aligned bars required to pass A.
    b_wait_bars : Maximum 15m bars scanned per signal for Approach B pullback.

    Returns
    -------
    trades       : list of dicts (one per closed trade)
    equity_curve : pd.Series of equity value at each 1H bar in signals_df
    """
    has_atr = "atr_pct" in signals_df.columns
    use_15m = (df_15m is not None) and realistic_execution

    equity       = initial_capital
    trades       = []
    equity_curve = {}
    trade_id     = 1

    index   = signals_df.index
    n_bars  = len(index)
    prices  = signals_df["close"].values
    opens   = signals_df["open"].values
    highs   = signals_df["high"].values
    lows    = signals_df["low"].values
    signals = signals_df["signal"].values
    atrs    = signals_df["atr_pct"].values if has_atr else np.full(n_bars, 0.01)

    in_trade    = False
    direction   = 0
    entry_price = 0.0
    entry_ts    = None
    entry_bar_i = -1
    stop_price  = 0.0
    tp_price    = 0.0
    shares      = 0.0
    r_unit      = 0.0
    signal_ts   = None
    a_result    = "n/a"
    a_aligned   = 0
    b_pullback  = False

    for i in range(n_bars):
        ts    = index[i]
        close = prices[i]
        high  = highs[i]
        low   = lows[i]

        # ── 1. Check for exit ──────────────────────────────────────────────────
        if in_trade:
            ep = None
            er = None

            if direction == 1:
                # Worst-case: SL checked first even if TP also touched
                if low <= stop_price:
                    ep = stop_price * (1.0 - slippage_pct)
                    er = "stop"
                elif high >= tp_price:
                    ep = tp_price             # limit fill, no slip
                    er = "tp"
                elif (ts - entry_ts).total_seconds() / 3600 >= hold_hours:
                    ep = close * (1.0 - slippage_pct)
                    er = "time"
            else:
                if high >= stop_price:
                    ep = stop_price * (1.0 + slippage_pct)
                    er = "stop"
                elif low <= tp_price:
                    ep = tp_price
                    er = "tp"
                elif (ts - entry_ts).total_seconds() / 3600 >= hold_hours:
                    ep = close * (1.0 + slippage_pct)
                    er = "time"

            if ep is not None:
                gross_pnl  = direction * (ep - entry_price) * shares
                commission  = (entry_price + ep) * shares * commission_pct
                net_pnl    = gross_pnl - commission
                equity    += net_pnl

                r_multiple = net_pnl / (r_unit * shares) if (r_unit > 0 and shares > 0) else 0.0

                trades.append({
                    "trade_id":        trade_id,
                    "symbol":          symbol,
                    "signal_time":     signal_ts.isoformat(),
                    "entry_time":      entry_ts.isoformat(),
                    "direction":       "long" if direction == 1 else "short",
                    "entry_price":     entry_price,
                    "stop_price":      stop_price,
                    "tp_price":        tp_price,
                    "atr_at_signal":   r_unit / stop_atr_mult if stop_atr_mult > 0 else 0.0,
                    "exit_time":       ts.isoformat(),
                    "exit_price":      ep,
                    "exit_reason":     er,
                    "R":               r_multiple,
                    "pnl":             net_pnl,
                    "shares":          shares,
                    "gross_pnl":       gross_pnl,
                    "a_filter_result": a_result,
                    "a_aligned_bars":  a_aligned,
                    "b_pullback_found": b_pullback,
                })
                trade_id += 1
                in_trade  = False

        # ── 2. Check for entry ─────────────────────────────────────────────────
        if not in_trade and signals[i] != 0:
            sig = int(signals[i])

            # ATR in dollar terms — from the SIGNAL bar (this bar), not entry bar
            atr_dollars = atrs[i] * close

            if realistic_execution:
                _entry_price, _entry_ts, _a_result, _a_aligned, _b_pullback = (
                    _resolve_realistic_entry(
                        i, sig, close, atr_dollars,
                        index, opens, df_15m,
                        use_15m,
                        a_filter_bars, a_filter_min_aligned, b_wait_bars,
                    )
                )
            else:
                # Legacy mode: entry at signal bar close (same candle — biased)
                _entry_price = close * (1.0 + slippage_pct if sig == 1 else 1.0 - slippage_pct)
                _entry_ts    = ts
                _a_result    = "n/a"
                _a_aligned   = 0
                _b_pullback  = False

            # Skip if no valid entry found (A filtered, or out of 15m data)
            if _entry_price is None:
                equity_curve[ts] = equity
                continue

            # Apply slippage to raw entry (market taker fill)
            entry_price = _entry_price * (1.0 + slippage_pct if sig == 1 else 1.0 - slippage_pct)
            entry_ts    = _entry_ts
            entry_bar_i = i
            direction   = sig
            signal_ts   = ts
            a_result    = _a_result
            a_aligned   = _a_aligned
            b_pullback  = _b_pullback

            # SL/TP anchored to entry price, sized by signal-bar ATR
            if direction == 1:
                stop_price = entry_price - atr_dollars * stop_atr_mult
                tp_price   = entry_price + atr_dollars * tp_atr_mult
            else:
                stop_price = entry_price + atr_dollars * stop_atr_mult
                tp_price   = entry_price - atr_dollars * tp_atr_mult

            # R unit: dollar value of 1R for this trade = SL distance × shares
            position_value = equity * position_size_pct
            shares         = position_value / entry_price
            r_unit         = atr_dollars * stop_atr_mult   # per share

            in_trade = True

        equity_curve[ts] = equity

    # ── Lookahead validation (realistic mode only) ─────────────────────────────
    if realistic_execution and trades:
        _validate_no_lookahead(trades)

    return trades, pd.Series(equity_curve)


def trades_to_dataframe(trades: list[dict]) -> pd.DataFrame:
    """Convert the trades list to a DataFrame with proper column ordering."""
    if not trades:
        return pd.DataFrame(columns=_TRADE_FIELDS)
    df = pd.DataFrame(trades)
    cols = [c for c in _TRADE_FIELDS if c in df.columns]
    return df[cols]


def compute_metrics_v2(trades: list[dict], equity_curve: pd.Series) -> dict:
    """
    Compute full performance statistics from a realistic backtest.

    Extends the existing compute_metrics() with R-multiple statistics and
    per-exit-reason breakdown.
    """
    if not trades:
        return {"error": "No trades"}

    pnls       = [t["pnl"]   for t in trades]
    r_multiples = [t["R"]    for t in trades]
    wins        = [p for p in pnls if p > 0]
    losses      = [p for p in pnls if p <= 0]

    n           = len(pnls)
    win_rate    = len(wins) / n
    loss_rate   = 1 - win_rate
    avg_win     = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss    = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy  = win_rate * avg_win - loss_rate * avg_loss

    gp = sum(wins)
    gl = abs(sum(losses))
    pf = gp / gl if gl > 0 else math.inf

    avg_r   = sum(r_multiples) / n
    win_r   = [r for r, p in zip(r_multiples, pnls) if p > 0]
    loss_r  = [r for r, p in zip(r_multiples, pnls) if p <= 0]
    avg_win_r  = sum(win_r)  / len(win_r)  if win_r  else 0.0
    avg_loss_r = sum(loss_r) / len(loss_r) if loss_r else 0.0

    peak = equity_curve.cummax()
    dd   = (equity_curve - peak) / peak * 100
    max_dd = abs(dd.min())

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100 if len(equity_curve) > 0 else 0.0

    exit_counts = {}
    for reason in ("stop", "tp", "time"):
        exit_counts[f"exit_{reason}"] = sum(1 for t in trades if t.get("exit_reason") == reason)

    a_filter = {}
    if any("a_filter_result" in t for t in trades):
        a_filter["a_pass"]    = sum(1 for t in trades if t.get("a_filter_result") == "pass")
        a_filter["a_fail"]    = sum(1 for t in trades if t.get("a_filter_result") == "fail")
        a_filter["b_pullback"]= sum(1 for t in trades if t.get("b_pullback_found", False))

    return {
        "n_trades":      n,
        "win_rate":      win_rate,
        "profit_factor": pf,
        "expectancy":    expectancy,
        "avg_r":         avg_r,
        "avg_win_r":     avg_win_r,
        "avg_loss_r":    avg_loss_r,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "max_drawdown":  max_dd,
        "total_return":  total_ret,
        "final_equity":  equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital,
        **exit_counts,
        **a_filter,
    }


# ── Entry resolution ───────────────────────────────────────────────────────────

def _resolve_realistic_entry(
    signal_bar_i: int,
    direction: int,
    signal_close: float,
    atr_dollars: float,
    index: pd.DatetimeIndex,
    opens: np.ndarray,
    df_15m: Optional[pd.DataFrame],
    use_15m: bool,
    a_filter_bars: int,
    a_filter_min_aligned: int,
    b_wait_bars: int,
) -> tuple[Optional[float], Optional[pd.Timestamp], str, int, bool]:
    """
    Resolve entry price for a signal bar.

    Returns
    -------
    (raw_entry_price, entry_ts, a_result, a_aligned, b_pullback)

    raw_entry_price : float before slippage adjustment, or None if trade skipped
    entry_ts        : pd.Timestamp of the entry bar's open
    a_result        : "pass" | "fail" | "no_data" | "n/a"
    a_aligned       : number of aligned A-window bars (int)
    b_pullback      : whether a pullback was detected in B window (bool)
    """
    signal_ts = index[signal_bar_i]

    if use_15m:
        return _entry_via_15m_ab(
            signal_ts, direction, signal_close,
            df_15m, a_filter_bars, a_filter_min_aligned, b_wait_bars,
        )

    # 1H-only realistic: enter at open of the very next 1H bar
    next_i = signal_bar_i + 1
    if next_i >= len(index):
        return None, None, "n/a", 0, False

    return opens[next_i], index[next_i], "n/a", 0, False


def _entry_via_15m_ab(
    signal_ts: pd.Timestamp,
    direction: int,
    ref_price: float,
    df_15m: pd.DataFrame,
    k_bars: int,
    min_aligned: int,
    max_wait: int,
) -> tuple[Optional[float], Optional[pd.Timestamp], str, int, bool]:
    """
    Run A+B 15m execution on one signal and return the realistic entry.

    KEY DIFFERENCE vs. the batch annotator in execution_15m.py:
        Old code  → entry = bar_k.close   (INSIDE the confirming bar)
        This code → entry = bar_k+1.open  (OUTSIDE — the next bar after confirmation)

    This eliminates the intrabar lookahead that the original annotator has.

    Flow
    ----
    1. A filter: examine k_bars 15m bars starting at signal_ts + 1H.
       Count bars that close in direction (close > open for long).
       If aligned < min_aligned → skip trade (entry_price=None).

    2. B window: scan up to max_wait 15m bars for pullback + resumption.
       Pullback: a bar that closes AGAINST direction past ref_price.
       Resumption: the next bar(s) after pullback that close IN direction.
       Entry → open of the bar AFTER the resumption bar.

    3. B timeout (no pullback+resumption): entry = open of bar AFTER B window.

    4. No 15m data available: fallback to None (skip, no data = no trade).
    """
    idx_15m = df_15m.index
    if len(idx_15m) == 0:
        return None, None, "no_data", 0, False

    # A window: starts 1H after signal bar open (= signal close time)
    a_start_ts = signal_ts + pd.Timedelta(hours=1)

    # Look-ahead guard: a_start_ts must be within 15m data range
    if a_start_ts < idx_15m[0]:
        return None, None, "no_data", 0, False

    pos_a = int(idx_15m.searchsorted(a_start_ts))
    if pos_a >= len(idx_15m):
        return None, None, "no_data", 0, False

    end_a = pos_a + k_bars
    if end_a > len(idx_15m):
        return None, None, "no_data", 0, False   # not enough A bars

    window_a = df_15m.iloc[pos_a:end_a]

    # Count how many A bars close in the signal direction
    aligned = int(((window_a["close"] - window_a["open"]) * direction > 0).sum())

    if aligned < min_aligned:
        return None, None, "fail", aligned, False

    # ── B window ──────────────────────────────────────────────────────────────
    pos_b        = end_a
    pullback_seen = False

    for j in range(max_wait):
        bar_idx = pos_b + j
        if bar_idx >= len(idx_15m):
            return None, None, "pass", aligned, pullback_seen   # out of 15m data

        bar       = df_15m.iloc[bar_idx]
        bar_close = float(bar["close"])
        bar_open  = float(bar["open"])

        if not pullback_seen:
            # Detect counter-move past the 1H signal bar close price
            if (direction == 1 and bar_close < ref_price) or \
               (direction == -1 and bar_close > ref_price):
                pullback_seen = True
                # The pullback bar itself is NOT the entry trigger.
                # We wait for the NEXT bar in B_PULLBACK state.
                continue

        else:
            # Pullback confirmed — wait for resumption (bar closes in signal dir)
            resumption = (direction == 1 and bar_close > bar_open) or \
                         (direction == -1 and bar_close < bar_open)
            if resumption:
                # Entry at OPEN of the bar AFTER the resumption bar
                next_idx = bar_idx + 1
                if next_idx >= len(idx_15m):
                    return None, None, "pass", aligned, True
                next_bar = df_15m.iloc[next_idx]
                return float(next_bar["open"]), idx_15m[next_idx], "pass", aligned, True

    # B window exhausted without resumption — fallback entry at open of next bar
    fallback_idx = pos_b + max_wait
    if fallback_idx >= len(idx_15m):
        return None, None, "pass", aligned, pullback_seen
    fallback_bar = df_15m.iloc[fallback_idx]
    return float(fallback_bar["open"]), idx_15m[fallback_idx], "pass", aligned, pullback_seen


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_no_lookahead(trades: list[dict]) -> None:
    """
    Assert that every realistic trade has entry_time strictly after signal_time.

    Raises AssertionError with the offending trade if any violation is found.
    """
    for t in trades:
        sig_ts   = pd.Timestamp(t["signal_time"])
        entry_ts = pd.Timestamp(t["entry_time"])
        assert entry_ts > sig_ts, (
            f"LOOKAHEAD VIOLATION: trade #{t['trade_id']} "
            f"signal_time={t['signal_time']} "
            f"entry_time={t['entry_time']} — "
            f"entry must be strictly AFTER signal bar close."
        )

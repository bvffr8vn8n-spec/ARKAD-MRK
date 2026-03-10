"""
backtest/engine.py — Simulates trades from a signal DataFrame.

Rules:
  - Long and short, one position at a time.
  - signal = 1  → long entry
  - signal = -1 → short entry
  - signal = 0  → no trade
  - Exit rules (checked in order each bar):
      Long:
        1. Stop loss   : bar low  <= stop_price  → fill at stop_price × (1 − slip)
        2. Take profit : bar high >= tp_price    → fill at tp_price   (limit, no slip)
        3. Time exit   : HOLD_BARS elapsed       → fill at close      × (1 − slip)
      Short:
        1. Stop loss   : bar high >= stop_price  → fill at stop_price × (1 + slip)
        2. Take profit : bar low  <= tp_price    → fill at tp_price   (limit, no slip)
        3. Time exit   : HOLD_BARS elapsed       → fill at close      × (1 + slip)
  - Stop and TP are set at entry using ATR multiples scaled to entry volatility.
  - Position size = POSITION_SIZE_PCT of current equity.
  - Round-trip commission + slippage applied to each trade.
  - No pyramiding (a new signal while in a trade is ignored).

Slippage model
--------------
  SLIPPAGE_PCT (config) represents taker slippage per side.
  Market orders always fill at the unfavourable side of the spread.
  Limit orders (TP) fill at exact price.

Returns
-------
trades       : list[dict]  — one record per closed trade
equity_curve : pd.Series   — equity value at each bar in the test window
"""

import pandas as pd

import config


def run_backtest(signals_df: pd.DataFrame):
    """
    Parameters
    ----------
    signals_df : DataFrame indexed by date, must have columns:
                 open, high, low, close, signal (1=long, -1=short, 0=flat)
                 and optionally atr_pct (used to set stop/TP levels).

    Returns
    -------
    trades       : list of trade dictionaries
    equity_curve : pd.Series of equity over time
    """
    has_atr = "atr_pct" in signals_df.columns
    slip    = config.SLIPPAGE_PCT

    equity       = config.INITIAL_CAPITAL
    trades       = []
    equity_curve = {}

    in_trade    = False
    direction   = 0        # +1 = long, -1 = short
    entry_price = 0.0
    entry_date  = None
    stop_price  = 0.0
    tp_price    = 0.0
    max_bar_idx = 0
    shares      = 0.0

    prices  = signals_df["close"]
    highs   = signals_df["high"]
    lows    = signals_df["low"]
    signals = signals_df["signal"]
    index   = signals_df.index

    for i, date in enumerate(index):
        close = prices.iloc[i]
        high  = highs.iloc[i]
        low   = lows.iloc[i]

        # ── Check for exit ────────────────────────────────────────────────────
        if in_trade:
            exit_price  = None
            exit_reason = None

            if direction == 1:
                # Long exits
                if low <= stop_price:                       # stop hit
                    exit_price  = stop_price * (1.0 - slip)
                    exit_reason = "stop"
                elif high >= tp_price:                      # TP hit (limit)
                    exit_price  = tp_price
                    exit_reason = "tp"
                elif i >= max_bar_idx:                      # time exit
                    exit_price  = close * (1.0 - slip)
                    exit_reason = "time"

            else:
                # Short exits
                if high >= stop_price:                      # stop hit (bought back high)
                    exit_price  = stop_price * (1.0 + slip)
                    exit_reason = "stop"
                elif low <= tp_price:                       # TP hit (limit)
                    exit_price  = tp_price
                    exit_reason = "tp"
                elif i >= max_bar_idx:                      # time exit (bought back high)
                    exit_price  = close * (1.0 + slip)
                    exit_reason = "time"

            if exit_price is not None:
                # PnL: positive when exit is favourable for the direction taken
                gross_pnl  = direction * (exit_price - entry_price) * shares
                commission = (entry_price + exit_price) * shares * config.COMMISSION_PCT
                net_pnl    = gross_pnl - commission
                equity    += net_pnl

                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "direction":   "long" if direction == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "stop_price":  stop_price,
                    "tp_price":    tp_price,
                    "exit_reason": exit_reason,
                    "shares":      shares,
                    "gross_pnl":   gross_pnl,
                    "net_pnl":     net_pnl,
                    # return_pct is always positive for a winning trade
                    "return_pct":  direction * (exit_price / entry_price - 1) * 100,
                })
                in_trade = False

        # ── Check for entry ───────────────────────────────────────────────────
        if not in_trade:
            sig = signals.iloc[i]

            if sig == 1:
                # Long entry: market buy fills above close
                entry_price = close * (1.0 + slip)
                direction   = 1
            elif sig == -1:
                # Short entry: market sell fills below close
                entry_price = close * (1.0 - slip)
                direction   = -1
            else:
                equity_curve[date] = equity
                continue

            position_value = equity * config.POSITION_SIZE_PCT
            shares         = position_value / entry_price
            entry_date     = date
            max_bar_idx    = i + config.HOLD_BARS

            if has_atr:
                atr_dollars = signals_df["atr_pct"].iloc[i] * close
            else:
                atr_dollars = close * 0.01

            if direction == 1:
                stop_price = entry_price - atr_dollars * config.STOP_LOSS_ATR_MULT
                tp_price   = entry_price + atr_dollars * config.TAKE_PROFIT_ATR_MULT
            else:
                stop_price = entry_price + atr_dollars * config.STOP_LOSS_ATR_MULT
                tp_price   = entry_price - atr_dollars * config.TAKE_PROFIT_ATR_MULT

            in_trade = True

        equity_curve[date] = equity

    return trades, pd.Series(equity_curve)

"""
backtest/engine.py — Simulates trades from a signal DataFrame.

Two exit modes (chosen via `exit_mode` parameter):

  "single" (default) — Single TP/SL, matches legacy behaviour.
      Long:
        1. Stop loss   : bar low  <= stop_price  → fill at stop_price × (1 − slip)
        2. Take profit : bar high >= tp_price    → fill at tp_price   (limit, no slip)
        3. Time exit   : HOLD_BARS elapsed       → fill at close      × (1 − slip)
      Short: mirrored.

  "scaled" — Three partial exits with BE stop after TP1 (mirrors paper_trading).
      Pre-computed at entry (in price units):
        tp1_price = entry ± SCALED_TP1_R × sl_dist
        tp2_price = entry ± SCALED_TP2_R × sl_dist
        tp3_price = tp_price (the original single TP, = SCALED_TP3_R × sl_dist)

      Per-bar priority:
        1. BE stop (only when TP1 was hit in a PRIOR bar)
             → close remaining_fraction at entry price (BE = 0R), reason="stop"
        2. TP1 (if not yet hit)
             → close SCALED_TP1_FRAC at tp1_price, move stop to BE
             → SAME bar: also check TP2 / TP3 (cascading partials)
        3. TP2 (only after TP1) → close SCALED_TP2_FRAC at tp2_price
        4. TP3 (only after TP2) → close remaining at tp_price, exit reason="tp"
        5. Time exit on the remaining fraction → reason="time"

      Final trade record's `net_pnl` and `R` reflect the SUM of all partials.

General rules:
  - Long and short, one position at a time (no pyramiding).
  - Stop and TP sized from ATR at the signal/entry bar.
  - Position size = POSITION_SIZE_PCT of current equity.
  - Round-trip commission + slippage applied to each partial.
  - Limit fills (TP1/TP2/TP3) have no slippage; market fills (SL/BE/time) do.

Returns
-------
trades       : list[dict]  — one record per closed trade
equity_curve : pd.Series   — equity value at each bar in the test window
"""

import math

import pandas as pd

import config


def run_backtest(
    signals_df: pd.DataFrame,
    exit_mode: str | None = None,
    early_kill: dict | None = None,
):
    """
    Parameters
    ----------
    signals_df : DataFrame indexed by date, must have columns:
                 open, high, low, close, signal (1=long, -1=short, 0=flat)
                 and optionally atr_pct (used to set stop/TP levels).
    exit_mode  : "single" | "scaled" | None
                 None → use config.BACKTEST_EXIT_MODE (defaults to "single").
    early_kill : optional dict for experimental early-kill rule (scaled mode
                 only).  Format: {"at_h": int, "mfe_thr": float}.  When
                 elapsed bars since entry == at_h AND running MFE in R-units
                 < mfe_thr AND TP1 has NOT yet fired, the trade is force-
                 closed at bar close with exit_reason = "early_kill".
                 None (default) preserves legacy behaviour.

                 Used by experiments/early_kill_sweep.py to bracket the
                 P&L impact of an MFE-based time cut.

    Returns
    -------
    trades       : list of trade dictionaries
    equity_curve : pd.Series of equity over time
    """
    if exit_mode is None:
        exit_mode = getattr(config, "BACKTEST_EXIT_MODE", "single")
    if exit_mode not in ("single", "scaled"):
        raise ValueError(f"exit_mode must be 'single' or 'scaled', got {exit_mode!r}")

    ek_at_h    = None
    ek_mfe_thr = None
    if early_kill is not None:
        if exit_mode != "scaled":
            raise ValueError("early_kill is only supported for exit_mode='scaled'")
        ek_at_h    = int(early_kill["at_h"])
        ek_mfe_thr = float(early_kill["mfe_thr"])

    has_atr            = "atr_pct"          in signals_df.columns
    has_entry_override = "entry_price_15m" in signals_df.columns
    slip               = config.SLIPPAGE_PCT
    comm               = config.COMMISSION_PCT

    # Scaled-mode parameters (mirror paper_trading/config_live.py)
    SC_TP1_R   = getattr(config, "SCALED_TP1_R",   0.65)
    SC_TP2_R   = getattr(config, "SCALED_TP2_R",   1.00)
    SC_TP1_FR  = getattr(config, "SCALED_TP1_FRAC", 0.50)
    SC_TP2_FR  = getattr(config, "SCALED_TP2_FRAC", 0.25)
    SC_TP3_FR  = getattr(config, "SCALED_TP3_FRAC", 0.25)

    equity       = config.INITIAL_CAPITAL
    trades       = []
    equity_curve = {}

    # Position state
    in_trade    = False
    direction   = 0        # +1 = long, -1 = short
    entry_price = 0.0
    entry_date  = None
    stop_price  = 0.0
    tp_price    = 0.0     # = TP3 in scaled mode
    max_bar_idx = 0
    shares      = 0.0     # total shares opened at entry

    # Scaled-only state
    tp1_price = 0.0
    tp2_price = 0.0
    tp1_hit   = False

    # Early-kill tracker — only used when `early_kill` is set.  `max_fav_price`
    # is running max of favourable excursion (in price units) since entry;
    # `entry_bar_idx` is the bar index where the trade opened so we can detect
    # the `at_h`-th post-entry bar.  Reset at every entry.
    max_fav_price = 0.0
    entry_bar_idx = 0
    tp2_hit   = False
    tp3_hit   = False
    be_moved  = False
    realized_pnl = 0.0
    realized_r   = 0.0
    remaining_frac = 1.0
    tp1_date = None
    tp2_date = None
    tp3_date = None
    be_date  = None

    prices  = signals_df["close"]
    highs   = signals_df["high"]
    lows    = signals_df["low"]
    signals = signals_df["signal"]
    index   = signals_df.index

    def _close_partial(fill_price, fraction, is_limit):
        """Compute (net_pnl, r_contrib) for closing `fraction` of position."""
        partial_shares = shares * fraction
        gross = direction * (fill_price - entry_price) * partial_shares
        commission = (entry_price + fill_price) * partial_shares * comm
        pnl = gross - commission
        sl_dist = abs(entry_price - stop_price)
        r_contrib = (
            direction * (fill_price - entry_price) / sl_dist * fraction
            if sl_dist > 0 else 0.0
        )
        return pnl, r_contrib

    def _finalize_single(exit_price, exit_reason, exit_date):
        """Build trade record for single-mode exit."""
        gross = direction * (exit_price - entry_price) * shares
        commission = (entry_price + exit_price) * shares * comm
        net = gross - commission
        sl_dist = abs(entry_price - stop_price)
        r_mult = direction * (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
        return {
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "direction":   "long" if direction == 1 else "short",
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "stop_price":  stop_price,
            "tp_price":    tp_price,
            "exit_reason": exit_reason,
            "shares":      shares,
            "gross_pnl":   gross,
            "net_pnl":     net,
            "R":           r_mult,
            "return_pct":  direction * (exit_price / entry_price - 1) * 100,
            "exit_mode":   "single",
        }

    def _finalize_scaled(last_fill, exit_reason, exit_date):
        """Build trade record for scaled-mode exit (after all partials)."""
        sl_dist = abs(entry_price - stop_price)
        return {
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "direction":   "long" if direction == 1 else "short",
            "entry_price": entry_price,
            "exit_price":  last_fill,
            "stop_price":  stop_price,
            "tp_price":    tp_price,
            "tp1_price":   tp1_price,
            "tp2_price":   tp2_price,
            "exit_reason": exit_reason,
            "shares":      shares,
            "net_pnl":     realized_pnl,
            "R":           realized_r,
            "return_pct":  realized_pnl / (shares * entry_price) * 100 if shares > 0 else 0.0,
            "tp1_hit":     tp1_hit,
            "tp2_hit":     tp2_hit,
            "tp3_hit":     tp3_hit,
            "be_hit":      (exit_reason == "stop" and tp1_hit),
            "tp1_date":    tp1_date,
            "tp2_date":    tp2_date,
            "tp3_date":    tp3_date,
            "be_date":     be_date,
            "exit_mode":   "scaled",
        }

    for i, date in enumerate(index):
        close = prices.iloc[i]
        high  = highs.iloc[i]
        low   = lows.iloc[i]

        # ── Check for exit ────────────────────────────────────────────────────
        if in_trade:
            if exit_mode == "single":
                exit_price  = None
                exit_reason = None
                if direction == 1:
                    if low <= stop_price:
                        exit_price  = stop_price * (1.0 - slip)
                        exit_reason = "stop"
                    elif high >= tp_price:
                        exit_price  = tp_price
                        exit_reason = "tp"
                    elif i >= max_bar_idx:
                        exit_price  = close * (1.0 - slip)
                        exit_reason = "time"
                else:
                    if high >= stop_price:
                        exit_price  = stop_price * (1.0 + slip)
                        exit_reason = "stop"
                    elif low <= tp_price:
                        exit_price  = tp_price
                        exit_reason = "tp"
                    elif i >= max_bar_idx:
                        exit_price  = close * (1.0 + slip)
                        exit_reason = "time"

                if exit_price is not None:
                    rec = _finalize_single(exit_price, exit_reason, date)
                    equity += rec["net_pnl"]
                    trades.append(rec)
                    in_trade = False

            else:  # scaled
                # Track running max-favourable-excursion in price units
                # (used by the optional early-kill rule below)
                if ek_at_h is not None:
                    if direction == 1:
                        fav = high - entry_price
                    else:
                        fav = entry_price - low
                    if fav > max_fav_price:
                        max_fav_price = fav

                # Snapshot whether TP1 was hit BEFORE this bar
                tp1_was_hit = tp1_hit

                # 1. BE stop (only if TP1 was hit in a PRIOR bar)
                if tp1_was_hit and not tp3_hit:
                    be_px = entry_price
                    be_triggered = (
                        (direction == 1  and low  <= be_px) or
                        (direction == -1 and high >= be_px)
                    )
                    if be_triggered:
                        # Market-fill on the remaining fraction (slippage adverse)
                        fill = be_px * (1.0 - slip if direction == 1 else 1.0 + slip)
                        pnl, r = _close_partial(fill, remaining_frac, is_limit=False)
                        realized_pnl += pnl
                        realized_r   += r
                        be_date = date
                        rec = _finalize_scaled(fill, "stop", date)
                        equity += realized_pnl
                        trades.append(rec)
                        in_trade = False
                        equity_curve[date] = equity
                        continue

                # 2. SL (only if TP1 not yet hit — pre-TP1 phase)
                if not tp1_was_hit:
                    sl_triggered = (
                        (direction == 1  and low  <= stop_price) or
                        (direction == -1 and high >= stop_price)
                    )
                    if sl_triggered:
                        fill = stop_price * (1.0 - slip if direction == 1 else 1.0 + slip)
                        pnl, r = _close_partial(fill, remaining_frac, is_limit=False)
                        realized_pnl += pnl
                        realized_r   += r
                        rec = _finalize_scaled(fill, "stop", date)
                        equity += realized_pnl
                        trades.append(rec)
                        in_trade = False
                        equity_curve[date] = equity
                        continue

                # 3. TP1 (cascade-style: same bar can also hit TP2/TP3)
                if not tp1_hit:
                    tp1_triggered = (
                        (direction == 1  and high >= tp1_price) or
                        (direction == -1 and low  <= tp1_price)
                    )
                    if tp1_triggered:
                        fill = tp1_price
                        pnl, r = _close_partial(fill, SC_TP1_FR, is_limit=True)
                        realized_pnl += pnl
                        realized_r   += r
                        tp1_hit = True
                        be_moved = True
                        tp1_date = date
                        remaining_frac = 1.0 - SC_TP1_FR

                # 4. TP2 (only after TP1)
                if tp1_hit and not tp2_hit:
                    tp2_triggered = (
                        (direction == 1  and high >= tp2_price) or
                        (direction == -1 and low  <= tp2_price)
                    )
                    if tp2_triggered:
                        fill = tp2_price
                        pnl, r = _close_partial(fill, SC_TP2_FR, is_limit=True)
                        realized_pnl += pnl
                        realized_r   += r
                        tp2_hit = True
                        tp2_date = date
                        remaining_frac = 1.0 - SC_TP1_FR - SC_TP2_FR

                # 5. TP3 (only after TP2) — final exit
                if tp2_hit and not tp3_hit:
                    tp3_triggered = (
                        (direction == 1  and high >= tp_price) or
                        (direction == -1 and low  <= tp_price)
                    )
                    if tp3_triggered:
                        fill = tp_price
                        pnl, r = _close_partial(fill, remaining_frac, is_limit=True)
                        realized_pnl += pnl
                        realized_r   += r
                        tp3_hit = True
                        tp3_date = date
                        remaining_frac = 0.0
                        rec = _finalize_scaled(fill, "tp", date)
                        equity += realized_pnl
                        trades.append(rec)
                        in_trade = False
                        equity_curve[date] = equity
                        continue

                # 6a. Early-kill (experimental) — fires once on the at_h-th
                #     post-entry bar IF TP1 has not yet hit AND running MFE in
                #     R-units is below the threshold.  Force-closes the
                #     remaining fraction at bar close (taker fill, like time).
                if ek_at_h is not None and not tp1_hit:
                    elapsed_bars = i - entry_bar_idx
                    if elapsed_bars == ek_at_h:
                        sl_dist = abs(entry_price - stop_price)
                        mfe_r = max_fav_price / sl_dist if sl_dist > 0 else 0.0
                        if mfe_r < ek_mfe_thr:
                            fill = close * (1.0 - slip if direction == 1 else 1.0 + slip)
                            pnl, r = _close_partial(fill, remaining_frac, is_limit=False)
                            realized_pnl += pnl
                            realized_r   += r
                            rec = _finalize_scaled(fill, "early_kill", date)
                            equity += realized_pnl
                            trades.append(rec)
                            in_trade = False
                            equity_curve[date] = equity
                            continue

                # 6b. Time exit on remaining fraction
                if i >= max_bar_idx and remaining_frac > 0:
                    fill = close * (1.0 - slip if direction == 1 else 1.0 + slip)
                    pnl, r = _close_partial(fill, remaining_frac, is_limit=False)
                    realized_pnl += pnl
                    realized_r   += r
                    rec = _finalize_scaled(fill, "time", date)
                    equity += realized_pnl
                    trades.append(rec)
                    in_trade = False
                    equity_curve[date] = equity
                    continue

        # ── Check for entry ───────────────────────────────────────────────────
        if not in_trade:
            sig = signals.iloc[i]

            if sig == 1:
                raw = signals_df["entry_price_15m"].iloc[i] if has_entry_override else float("nan")
                base = raw if (has_entry_override and not math.isnan(float(raw))) else close
                entry_price = base * (1.0 + slip)
                direction   = 1
            elif sig == -1:
                raw = signals_df["entry_price_15m"].iloc[i] if has_entry_override else float("nan")
                base = raw if (has_entry_override and not math.isnan(float(raw))) else close
                entry_price = base * (1.0 - slip)
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

            # Pre-compute scaled TP levels (used only in scaled mode)
            sl_dist = abs(entry_price - stop_price)
            if direction == 1:
                tp1_price = entry_price + SC_TP1_R * sl_dist
                tp2_price = entry_price + SC_TP2_R * sl_dist
            else:
                tp1_price = entry_price - SC_TP1_R * sl_dist
                tp2_price = entry_price - SC_TP2_R * sl_dist

            # Reset scaled-only state
            tp1_hit = tp2_hit = tp3_hit = be_moved = False
            realized_pnl = 0.0
            realized_r   = 0.0
            remaining_frac = 1.0
            tp1_date = tp2_date = tp3_date = be_date = None

            # Reset early-kill tracker
            entry_bar_idx = i
            max_fav_price = 0.0

            in_trade = True

        equity_curve[date] = equity

    return trades, pd.Series(equity_curve)

"""
experiments/label_atr_sweep.py — Sweep LABEL_ATR_MULT to find the PF>1 / frequency crossover.

Loads data and features once, then for each LABEL_ATR_MULT:
  - Relabels with the new multiplier
  - Trains a fresh RF 2-class model
  - Generates signals and applies trend filter
  - Runs a threshold sweep (0.50 – 0.60)
  - Runs walk-forward (3 expanding windows)
  - Runs final backtest at threshold = 0.55
  - Collects all metrics for a side-by-side comparison table

Benchmarks include the two already-known points:
  0.50 — previous baseline (too noisy, PF~0.66)
  1.00 — strict labels (clean edge, too few trades)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import add_regime_columns, add_session_column, apply_trend_filter
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


DATA_PATH   = "data/BTCUSDT_1h_4y.csv"
SWEEP_MULTS = [0.65, 0.75, 0.85]
THRESHOLDS  = [0.50, 0.52, 0.55, 0.58, 0.60]

# Benchmark values from prior runs (filled in manually — no re-run needed)
BENCHMARKS = {
    0.50: {
        "n_buy": 14895, "n_sell": 14168, "n_neutral": 5778,
        "buy_above": {0.50: 3775, 0.52: 2886, 0.55: 1088, 0.58: 68, 0.60: 28},
        "sell_above": {0.50: 3194, 0.52: 2201, 0.55: 721, 0.58: 114, 0.60: 81},
        "sweep": {
            0.50: (557, 0.68, -2.63),
            0.52: (462, 0.68, -2.72),
            0.55: (211, 0.76, -2.18),
            0.58: (23,  1.23, 1.61),
            0.60: (10,  3.21, 12.28),
        },
        "wf_pf": [0.57, 0.58, 0.70], "wf_trades": [210, 175, 191],
        "bt55_trades": 340, "bt55_pf": 0.72, "bt55_exp": -2.34,
    },
    1.00: {
        "n_buy": 12347, "n_sell": 11724, "n_neutral": 10770,
        "buy_above": {0.50: 4304, 0.52: 2399, 0.55: 60, 0.58: 17, 0.60: 15},
        "sell_above": {0.50: 2665, 0.52: 1240, 0.55: 57, 0.58: 5, 0.60: 1},
        "sweep": {
            0.50: (594, 0.66, -2.73),
            0.52: (387, 0.69, -2.63),
            0.55: (20,  2.53, 5.87),
            0.58: (9,   2.04, 7.23),
            0.60: (8,   2.22, 8.77),
        },
        "wf_pf": [0.57, 0.65, 1.24], "wf_trades": [167, 110, 23],
        "bt55_trades": 46, "bt55_pf": 1.02, "bt55_exp": 0.14,
    },
}


# -- Helpers -------------------------------------------------------------------

def _wf_windows(df):
    """Build (train_start, train_end, test_end) index positions."""
    n = len(df)
    windows = []
    for tf in config.WF_TRAIN_FRACTIONS:
        te = tf + config.WF_TEST_FRACTION
        windows.append((0, int(n * tf), min(int(n * te), n), tf, te))
    return windows


def _run_sweep(signals_df, feature_cols, thresholds):
    rows = {}
    for t in thresholds:
        sweep_df = signals_df.copy()
        sweep_df["signal"] = (sweep_df["buy_prob"] >= t).astype(int) - \
                             (sweep_df["sell_prob"] >= t).astype(int)
        # 2-class: conflict → flat
        conflict = (sweep_df["buy_prob"] >= t) & (sweep_df["sell_prob"] >= t)
        sweep_df.loc[conflict, "signal"] = 0
        trades, eq = run_backtest(sweep_df)
        m = compute_metrics(trades, eq)
        if "error" in m or m.get("n_trades", 0) == 0:
            rows[t] = (0, float("nan"), float("nan"))
        else:
            rows[t] = (int(m["n_trades"]), float(m["profit_factor"]),
                       float(m["expectancy"]))
    return rows


def _run_wf(df, feature_cols):
    windows = _wf_windows(df)
    pf_list, trade_list = [], []
    for (ts, te, tend, tf, tef) in windows:
        purge = te - config.FORWARD_RETURN_WINDOW
        train = df.iloc[ts:purge]
        test  = df.iloc[te:tend]
        if len(train) < 50 or len(test) < 5:
            continue
        model   = fit_model(train, feature_cols)
        sig_df  = apply_signals(model, feature_cols, test)
        trades, eq = run_backtest(sig_df)
        m = compute_metrics(trades, eq)
        n  = int(m.get("n_trades", 0) or 0)
        pf = float(m.get("profit_factor", float("nan")) or float("nan"))
        pf_list.append(pf)
        trade_list.append(n)
        print(f"    WF [{tf*100:.0f}–{tef*100:.0f}%]: {n} trades  PF={pf:.2f}")
    return pf_list, trade_list


def _run_backtest_at_threshold(signals_df, threshold):
    """Apply trend filter + threshold=0.55 and run backtest."""
    df2 = signals_df.copy()
    df2["signal"] = (df2["buy_prob"] >= threshold).astype(int) - \
                    (df2["sell_prob"] >= threshold).astype(int)
    conflict = (df2["buy_prob"] >= threshold) & (df2["sell_prob"] >= threshold)
    df2.loc[conflict, "signal"] = 0
    # apply trend filter
    df2 = apply_trend_filter(df2)
    df2["signal"] = df2["signal_trend_filtered"]
    trades, eq = run_backtest(df2)
    m = compute_metrics(trades, eq)
    if "error" in m or m.get("n_trades", 0) == 0:
        return 0, float("nan"), float("nan")
    return int(m["n_trades"]), float(m["profit_factor"]), float(m["expectancy"])


# -- Main sweep ----------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print(f"  LABEL_ATR_MULT Sweep: {SWEEP_MULTS}")
    print(f"{'='*60}\n")

    # Load + features once (features don't depend on LABEL_ATR_MULT)
    print("[1] Loading and preparing data...")
    raw_df = load_ohlcv(DATA_PATH)
    raw_df = generate_features(raw_df)
    raw_df = add_regime_columns(raw_df)
    raw_df = add_session_column(raw_df)
    print(f"    {len(raw_df):,} bars loaded, {len(raw_df.columns)} columns\n")

    n_test_bars = int(len(raw_df) * config.TEST_SIZE)  # approx

    results = {}

    for mult in SWEEP_MULTS:
        print(f"{'-'*60}")
        print(f"  LABEL_ATR_MULT = {mult}")
        print(f"{'-'*60}")

        # Patch multiplier
        config.LABEL_ATR_MULT = mult

        # Relabel + drop NaN
        df = add_labels(raw_df.copy())
        df.dropna(inplace=True)

        n_buy    = int((df["label"] ==  1).sum())
        n_sell   = int((df["label"] == -1).sum())
        n_neut   = int((df["label"] ==  0).sum())
        print(f"  Labels: {n_buy} buy  |  {n_sell} sell  |  {n_neut} neutral")

        # Train/test split
        feature_cols = get_feature_columns(df)
        split        = int(len(df) * (1 - config.TEST_SIZE))
        train_df     = df.iloc[:split]
        test_df      = df.iloc[split:]

        # Train model
        print(f"  Training model on {len(train_df):,} bars...")
        model = fit_model(train_df, feature_cols)

        # Generate signals on test set
        signals_df = apply_signals(model, feature_cols, test_df)
        n_test = len(signals_df)

        # Probability distribution
        buy_above  = {t: int((signals_df["buy_prob"]  >= t).sum()) for t in THRESHOLDS}
        sell_above = {t: int((signals_df["sell_prob"] >= t).sum()) for t in THRESHOLDS}

        print(f"  Prob dist ({n_test:,} test bars):")
        print(f"    {'Thr':>5}  {'BUY>=':>7}  {'SELL>=':>8}")
        for t in THRESHOLDS:
            print(f"    {t:.2f}  {buy_above[t]:>7}  {sell_above[t]:>8}")

        # Threshold sweep (using both buy and sell signals)
        print(f"  Running threshold sweep...")
        sweep = _run_sweep(signals_df, feature_cols, THRESHOLDS)

        # Walk-forward
        print(f"  Running walk-forward...")
        wf_pf, wf_trades = _run_wf(df, feature_cols)

        # Final backtest at threshold 0.55 (with trend filter)
        bt_trades, bt_pf, bt_exp = _run_backtest_at_threshold(signals_df, 0.55)
        print(f"  Backtest @0.55: {bt_trades} trades  PF={bt_pf:.2f}  exp=${bt_exp:.2f}\n")

        results[mult] = {
            "n_buy": n_buy, "n_sell": n_sell, "n_neutral": n_neut,
            "buy_above": buy_above, "sell_above": sell_above,
            "sweep": sweep,
            "wf_pf": wf_pf, "wf_trades": wf_trades,
            "bt55_trades": bt_trades, "bt55_pf": bt_pf, "bt55_exp": bt_exp,
        }

    # Add benchmarks
    all_results = {**BENCHMARKS, **results}
    all_mults   = sorted(all_results.keys())

    # -- Print comparison table ------------------------------------------------
    col_w = 12
    lbl_w = 30
    sep   = "-" * (lbl_w + col_w * len(all_mults) + 2)

    def _hdr(all_mults):
        return f"  {'':>{lbl_w}}" + "".join(f"{'mult='+str(m):>{col_w}}" for m in all_mults)

    def _row(label, values):
        return f"  {label:<{lbl_w}}" + "".join(f"{str(v):>{col_w}}" for v in values)

    def _fmt(v, fmt=""):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        try:
            return format(v, fmt)
        except Exception:
            return str(v)

    print(f"\n{'='*60}")
    print(f"  Crossover Comparison — All LABEL_ATR_MULT Values")
    print(f"{'='*60}")
    print(f"  {sep}")
    print(_hdr(all_mults))
    print(f"  {sep}")

    # Class distribution
    print(f"  {'-- Class distribution --':<{lbl_w}}")
    for key, label in [("n_buy","BUY labels"), ("n_sell","SELL labels"), ("n_neutral","NEUTRAL labels")]:
        vals = [f"{all_results[m][key]:,}" for m in all_mults]
        print(_row(f"  {label}", vals))

    print(f"  {sep}")

    # Probability distribution
    print(f"  {'-- Bars above threshold --':<{lbl_w}}")
    for t in THRESHOLDS:
        vals = [f"B:{all_results[m]['buy_above'][t]}/S:{all_results[m]['sell_above'][t]}"
                for m in all_mults]
        print(_row(f"  BUY/SELL >= {t:.2f}", vals))

    print(f"  {sep}")

    # Threshold sweep
    print(f"  {'-- Threshold sweep --':<{lbl_w}}")
    for t in THRESHOLDS:
        for sub, idx, fmt in [("trades", 0, "d"), ("PF", 1, ".2f"), ("exp$", 2, ".2f")]:
            vals = []
            for m in all_mults:
                v = all_results[m]["sweep"].get(t, (0, float("nan"), float("nan")))[idx]
                vals.append(_fmt(v, fmt))
            label = f"  @{t:.2f} {sub}"
            print(_row(label, vals))

    print(f"  {sep}")

    # Walk-forward
    print(f"  {'-- Walk-forward --':<{lbl_w}}")
    for wi in range(3):
        label_pf = f"  WF win{wi+1} PF"
        label_tr = f"  WF win{wi+1} trades"
        pf_vals, tr_vals = [], []
        for m in all_mults:
            pf_list = all_results[m]["wf_pf"]
            tr_list = all_results[m]["wf_trades"]
            if wi < len(pf_list):
                pf_vals.append(_fmt(pf_list[wi], ".2f"))
                tr_vals.append(str(tr_list[wi]))
            else:
                pf_vals.append("N/A")
                tr_vals.append("N/A")
        print(_row(label_pf, pf_vals))
        print(_row(label_tr, tr_vals))

    # WF averages
    avg_pf_vals = []
    for m in all_mults:
        pf_list = [p for p in all_results[m]["wf_pf"] if not np.isnan(p)]
        avg_pf_vals.append(_fmt(np.mean(pf_list) if pf_list else float("nan"), ".2f"))
    print(_row("  WF avg PF", avg_pf_vals))

    print(f"  {sep}")

    # Final backtest at 0.55
    print(f"  {'-- Backtest at threshold 0.55 --':<{lbl_w}}")
    for key, label, fmt in [
        ("bt55_trades", "Trades", "d"),
        ("bt55_pf",     "PF",     ".2f"),
        ("bt55_exp",    "Exp $",  ".2f"),
    ]:
        vals = [_fmt(all_results[m][key], fmt) for m in all_mults]
        print(_row(f"  {label}", vals))

    # Trades per day
    test_days = n_test_bars / 24
    print(_row("  Trades/day", [
        _fmt(all_results[m]["bt55_trades"] / test_days, ".2f") for m in all_mults
    ]))

    print(f"  {sep}\n")


if __name__ == "__main__":
    main()

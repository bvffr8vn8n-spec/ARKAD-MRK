"""
experiments/horizon_sweep.py — Sweep FORWARD_RETURN_WINDOW at fixed LABEL_ATR_MULT=0.85.

Tests 12H and 16H as candidate middle-ground horizons between:
  8H  — high frequency (0.81/d) but edge collapsed (PF=0.54 WF=0.47)
  24H — confirmed edge (PF=1.16 BT, WF=0.64) but too rare (0.23/d)

TP/SL scaled proportionally to each horizon's median reachable move:
  8H  : TP=1.5, SL=1.0  R:R=1.50  (median move ~0.93 ATR)
  12H : TP=1.8, SL=1.1  R:R=1.64  (median move ~1.20 ATR estimated)
  16H : TP=2.0, SL=1.2  R:R=1.67  (median move ~1.50 ATR estimated)
  24H : TP=2.5, SL=1.5  R:R=1.67  (median move ~1.87 ATR)

LABEL_ATR_MULT fixed at 0.85 throughout.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.intraday import add_5m_features
from features.market_regime import add_regime_columns, add_session_column, apply_trend_filter
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


DATA_PATH       = "data/BTCUSDT_1h_4y.csv"
DATA_PATH_5M    = "data/BTCUSDT_5m_4y.csv"   # set to None to disable intraday features
LABEL_ATR_MULT  = 0.85

# Horizon configs: (FORWARD_RETURN_WINDOW, HOLD_BARS, TP_MULT, SL_MULT)
HORIZONS = [
    (12, 12, 1.8, 1.1),
    (16, 16, 2.0, 1.2),
]

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60]

# Benchmarks from completed runs (8H and 24H at LABEL_ATR_MULT=0.85)
BENCHMARKS = {
    "8H": {
        "fwd": 8, "tp": 1.5, "sl": 1.0,
        "n_buy": 9449, "n_sell": 9050, "n_neutral": 16358,
        "median_atr": 0.93,
        "buy_above": {0.50: 3601, 0.52: 2341, 0.55: 375, 0.58: 5, 0.60: 1},
        "sell_above": {0.50: 3371, 0.52: 1859, 0.55: 107, 0.58: 18, 0.60: 15},
        "buy_max": 0.6123, "sell_max": 0.7814,
        "sweep": {
            0.50: (1179, 0.57, -1.99),
            0.52: (836,  0.61, -1.92),
            0.55: (185,  0.62, -2.16),
            0.58: (3,    0.02, -6.67),
            0.60: (1,    0.00, -16.26),
        },
        "wf_pf": [0.46, 0.44, 0.52], "wf_trades": [29, 55, 71],
        "bt55_trades": 236, "bt55_pf": 0.54, "bt55_exp": -2.70,
    },
    "24H": {
        "fwd": 24, "tp": 2.5, "sl": 1.5,
        "n_buy": 13067, "n_sell": 12428, "n_neutral": 9346,
        "median_atr": 1.87,
        "buy_above": {0.50: 3993, 0.52: 2710, 0.55: 118, 0.58: 25, 0.60: 17},
        "sell_above": {0.50: 2976, 0.52: 1775, 0.55: 74, 0.58: 17, 0.60: 11},
        "buy_max": 0.72, "sell_max": 0.74,
        "sweep": {
            0.50: (773, 0.70, -2.28),
            0.52: (657, 0.70, -2.40),
            0.55: (70,  1.11, 0.78),
            0.58: (21,  2.35, 9.05),
            0.60: (14,  2.24, 8.09),
        },
        "wf_pf": [0.58, 0.67, 0.68], "wf_trades": [199, 104, 48],
        "bt55_trades": 66, "bt55_pf": 1.16, "bt55_exp": 1.10,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wf_windows(df):
    n = len(df)
    windows = []
    for tf in config.WF_TRAIN_FRACTIONS:
        te = tf + config.WF_TEST_FRACTION
        windows.append((0, int(n * tf), min(int(n * te), n), tf, te))
    return windows


def _run_sweep(signals_df, thresholds):
    rows = {}
    for t in thresholds:
        df2 = signals_df.copy()
        buy_sig  = df2["buy_prob"]  >= t
        sell_sig = df2["sell_prob"] >= t
        conflict = buy_sig & sell_sig
        df2["signal"] = buy_sig.astype(int) - sell_sig.astype(int)
        df2.loc[conflict, "signal"] = 0
        trades, eq = run_backtest(df2)
        m = compute_metrics(trades, eq)
        if "error" in m or m.get("n_trades", 0) == 0:
            rows[t] = (0, float("nan"), float("nan"))
        else:
            rows[t] = (int(m["n_trades"]), float(m["profit_factor"]),
                       float(m["expectancy"]))
    return rows


def _run_wf(df, feature_cols):
    windows  = _wf_windows(df)
    pf_list, trade_list = [], []
    for (ts, te, tend, tf, tef) in windows:
        purge = te - config.FORWARD_RETURN_WINDOW
        train = df.iloc[ts:purge]
        test  = df.iloc[te:tend]
        if len(train) < 50 or len(test) < 5:
            continue
        model      = fit_model(train, feature_cols)
        sig_df     = apply_signals(model, feature_cols, test)
        trades, eq = run_backtest(sig_df)
        m  = compute_metrics(trades, eq)
        n  = int(m.get("n_trades", 0) or 0)
        pf = float(m.get("profit_factor", float("nan")) or float("nan"))
        pf_list.append(pf)
        trade_list.append(n)
        print(f"    WF [{tf*100:.0f}-{tef*100:.0f}%]: {n} trades  PF={pf:.2f}")
    return pf_list, trade_list


def _backtest_at_threshold(signals_df, threshold):
    df2 = signals_df.copy()
    buy_sig  = df2["buy_prob"]  >= threshold
    sell_sig = df2["sell_prob"] >= threshold
    conflict = buy_sig & sell_sig
    df2["signal"] = buy_sig.astype(int) - sell_sig.astype(int)
    df2.loc[conflict, "signal"] = 0
    df2 = apply_trend_filter(df2)
    df2["signal"] = df2["signal_trend_filtered"]
    trades, eq = run_backtest(df2)
    m = compute_metrics(trades, eq)
    if "error" in m or m.get("n_trades", 0) == 0:
        return 0, float("nan"), float("nan")
    return int(m["n_trades"]), float(m["profit_factor"]), float(m["expectancy"])


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Horizon Sweep (LABEL_ATR_MULT={LABEL_ATR_MULT})")
    print(f"  Testing: {[h[0] for h in HORIZONS]}H")
    print(f"{'='*60}\n")

    # Load + features once
    print("[1] Loading and preparing data...")
    raw_df = load_ohlcv(DATA_PATH)
    raw_df = generate_features(raw_df)
    if DATA_PATH_5M:
        print(f"    Loading 5m intraday features from {DATA_PATH_5M}...")
        raw_df = add_5m_features(raw_df, DATA_PATH_5M)
    raw_df = add_regime_columns(raw_df)
    raw_df = add_session_column(raw_df)
    print(f"    {len(raw_df):,} bars, {len(raw_df.columns)} columns\n")

    n_test_bars = int(len(raw_df) * config.TEST_SIZE)
    test_days   = n_test_bars / 24   # 1H bars

    results = {}
    config.LABEL_ATR_MULT = LABEL_ATR_MULT

    for (fwd, hold, tp, sl) in HORIZONS:
        label = f"{fwd}H"
        print(f"{'-'*60}")
        print(f"  Horizon = {label}  |  TP={tp}xATR  SL={sl}xATR  R:R={tp/sl:.2f}")
        print(f"{'-'*60}")

        # Patch config
        config.FORWARD_RETURN_WINDOW = fwd
        config.HOLD_BARS             = hold
        config.TAKE_PROFIT_ATR_MULT  = tp
        config.STOP_LOSS_ATR_MULT    = sl

        # Relabel + drop NaN
        df = add_labels(raw_df.copy())
        df.dropna(inplace=True)

        n_buy    = int((df["label"] ==  1).sum())
        n_sell   = int((df["label"] == -1).sum())
        n_neut   = int((df["label"] ==  0).sum())
        med_atr  = float(df["fwd_return_atr"].abs().median())
        print(f"  Labels: {n_buy} buy | {n_sell} sell | {n_neut} neutral")
        print(f"  Median |fwd_return_atr| = {med_atr:.2f} ATR units")

        # Train/test split
        feature_cols = get_feature_columns(df)
        split        = int(len(df) * (1 - config.TEST_SIZE))
        train_df     = df.iloc[:split]
        test_df      = df.iloc[split:]

        print(f"  Training on {len(train_df):,} bars...")
        model      = fit_model(train_df, feature_cols)
        signals_df = apply_signals(model, feature_cols, test_df)
        n_test     = len(signals_df)

        # Probability distribution
        buy_above  = {t: int((signals_df["buy_prob"]  >= t).sum()) for t in THRESHOLDS}
        sell_above = {t: int((signals_df["sell_prob"] >= t).sum()) for t in THRESHOLDS}
        buy_max    = float(signals_df["buy_prob"].max())
        sell_max   = float(signals_df["sell_prob"].max())

        print(f"  Prob dist ({n_test:,} test bars)  buy_max={buy_max:.4f}  sell_max={sell_max:.4f}")
        print(f"    {'Thr':>5}  {'BUY>=':>7}  {'SELL>=':>8}")
        for t in THRESHOLDS:
            print(f"    {t:.2f}  {buy_above[t]:>7}  {sell_above[t]:>8}")

        # Threshold sweep
        print(f"  Running threshold sweep...")
        sweep = _run_sweep(signals_df, THRESHOLDS)

        # Walk-forward
        print(f"  Running walk-forward...")
        wf_pf, wf_trades = _run_wf(df, feature_cols)

        # Final backtest at 0.55 with trend filter
        bt_trades, bt_pf, bt_exp = _backtest_at_threshold(signals_df, 0.55)
        print(f"  Backtest @0.55: {bt_trades} trades  PF={bt_pf:.2f}  exp=${bt_exp:.2f}\n")

        results[label] = {
            "fwd": fwd, "tp": tp, "sl": sl,
            "n_buy": n_buy, "n_sell": n_sell, "n_neutral": n_neut,
            "median_atr": med_atr,
            "buy_above": buy_above, "sell_above": sell_above,
            "buy_max": buy_max, "sell_max": sell_max,
            "sweep": sweep,
            "wf_pf": wf_pf, "wf_trades": wf_trades,
            "bt55_trades": bt_trades, "bt55_pf": bt_pf, "bt55_exp": bt_exp,
        }

    # ── Comparison table ──────────────────────────────────────────────────────
    all_keys = ["8H"] + [f"{h[0]}H" for h in HORIZONS] + ["24H"]
    all_data = {}
    for k in all_keys:
        if k in BENCHMARKS:
            all_data[k] = BENCHMARKS[k]
        else:
            all_data[k] = results[k]

    col_w = 11
    lbl_w = 28
    sep   = "-" * (lbl_w + col_w * len(all_keys) + 2)

    def _row(label, values):
        return f"  {label:<{lbl_w}}" + "".join(f"{str(v):>{col_w}}" for v in values)

    def _fmt(v, fmt=""):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "N/A"
        try:
            return format(v, fmt)
        except Exception:
            return str(v)

    print(f"\n{'='*60}")
    print(f"  Horizon Crossover — All Configs (LABEL_ATR_MULT=0.85)")
    print(f"{'='*60}")
    print(f"  {sep}")
    header = f"  {'':>{lbl_w}}" + "".join(f"{k:>{col_w}}" for k in all_keys)
    print(header)
    print(f"  {sep}")

    # Config row
    tp_row  = [_fmt(all_data[k]["tp"],  ".1f") for k in all_keys]
    sl_row  = [_fmt(all_data[k]["sl"],  ".1f") for k in all_keys]
    rr_row  = [_fmt(all_data[k]["tp"] / all_data[k]["sl"], ".2f") for k in all_keys]
    print(_row("TP x ATR", tp_row))
    print(_row("SL x ATR", sl_row))
    print(_row("R:R", rr_row))
    print(f"  {sep}")

    # Class distribution
    print(_row("BUY labels",     [f"{all_data[k]['n_buy']:,}"     for k in all_keys]))
    print(_row("SELL labels",    [f"{all_data[k]['n_sell']:,}"    for k in all_keys]))
    print(_row("NEUTRAL labels", [f"{all_data[k]['n_neutral']:,}" for k in all_keys]))
    print(_row("Median |fwd_ATR|", [_fmt(all_data[k]["median_atr"], ".2f") for k in all_keys]))
    print(f"  {sep}")

    # Prob distribution
    print(_row("BUY max prob",  [_fmt(all_data[k].get("buy_max",  float("nan")), ".4f") for k in all_keys]))
    print(_row("SELL max prob", [_fmt(all_data[k].get("sell_max", float("nan")), ".4f") for k in all_keys]))
    for t in THRESHOLDS:
        vals = [f"B:{all_data[k]['buy_above'][t]}/S:{all_data[k]['sell_above'][t]}"
                for k in all_keys]
        print(_row(f">={t:.2f} BUY/SELL", vals))
    print(f"  {sep}")

    # Threshold sweep
    for t in THRESHOLDS:
        for sub, idx, fmt in [("trades", 0, "d"), ("PF", 1, ".2f"), ("exp$", 2, ".2f")]:
            vals = []
            for k in all_keys:
                v = all_data[k]["sweep"].get(t, (0, float("nan"), float("nan")))[idx]
                vals.append(_fmt(v, fmt))
            print(_row(f"@{t:.2f} {sub}", vals))
    print(f"  {sep}")

    # Walk-forward
    for wi in range(3):
        pf_vals = []
        tr_vals = []
        for k in all_keys:
            pl = all_data[k]["wf_pf"]
            tl = all_data[k]["wf_trades"]
            pf_vals.append(_fmt(pl[wi], ".2f") if wi < len(pl) else "N/A")
            tr_vals.append(str(tl[wi])          if wi < len(tl) else "N/A")
        print(_row(f"WF win{wi+1} PF",     pf_vals))
        print(_row(f"WF win{wi+1} trades", tr_vals))

    avg_pf_vals = []
    for k in all_keys:
        pl = [p for p in all_data[k]["wf_pf"] if not np.isnan(p)]
        avg_pf_vals.append(_fmt(np.mean(pl) if pl else float("nan"), ".2f"))
    print(_row("WF avg PF", avg_pf_vals))
    print(f"  {sep}")

    # Final backtest at 0.55
    print(_row("BT@0.55 trades",    [_fmt(all_data[k]["bt55_trades"], "d")    for k in all_keys]))
    print(_row("BT@0.55 PF",        [_fmt(all_data[k]["bt55_pf"],     ".2f")  for k in all_keys]))
    print(_row("BT@0.55 exp$",      [_fmt(all_data[k]["bt55_exp"],    ".2f")  for k in all_keys]))
    print(_row("BT@0.55 trades/day",
               [_fmt(all_data[k]["bt55_trades"] / test_days, ".2f") for k in all_keys]))
    print(f"  {sep}\n")


if __name__ == "__main__":
    main()

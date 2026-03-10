"""
experiments/filter_relax_sweep.py — Filter relaxation sweep for the 24H strategy.

Objective
---------
The confirmed 24H edge (PF=1.16, exp=+$1.10) trades only ~0.23/d at threshold=0.55.
This sweep finds how much frequency can be recovered by relaxing each filter while
preserving PF > 1 and positive expectancy.

Filters tested
--------------
  Trend filter : blocks bars in "range" regime (ALLOWED_TRENDS excludes "range")
  Vol filter   : blocks bars in "low_vol" regime
  Threshold    : minimum buy/sell probability to emit a signal

Approach
--------
  1. Load data + intraday features once.
  2. Train one RF model on the full 80% train split.
  3. Apply every combination of (trend_gate, vol_gate, threshold) to the test set.
  4. Run backtest for each combination.
  5. Print comparison table + mark the Pareto-efficient configs (PF>1, exp>0).

Configs tested
--------------
  trend_gate : True  = only trend_up / trend_down  (current)
               False = also allow range bars
  vol_gate   : True  = block low_vol bars           (current)
               False = allow low_vol bars too
  threshold  : 0.50, 0.52, 0.55 (0.58 yields < 20 trades in most configs)
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.intraday import add_5m_features
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


DATA_PATH_1H = "data/BTCUSDT_1h_4y.csv"
DATA_PATH_5M = "data/BTCUSDT_5m_4y.csv"

# Fixed 24H config
FWD         = 24
HOLD        = 24
TP          = 2.5
SL          = 1.5
ATR_MULT    = 0.85
THRESHOLDS  = [0.50, 0.52, 0.55, 0.58]


def _apply_filters(signals_df, trend_gate: bool, vol_gate: bool, threshold: float):
    """
    Apply the specified filter combination and return a backtest-ready DataFrame.

    trend_gate : if True, zero signals where trend == "range"
    vol_gate   : if True, zero signals where vol_regime == "low_vol"
    threshold  : apply buy/sell probability threshold to set final signal
    """
    df = signals_df.copy()

    # Step 1: threshold gate — re-derive signal from probabilities
    buy_sig  = df["buy_prob"]  >= threshold
    sell_sig = df["sell_prob"] >= threshold
    conflict = buy_sig & sell_sig
    signal   = buy_sig.astype(int) - sell_sig.astype(int)
    signal[conflict] = 0

    # Step 2: trend gate
    if trend_gate:
        in_trend = df["trend"].isin(["trend_up", "trend_down"])
        signal   = signal * in_trend.astype(int)

    # Step 3: vol gate
    if vol_gate:
        not_low_vol = df["vol_regime"] != "low_vol"
        signal      = signal * not_low_vol.astype(int)

    df["signal"] = signal
    return df


def _backtest(df):
    trades, eq = run_backtest(df)
    m = compute_metrics(trades, eq)
    if "error" in m or not m.get("n_trades"):
        return 0, float("nan"), float("nan"), float("nan")
    return (
        int(m["n_trades"]),
        float(m.get("profit_factor", float("nan"))),
        float(m.get("expectancy",    float("nan"))),
        float(m.get("win_rate",      float("nan"))),
    )


def _fmt(v, fmt=""):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def main():
    print(f"\n{'='*70}")
    print(f"  Filter Relaxation Sweep  |  24H horizon, LABEL_ATR_MULT={ATR_MULT}")
    print(f"  TP={TP}xATR  SL={SL}xATR  R:R={TP/SL:.2f}")
    print(f"{'='*70}\n")

    # ── Patch config for 24H ─────────────────────────────────────────────────
    config.FORWARD_RETURN_WINDOW = FWD
    config.HOLD_BARS             = HOLD
    config.TAKE_PROFIT_ATR_MULT  = TP
    config.STOP_LOSS_ATR_MULT    = SL
    config.LABEL_ATR_MULT        = ATR_MULT

    # ── Load and prepare data ────────────────────────────────────────────────
    print("[1] Loading data...")
    raw_df = load_ohlcv(DATA_PATH_1H)
    raw_df = generate_features(raw_df)
    print(f"    Loading 5m intraday features...")
    raw_df = add_5m_features(raw_df, DATA_PATH_5M)
    raw_df = add_regime_columns(raw_df)
    raw_df = add_session_column(raw_df)
    print(f"    {len(raw_df):,} bars, {len(raw_df.columns)} columns\n")

    # ── Label and split ──────────────────────────────────────────────────────
    df = add_labels(raw_df.copy())
    df.dropna(inplace=True)

    n_buy  = int((df["label"] ==  1).sum())
    n_sell = int((df["label"] == -1).sum())
    n_neut = int((df["label"] ==  0).sum())
    print(f"[2] Labels: {n_buy:,} buy | {n_sell:,} sell | {n_neut:,} neutral\n")

    feature_cols = get_feature_columns(df)
    split        = int(len(df) * (1 - config.TEST_SIZE))
    train_df     = df.iloc[:split]
    test_df      = df.iloc[split:]
    test_days    = len(test_df) / 24

    # ── Train model once ─────────────────────────────────────────────────────
    print(f"[3] Training model on {len(train_df):,} bars...")
    model      = fit_model(train_df, feature_cols)
    signals_df = apply_signals(model, feature_cols, test_df)
    n_test     = len(signals_df)

    buy_max  = float(signals_df["buy_prob"].max())
    sell_max = float(signals_df["sell_prob"].max())
    print(f"    {n_test:,} test bars  (~{test_days:.0f} days)")
    print(f"    buy_prob max={buy_max:.4f}  sell_prob max={sell_max:.4f}\n")

    # ── Regime counts in test set ────────────────────────────────────────────
    trend_counts = signals_df["trend"].value_counts()
    vol_counts   = signals_df["vol_regime"].value_counts()
    print(f"    Test-set regime distribution:")
    for lbl in ["trend_up", "range", "trend_down"]:
        n = trend_counts.get(lbl, 0)
        print(f"      {lbl:<14} {n:>5} bars  ({n/n_test*100:.1f}%)")
    for lbl in ["low_vol", "normal_vol", "high_vol"]:
        n = vol_counts.get(lbl, 0)
        print(f"      {lbl:<14} {n:>5} bars  ({n/n_test*100:.1f}%)")
    print()

    # ── Probability distribution ─────────────────────────────────────────────
    print(f"    Probability distribution (all {n_test:,} test bars):")
    print(f"      {'Thr':>5}  {'BUY>=':>7}  {'SELL>=':>8}  {'Total':>7}")
    for t in THRESHOLDS:
        nb = int((signals_df["buy_prob"]  >= t).sum())
        ns = int((signals_df["sell_prob"] >= t).sum())
        print(f"      {t:.2f}  {nb:>7}  {ns:>8}  {nb+ns:>7}")
    print()

    # ── Run all filter combinations ──────────────────────────────────────────
    print("[4] Running filter combinations...\n")

    configs = []
    for trend_gate in [True, False]:
        for vol_gate in [True, False]:
            for thr in THRESHOLDS:
                label = (
                    f"trend={'ON' if trend_gate else 'OFF'} "
                    f"vol={'ON' if vol_gate else 'OFF'} "
                    f"thr={thr:.2f}"
                )
                df2 = _apply_filters(signals_df, trend_gate, vol_gate, thr)
                n_sig = int((df2["signal"] != 0).sum())
                trades, pf, exp, wr = _backtest(df2)
                tpd = trades / test_days if test_days > 0 else 0.0
                profitable = (
                    isinstance(pf, float) and not np.isnan(pf) and pf > 1.0 and
                    isinstance(exp, float) and not np.isnan(exp) and exp > 0
                )
                configs.append({
                    "label":       label,
                    "trend_gate":  trend_gate,
                    "vol_gate":    vol_gate,
                    "threshold":   thr,
                    "n_signals":   n_sig,
                    "trades":      trades,
                    "pf":          pf,
                    "exp":         exp,
                    "win_rate":    wr,
                    "tpd":         tpd,
                    "profitable":  profitable,
                })
                flag = " <-- PF>1 exp>0" if profitable else ""
                print(f"    {label:<40}  "
                      f"sigs={n_sig:>4}  trades={trades:>4}  "
                      f"PF={_fmt(pf, '.2f'):>5}  exp=${_fmt(exp, '.2f'):>6}  "
                      f"{flag}")

    # ── Summary table ────────────────────────────────────────────────────────
    col_w = 10
    lbl_w = 38
    sep   = "-" * (lbl_w + col_w * 6 + 4)

    print(f"\n{'='*70}")
    print(f"  Summary Table  (baseline = trend=ON vol=ON thr=0.55)")
    print(f"{'='*70}")
    print(f"  {sep}")
    hdr = (f"  {'Config':<{lbl_w}}"
           f"{'Signals':>{col_w}}"
           f"{'Trades':>{col_w}}"
           f"{'t/day':>{col_w}}"
           f"{'PF':>{col_w}}"
           f"{'Exp $':>{col_w}}"
           f"{'WR':>{col_w}}")
    print(hdr)
    print(f"  {sep}")

    for c in configs:
        flag = " *" if c["profitable"] else ""
        row = (f"  {c['label'] + flag:<{lbl_w}}"
               f"{c['n_signals']:>{col_w}}"
               f"{c['trades']:>{col_w}}"
               f"{_fmt(c['tpd'], '.2f'):>{col_w}}"
               f"{_fmt(c['pf'],  '.2f'):>{col_w}}"
               f"{_fmt(c['exp'], '.2f'):>{col_w}}"
               f"{_fmt(c['win_rate'], '.1%'):>{col_w}}")
        print(row)

    print(f"  {sep}")
    print(f"  * = PF > 1 and expectancy > 0\n")

    # ── Pareto-efficient configs ─────────────────────────────────────────────
    profitable_cfgs = [c for c in configs if c["profitable"]]
    if profitable_cfgs:
        print(f"  Profitable configs ({len(profitable_cfgs)} found):")
        print(f"  {'Config':<40}  {'t/day':>6}  {'PF':>6}  {'Exp $':>7}")
        for c in sorted(profitable_cfgs, key=lambda x: -x["tpd"]):
            print(f"  {c['label']:<40}  "
                  f"{_fmt(c['tpd'], '.2f'):>6}  "
                  f"{_fmt(c['pf'], '.2f'):>6}  "
                  f"{_fmt(c['exp'], '.2f'):>7}")
    else:
        print("  No configs achieved both PF > 1 and expectancy > 0.")

    print()


if __name__ == "__main__":
    main()

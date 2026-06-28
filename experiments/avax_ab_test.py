"""
experiments/avax_ab_test.py
— A/B test for AVAX BT performance:
  A: train without TRAINING_START_DATE (use everything in CSV before cutoff)
  B: train WITH TRAINING_START_DATE = 2022-06-01 (current production)

Both runs use the SAME test window (>= TRAINING_CUTOFF_DATE).
Differences in trades / PF / sumR isolate the START anchor's effect on AVAX.

If A and B are close → the START anchor isn't the problem; AVAX's 2026
weakness is just the period.
If A is materially better than B → our anchor hurt AVAX, revisit.

Usage:
    python experiments/avax_ab_test.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd

import config
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column,
    apply_trend_filter, apply_vol_filter,
)
from features.execution_15m import load_5m_as_15m, annotate_signals_AB
from models.classifier import fit_model, get_feature_columns, apply_signals
from backtest.engine import run_backtest


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
ASSET = "AVAXUSDT"


def _build_signals(train_slice: pd.DataFrame, test_slice: pd.DataFrame,
                   df_15m: pd.DataFrame | None) -> pd.DataFrame:
    """Train, score, apply filters + 15m-AB execution layer."""
    feat_cols = get_feature_columns(train_slice)
    model     = fit_model(train_slice, feat_cols)
    scored    = apply_signals(model, feat_cols, test_slice.copy())

    filt = apply_trend_filter(scored)
    filt["signal"] = filt["signal_trend_filtered"]
    filt = apply_vol_filter(filt)
    filt["signal"] = filt["signal_vol_filtered"]

    if df_15m is not None:
        annotated = annotate_signals_AB(filt, df_15m)
        annotated["signal"] = annotated["signal_15m_A"]
        return annotated
    return filt


def _summarise(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": float("nan"), "sumR": 0.0, "avgR": 0.0}
    rs   = [t["R"]      for t in trades]
    pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gw   = sum(p for p in pnls if p > 0)
    gl   = abs(sum(p for p in pnls if p <= 0))
    return {
        "n":    len(trades),
        "wr":   wins / len(trades),
        "pf":   gw / gl if gl > 0 else float("inf"),
        "sumR": sum(rs),
        "avgR": sum(rs) / len(rs),
    }


def main() -> None:
    print("=" * 80)
    print(f"  ARKAD MRK — AVAX A/B test: TRAINING_START_DATE on/off")
    print("=" * 80)

    df = pd.read_csv(os.path.join(DATA_DIR, f"{ASSET}_1h_4y.csv"),
                     parse_dates=["date"]).set_index("date")

    df_feat = generate_features(df.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    start  = pd.Timestamp(config.TRAINING_START_DATE)
    cutoff = pd.Timestamp(config.TRAINING_CUTOFF_DATE)

    train_A = df_feat[df_feat.index < cutoff]                                  # no START
    train_B = df_feat[(df_feat.index >= start) & (df_feat.index < cutoff)]      # with START
    test    = df_feat[df_feat.index >= cutoff]

    print(f"  CSV span:           {df.index[0]} → {df.index[-1]}")
    print(f"  CUTOFF date:        {cutoff.date()}")
    print(f"  Train A (no START): {len(train_A):>6} bars  "
          f"({train_A.index[0].date()} → {train_A.index[-1].date()})")
    print(f"  Train B (=START):   {len(train_B):>6} bars  "
          f"({train_B.index[0].date()} → {train_B.index[-1].date()})")
    print(f"  Test window:        {len(test):>6} bars  "
          f"({test.index[0].date()} → {test.index[-1].date()})")
    print(f"  Δ in train size:    {len(train_A) - len(train_B):>6} bars "
          f"({(len(train_A) - len(train_B)) / len(train_A) * 100:.1f}% of A)")

    # Load 15m execution layer if available
    five_min_path = os.path.join(DATA_DIR, f"{ASSET}_5m_4y.csv")
    df_15m = load_5m_as_15m(five_min_path) if os.path.exists(five_min_path) else None
    print(f"  15m-AB execution:   {'yes' if df_15m is not None else 'no (5m CSV missing)'}")

    print()
    print(f"  Building signals A (no START anchor) ...")
    sig_A = _build_signals(train_A, test, df_15m)
    print(f"  Building signals B (current production) ...")
    sig_B = _build_signals(train_B, test, df_15m)

    print()
    print(f"  Signals in test window (non-zero):")
    print(f"    A: {(sig_A['signal'] != 0).sum()}")
    print(f"    B: {(sig_B['signal'] != 0).sum()}")

    trades_A, _ = run_backtest(sig_A, exit_mode="scaled")
    trades_B, _ = run_backtest(sig_B, exit_mode="scaled")

    sA = _summarise(trades_A)
    sB = _summarise(trades_B)

    print()
    print(f"  {'Metric':<10}  {'A (no START)':>14}  {'B (current)':>14}  {'B − A':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'N trades':<10}  {sA['n']:>14}  {sB['n']:>14}  "
          f"{sB['n'] - sA['n']:>+10}")
    print(f"  {'WR':<10}  {sA['wr']*100:>13.1f}%  {sB['wr']*100:>13.1f}%  "
          f"{(sB['wr'] - sA['wr'])*100:>+10.1f} pp")
    pfA = f"{sA['pf']:.3f}" if sA['pf'] != float('inf') else 'inf'
    pfB = f"{sB['pf']:.3f}" if sB['pf'] != float('inf') else 'inf'
    print(f"  {'PF':<10}  {pfA:>14}  {pfB:>14}  "
          f"{sB['pf'] - sA['pf']:>+10.3f}")
    print(f"  {'Sum R':<10}  {sA['sumR']:>+14.2f}  {sB['sumR']:>+14.2f}  "
          f"{sB['sumR'] - sA['sumR']:>+10.2f}")
    print(f"  {'Avg R':<10}  {sA['avgR']:>+14.3f}  {sB['avgR']:>+14.3f}  "
          f"{sB['avgR'] - sA['avgR']:>+10.3f}")

    print()
    print(f"  {'Verdict':<10}")
    print(f"  {'-' * 60}")
    delta_sumR = sB['sumR'] - sA['sumR']
    if abs(delta_sumR) < 1.0:
        print(f"  Δ Sum R = {delta_sumR:+.2f} R  → noise / period, not the START anchor")
    elif delta_sumR < -2.0:
        print(f"  Δ Sum R = {delta_sumR:+.2f} R  → START anchor HURT AVAX materially")
    elif delta_sumR > 2.0:
        print(f"  Δ Sum R = {delta_sumR:+.2f} R  → START anchor HELPED AVAX (unexpected)")
    else:
        print(f"  Δ Sum R = {delta_sumR:+.2f} R  → marginal, leans toward period")
    print()


if __name__ == "__main__":
    main()

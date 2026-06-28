"""
experiments/early_kill_sweep.py
— Backtest sweep over (checkpoint_hour, MFE_threshold) pairs to bracket the
upside of an early-kill rule on top of the scaled-exit engine.

For each Tier-1 asset:
  1. Load 1H + 5m CSV (merging recent data if present)
  2. Train CalibratedRF on bars in [TRAINING_START_DATE, TRAINING_CUTOFF_DATE)
  3. Generate signals on bars >= TRAINING_CUTOFF_DATE (the OOS / live window)
  4. Apply 15m-AB execution layer (same as paper trader)
  5. Run baseline backtest (no early kill) — establishes the reference
  6. For each (h, thr) config, run backtest with that early-kill rule
  7. Print per-config summary; aggregate across assets at the end

The intent is NOT to find the single best parameter — this is a hypothesis-
bracketing exercise.  Pair the BT output with the live `mfe_r_h6/h12/h18`
data that the 2026-06-XX trade_manager change is now logging; if BT and
paper agree on a corner, that corner is the candidate to deploy.

Usage:
    python experiments/early_kill_sweep.py
    python experiments/early_kill_sweep.py --hours 6,12,18 --thrs 0.2,0.3,0.4
    python experiments/early_kill_sweep.py --asset AVAXUSDT
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

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

ASSETS = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]

DEFAULT_HOURS = [6, 12, 18]
DEFAULT_THRS  = [0.20, 0.30, 0.40]


def _load_1h(symbol: str) -> pd.DataFrame:
    """Load 4y 1H CSV and optionally merge any recent_1h delta."""
    main = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_1h_4y.csv"),
                       parse_dates=["date"]).set_index("date")
    recent_path = os.path.join(DATA_DIR, f"{symbol}_recent_1h.csv")
    if os.path.exists(recent_path):
        rec = pd.read_csv(recent_path, parse_dates=["date"]).set_index("date")
        main = pd.concat([main, rec])
        main = main[~main.index.duplicated(keep="last")].sort_index()
    return main


def _build_signals(symbol: str) -> pd.DataFrame | None:
    """
    Train model on [START, CUTOFF), score on >= CUTOFF, apply trend+vol filters
    and the 15m-AB execution layer.  Returns a DataFrame ready for run_backtest.
    """
    df = _load_1h(symbol)
    if len(df) == 0:
        print(f"  {symbol}: no 1H data; skipping")
        return None

    df_feat = generate_features(df.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    start  = pd.Timestamp(config.TRAINING_START_DATE)
    cutoff = pd.Timestamp(config.TRAINING_CUTOFF_DATE)
    train  = df_feat[(df_feat.index >= start) & (df_feat.index < cutoff)]
    test   = df_feat[df_feat.index >= cutoff]

    if len(train) < 200 or len(test) < 50:
        print(f"  {symbol}: train={len(train)} test={len(test)} — too short, skipping")
        return None

    feat_cols = get_feature_columns(train)
    model     = fit_model(train, feat_cols)
    scored    = apply_signals(model, feat_cols, test.copy())

    filt = apply_trend_filter(scored)
    filt["signal"] = filt["signal_trend_filtered"]
    filt = apply_vol_filter(filt)
    filt["signal"] = filt["signal_vol_filtered"]

    # 15m-AB execution layer (if 5m data is available)
    five_min_path = os.path.join(DATA_DIR, f"{symbol}_5m_4y.csv")
    if os.path.exists(five_min_path):
        df_15m = load_5m_as_15m(five_min_path)
        annotated = annotate_signals_AB(filt, df_15m)
        annotated["signal"] = annotated["signal_15m_A"]
        return annotated

    return filt


def _summarise(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": float("nan"), "sumR": 0.0,
                "avgR": 0.0, "tp": 0, "stop": 0, "time": 0, "kill": 0}
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
        "tp":   sum(1 for t in trades if t["exit_reason"] == "tp"),
        "stop": sum(1 for t in trades if t["exit_reason"] == "stop"),
        "time": sum(1 for t in trades if t["exit_reason"] == "time"),
        "kill": sum(1 for t in trades if t["exit_reason"] == "early_kill"),
    }


def _row(label: str, s: dict, base: dict | None = None) -> str:
    delta_sumR = f"({s['sumR'] - base['sumR']:+.2f})" if base else ""
    pf_str = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
    return (
        f"  {label:<25}  n={s['n']:>3}  "
        f"WR={s['wr']*100:>4.1f}%  PF={pf_str:>5}  "
        f"sumR={s['sumR']:>+6.2f}{delta_sumR:<9}  "
        f"avgR={s['avgR']:>+6.3f}  "
        f"TP/Stop/Time/Kill={s['tp']}/{s['stop']}/{s['time']}/{s['kill']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None,
                    help="Run on a single asset (default: all Tier-1)")
    ap.add_argument("--hours", default=",".join(str(h) for h in DEFAULT_HOURS),
                    help=f"Checkpoint hours, comma-sep (default: {DEFAULT_HOURS})")
    ap.add_argument("--thrs", default=",".join(str(t) for t in DEFAULT_THRS),
                    help=f"MFE thresholds (R-units), comma-sep (default: {DEFAULT_THRS})")
    args = ap.parse_args()

    assets = [args.asset.upper()] if args.asset else ASSETS
    hours  = [int(h.strip())   for h in args.hours.split(",")]
    thrs   = [float(t.strip()) for t in args.thrs.split(",")]

    sep = "=" * 100
    print(sep)
    print(f"  ARKAD MRK — Early-Kill BT Sweep")
    print(f"  Window: bars >= {config.TRAINING_CUTOFF_DATE}  |  "
          f"checkpoint hours: {hours}  |  MFE thresholds: {thrs}")
    print(sep)

    # Aggregator across assets — sum(R) per (h, thr); baseline + each config
    agg = defaultdict(lambda: {"sumR": 0.0, "n": 0, "kills": 0})

    for asset in assets:
        print(f"\n  ── {asset}  ──────────────────────────────────────────────────────────────")
        sig = _build_signals(asset)
        if sig is None:
            continue
        print(f"  signals window: {sig.index[0]} → {sig.index[-1]} "
              f"({len(sig)} bars, {(sig['signal'] != 0).sum()} non-zero signals)")

        base_trades, _ = run_backtest(sig, exit_mode="scaled")
        base = _summarise(base_trades)
        print(_row("BASELINE", base))
        agg[("baseline", None)]["sumR"] += base["sumR"]
        agg[("baseline", None)]["n"]    += base["n"]

        for h in hours:
            for thr in thrs:
                trades, _ = run_backtest(
                    sig, exit_mode="scaled",
                    early_kill={"at_h": h, "mfe_thr": thr},
                )
                s = _summarise(trades)
                print(_row(f"kill h={h:>2}, thr={thr:.2f}", s, base))
                agg[(h, thr)]["sumR"]  += s["sumR"]
                agg[(h, thr)]["n"]     += s["n"]
                agg[(h, thr)]["kills"] += s["kill"]

    # Aggregate table
    print()
    print(sep)
    print(f"  AGGREGATE across {len(assets)} asset(s)  — sum(R) and config ranking")
    print(sep)
    print(f"  {'config':<22}  {'total trades':>13}  {'kills':>6}  {'sum R':>9}  "
          f"{'Δ vs baseline':>16}")
    print(f"  {'-' * 80}")

    base_sumR = agg[("baseline", None)]["sumR"]
    base_n    = agg[("baseline", None)]["n"]
    print(f"  {'BASELINE':<22}  {base_n:>13}  {0:>6}  {base_sumR:>+9.2f}  "
          f"{'-':>16}")

    # Sort configs by sumR descending
    configs = [(k, v) for k, v in agg.items() if k != ("baseline", None)]
    configs.sort(key=lambda kv: kv[1]["sumR"], reverse=True)
    for (h, thr), v in configs:
        delta = v["sumR"] - base_sumR
        label = f"kill h={h:>2}, thr={thr:.2f}"
        print(f"  {label:<22}  {v['n']:>13}  {v['kills']:>6}  "
              f"{v['sumR']:>+9.2f}  {delta:>+16.2f}")

    print()


if __name__ == "__main__":
    main()

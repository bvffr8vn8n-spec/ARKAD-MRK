"""
experiments/live_vs_bt_mfe.py
— Compare live paper-trader MFE/MAE distributions to BT-expected ones at
   the 6h / 12h / 18h post-entry checkpoints.

This is the deciding test for the strategy after instrumentation is live:
  Scenario A (theory valid):   live MFE distribution ≈ BT MFE distribution
                                → strategy edge holds in live, paper losses
                                  are sample noise, proceed to walk-forward
  Scenario B (theory invalid): distributions diverge strongly
                                → live execution loses BT edge; diagnose
                                  drift / regime / slippage before any pilot
  Scenario C (partial):        OK on some assets, broken on others
                                → trade only the validated subset, dig into
                                  the rest

Method
------
1. Load `paper_trades_tier1.csv` (post-instrumentation, where mfe_r_h6 etc.
   are populated).  Filter trades with non-null checkpoint values.
2. For each Tier-1 asset, train+score+execute the same way the paper trader
   does, run a scaled-exit backtest, and post-process each BT trade to
   recompute MFE/MAE at the same checkpoints by walking the bar history.
3. Per (asset, checkpoint): two-sample KS test (if scipy available) + bucket
   counts.
4. Per-asset verdict + aggregate verdict.

Usage:
    python experiments/live_vs_bt_mfe.py
    python experiments/live_vs_bt_mfe.py --asset XRPUSDT
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
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

try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
LIVE_CSV = os.path.join(_ROOT, "paper_trades_tier1.csv")

ASSETS = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]
CHECKPOINTS = [6, 12, 18]   # post-entry hours, matching live trade_manager

# Verdict thresholds for KS p-value
P_VALID    = 0.10   # p > P_VALID → distributions statistically similar (A)
P_INVALID  = 0.01   # p < P_INVALID → distributions clearly different (B)
MIN_N      = 5      # minimum live trades per asset to bother comparing

# MFE buckets — must match mfe_drill.py for cross-script consistency
BUCKETS = [
    (0.00, 0.20, "[0.0 – 0.2)"),
    (0.20, 0.40, "[0.2 – 0.4)"),
    (0.40, 0.55, "[0.4 – 0.55)"),
    (0.55, 0.65, "[0.55 – 0.65)"),
    (0.65, 1.00, "[0.65 – 1.0)"),
    (1.00, 1.67, "[1.0 – 1.67)"),
    (1.67, 999.0, "[1.67+ )"),
]


def _bucket_for(v: float) -> str:
    for lo, hi, lbl in BUCKETS:
        if lo <= v < hi:
            return lbl
    return BUCKETS[-1][2]


def _load_live_mfe(asset: str) -> dict[int, list[float]]:
    """Return {6: [..], 12: [..], 18: [..]} for given asset's live trades."""
    if not os.path.exists(LIVE_CSV):
        return {h: [] for h in CHECKPOINTS}
    df = pd.read_csv(LIVE_CSV)
    df = df[df["symbol"] == asset]
    result = {}
    for h in CHECKPOINTS:
        col = f"mfe_r_h{h}"
        if col not in df.columns:
            result[h] = []
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        result[h] = vals
    return result


def _build_signals(asset: str) -> pd.DataFrame | None:
    """Same pipeline as early_kill_sweep — train [START, CUTOFF), score >= CUTOFF."""
    csv_path = os.path.join(DATA_DIR, f"{asset}_1h_4y.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")

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
        return None

    feat_cols = get_feature_columns(train)
    model     = fit_model(train, feat_cols)
    scored    = apply_signals(model, feat_cols, test.copy())
    filt = apply_trend_filter(scored)
    filt["signal"] = filt["signal_trend_filtered"]
    filt = apply_vol_filter(filt)
    filt["signal"] = filt["signal_vol_filtered"]

    five_min_path = os.path.join(DATA_DIR, f"{asset}_5m_4y.csv")
    if os.path.exists(five_min_path):
        df_15m = load_5m_as_15m(five_min_path)
        annotated = annotate_signals_AB(filt, df_15m)
        annotated["signal"] = annotated["signal_15m_A"]
        return annotated
    return filt


def _bt_mfe_at_checkpoints(signals_df: pd.DataFrame,
                             trades: list[dict]) -> dict[int, list[float]]:
    """
    For each BT trade, recompute MFE in R-units at h=6/12/18 bars after entry
    by walking the bar history.  Returns {6: [...], 12: [...], 18: [...]}.
    Trades whose exit happened BEFORE a checkpoint contribute no value to that
    checkpoint — matches live behaviour where mfe_r_h12 stays None on a 4-hour
    trade.
    """
    out: dict[int, list[float]] = {h: [] for h in CHECKPOINTS}
    idx = signals_df.index
    highs = signals_df["high"]
    lows  = signals_df["low"]

    for t in trades:
        try:
            entry_pos = idx.get_loc(t["entry_date"])
            exit_pos  = idx.get_loc(t["exit_date"])
        except KeyError:
            continue
        sl_dist = abs(t["entry_price"] - t["stop_price"])
        if sl_dist <= 0:
            continue
        direction = 1 if t["direction"] == "long" else -1

        for h in CHECKPOINTS:
            ckpt_pos = entry_pos + h
            if ckpt_pos >= exit_pos or ckpt_pos >= len(idx):
                # Trade closed before this checkpoint — no value
                continue
            window_h = highs.iloc[entry_pos + 1 : ckpt_pos + 1]
            window_l = lows.iloc [entry_pos + 1 : ckpt_pos + 1]
            if direction == 1:
                max_fav = float((window_h - t["entry_price"]).max())
            else:
                max_fav = float((t["entry_price"] - window_l).max())
            out[h].append(max_fav / sl_dist)

    return out


def _bucket_counts(values: list[float]) -> dict[str, int]:
    counts = {lbl: 0 for _, _, lbl in BUCKETS}
    for v in values:
        counts[_bucket_for(v)] += 1
    return counts


def _ks_p(a: list[float], b: list[float]) -> float | None:
    if not HAS_SCIPY:
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    return float(ks_2samp(a, b).pvalue)


def _verdict_label(p: float | None, n_live: int) -> str:
    if n_live < MIN_N:
        return f"INSUFFICIENT (live n={n_live})"
    if p is None:
        return "NO_TEST (scipy missing or n<2)"
    if p > P_VALID:
        return f"VALID   (KS p={p:.3f})"
    if p < P_INVALID:
        return f"INVALID (KS p={p:.4f})"
    return f"AMBIG   (KS p={p:.3f})"


def _print_asset_block(asset: str,
                       live: dict[int, list[float]],
                       bt:   dict[int, list[float]]) -> dict[int, str]:
    sep = "-" * 84
    print(f"\n  ── {asset} ────────────────────────────────────────────────────────────────")
    print(f"  Live trades with checkpoint data:  "
          f"h6={len(live[6])}  h12={len(live[12])}  h18={len(live[18])}")
    print(f"  BT trades with checkpoint data:    "
          f"h6={len(bt[6])}  h12={len(bt[12])}  h18={len(bt[18])}")

    per_h_verdict = {}
    for h in CHECKPOINTS:
        lv = live[h]
        bv = bt[h]
        p = _ks_p(lv, bv)
        verdict = _verdict_label(p, len(lv))
        per_h_verdict[h] = verdict

        live_med = np.median(lv) if lv else float("nan")
        bt_med   = np.median(bv) if bv else float("nan")

        print(f"\n    h={h}h  →  {verdict}")
        print(f"      median MFE_R:  live={live_med:.3f}   bt={bt_med:.3f}   "
              f"Δ={live_med - bt_med:+.3f}")
        if not lv or not bv:
            continue
        # Bucket comparison
        lc = _bucket_counts(lv)
        bc = _bucket_counts(bv)
        n_l = max(len(lv), 1)
        n_b = max(len(bv), 1)
        print(f"      {'bucket':<18}  {'live n':>7}  {'live %':>7}  "
              f"{'bt n':>7}  {'bt %':>7}  {'Δ%':>7}")
        for _, _, lbl in BUCKETS:
            pl = lc[lbl] / n_l * 100
            pb = bc[lbl] / n_b * 100
            print(f"      {lbl:<18}  {lc[lbl]:>7}  {pl:>6.1f}%  "
                  f"{bc[lbl]:>7}  {pb:>6.1f}%  {pl - pb:>+6.1f}")
    return per_h_verdict


def _aggregate_verdict(per_asset: dict[str, dict[int, str]]) -> str:
    """Coarse aggregate: how many (asset, h) cells are VALID vs INVALID vs ambiguous."""
    n_valid = 0
    n_invalid = 0
    n_ambig = 0
    n_insuff = 0
    n_total = 0
    for asset, ch in per_asset.items():
        for h, v in ch.items():
            n_total += 1
            if v.startswith("VALID"):    n_valid += 1
            elif v.startswith("INVALID"): n_invalid += 1
            elif v.startswith("AMBIG"):  n_ambig += 1
            else: n_insuff += 1

    testable = n_total - n_insuff
    if testable == 0:
        return f"NO DATA YET — accumulate more live trades (≥{MIN_N} per asset)"

    valid_frac   = n_valid / testable
    invalid_frac = n_invalid / testable

    if valid_frac >= 0.75:
        return (f"A — STRATEGY VALID  ({n_valid}/{testable} cells valid).  "
                f"Proceed to walk-forward + pilot.")
    if invalid_frac >= 0.75:
        return (f"B — STRATEGY INVALID  ({n_invalid}/{testable} cells diverged).  "
                f"Diagnose: drift, regime, slippage, or model staleness.  "
                f"Do NOT deploy real money.")
    return (f"C — PARTIAL  ({n_valid} valid, {n_ambig} ambiguous, {n_invalid} "
            f"invalid out of {testable}).  Trade only the asset/checkpoint "
            f"combos that validated; dig into the rest.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None, help="Run on a single asset")
    args = ap.parse_args()

    assets = [args.asset.upper()] if args.asset else ASSETS

    print("=" * 84)
    print("  ARKAD MRK — Live↔BT MFE Distribution Test")
    print(f"  Live CSV:        {LIVE_CSV}")
    print(f"  Checkpoint hrs:  {CHECKPOINTS}")
    print(f"  KS test:         {'scipy.stats.ks_2samp' if HAS_SCIPY else 'UNAVAILABLE (install scipy)'}")
    print(f"  Verdict gates:   VALID if p>{P_VALID}, INVALID if p<{P_INVALID}")
    print("=" * 84)

    if not os.path.exists(LIVE_CSV):
        print(f"\n  ✗ Live CSV not found: {LIVE_CSV}")
        print(f"    Run the paper trader first; this script needs >= {MIN_N} trades")
        print(f"    per asset with mfe_r_h6/h12/h18 columns populated.")
        return

    per_asset_verdicts: dict[str, dict[int, str]] = {}
    for asset in assets:
        live = _load_live_mfe(asset)
        sig  = _build_signals(asset)
        if sig is None:
            print(f"\n  {asset}: signal build failed (data?); skipping")
            continue
        trades_bt, _ = run_backtest(sig, exit_mode="scaled")
        bt = _bt_mfe_at_checkpoints(sig, trades_bt)
        per_asset_verdicts[asset] = _print_asset_block(asset, live, bt)

    print()
    print("=" * 84)
    print(f"  AGGREGATE VERDICT")
    print("=" * 84)
    print(f"  {_aggregate_verdict(per_asset_verdicts)}")
    print()


if __name__ == "__main__":
    main()

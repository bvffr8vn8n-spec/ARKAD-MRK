"""
experiments/walk_forward_v2.py
— Honest walk-forward validation using the realistic execution engine.

What changed vs the original walk_forward.py
---------------------------------------------
Old engine (biased):
    entry = signal_bar.close           <- same candle that produced the signal
    15m entry = 15m_bar.close          <- intrabar lookahead in B-pullback

This engine (honest):
    entry = NEXT 1H bar open           <- no same-candle entry
    A-filter applied via 15m data      <- only signal FILTERING, no entry price from 15m
    B-pullback NOT used for entry      <- its "edge" was intrabar lookahead

Setup per walk-forward window
------------------------------
  1. Train model on train slice (with purge gap)
  2. Score test slice -> raw signals
  3. Apply trend + vol filters (same as main pipeline)
  4. If 15m data available: apply A-filter (annotate_signals_A)
     -> use signal_15m_A as the final signal column
  5. Run engine_v2(realistic_execution=True, df_15m=None)
     -> entry at open of bar[i+1], not bar[i].close

Two modes run back-to-back for every window:
    BASELINE  : original engine (run_backtest)     <- the known biased reference
    REALISTIC : engine_v2, A-filter, next-open     <- honest new result

Usage
-----
    python experiments/walk_forward_v2.py                  # all Tier-1 assets
    python experiments/walk_forward_v2.py --asset AVAXUSDT # single asset
    python experiments/walk_forward_v2.py --all            # all 10 assets
"""

import argparse
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import pandas as pd

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column,
    apply_trend_filter, apply_vol_filter,
)
from features.execution_15m import load_5m_as_15m, annotate_signals_A
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest            # original biased engine
from backtest.metrics import compute_metrics
from backtest.engine_v2 import run_backtest_v2, compute_metrics_v2

# ── Assets ────────────────────────────────────────────────────────────────────

TIER1_ASSETS = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]

ALL_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "MATICUSDT",
]

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
OUT_DIR  = os.path.join(_ROOT, "experiments")


# ── Walk-forward core ─────────────────────────────────────────────────────────

def _build_windows(n: int) -> list[dict]:
    windows = []
    for train_frac in config.WF_TRAIN_FRACTIONS:
        test_frac = train_frac + config.WF_TEST_FRACTION
        windows.append({
            "train_start": 0,
            "train_end":   int(n * train_frac),
            "test_end":    min(int(n * test_frac), n),
            "train_frac":  train_frac,
            "test_frac":   min(test_frac, 1.0),
        })
    return windows


def run_wf_asset(
    asset: str,
    df_full: pd.DataFrame,
    df_15m: pd.DataFrame | None,
) -> dict:
    """
    Run walk-forward for one asset.
    Returns a dict of per-window metrics for both BASELINE and REALISTIC modes.
    """
    feature_cols = get_feature_columns(df_full)
    windows      = _build_windows(len(df_full))
    has_15m      = df_15m is not None and len(df_15m) > 0

    wf_rows = []

    for idx, w in enumerate(windows, start=1):
        purge_end = w["train_end"] - config.FORWARD_RETURN_WINDOW
        train_df  = df_full.iloc[w["train_start"]:purge_end]
        test_df   = df_full.iloc[w["train_end"]:w["test_end"]]

        date_lo = test_df.index[0].date()  if len(test_df) > 0 else "?"
        date_hi = test_df.index[-1].date() if len(test_df) > 0 else "?"

        print(f"      Win{idx}  train=0-{w['train_frac']*100:.0f}%"
              f"  test={w['train_frac']*100:.0f}-{w['test_frac']*100:.0f}%"
              f"  [{date_lo} -> {date_hi}]"
              f"  ({len(train_df):,}/{len(test_df):,} bars)", end="")

        if len(train_df) < 50 or len(test_df) < 5:
            print("  -- skip")
            continue

        # ── Train ──────────────────────────────────────────────────────────
        model      = fit_model(train_df, feature_cols)
        scored     = apply_signals(model, feature_cols, test_df)

        # ── Apply standard pipeline filters ────────────────────────────────
        filtered = apply_trend_filter(scored)
        filtered["signal"] = filtered["signal_trend_filtered"]
        filtered = apply_vol_filter(filtered)
        filtered["signal"] = filtered["signal_vol_filtered"]

        n_raw = int((filtered["signal"] != 0).sum())

        # ── BASELINE: original engine, same-bar close ───────────────────────
        trades_base, eq_base = run_backtest(filtered)
        m_base = compute_metrics(trades_base, eq_base)
        if "error" in m_base:
            m_base = {"n_trades": 0, "profit_factor": float("nan"),
                      "win_rate": float("nan"), "expectancy": float("nan"),
                      "max_drawdown": float("nan")}

        # ── REALISTIC: A-filter + next 1H open entry ───────────────────────
        # Step 1: apply A-filter if 15m data available
        if has_15m:
            ann = annotate_signals_A(
                filtered, df_15m,
                k_bars=config.__dict__.get("A_FILTER_BARS", 4),
                min_aligned=config.__dict__.get("A_FILTER_MIN_ALIGNED", 2),
            )
            real_sig = filtered.copy()
            real_sig["signal"] = ann["signal_15m_A"]
        else:
            real_sig = filtered.copy()

        n_after_a = int((real_sig["signal"] != 0).sum())

        # Step 2: realistic engine — entry at next 1H bar open
        trades_real, eq_real = run_backtest_v2(
            real_sig,
            df_15m=None,           # no 15m for entry — only next 1H open
            realistic_execution=True,
            symbol=asset,
        )
        m_real = compute_metrics_v2(trades_real, eq_real)
        if "error" in m_real:
            m_real = {"n_trades": 0, "profit_factor": float("nan"),
                      "win_rate": float("nan"), "expectancy": float("nan"),
                      "max_drawdown": float("nan")}

        # ── Logging ────────────────────────────────────────────────────────
        pf_base = m_base.get("profit_factor", float("nan"))
        pf_real = m_real.get("profit_factor", float("nan"))
        delta   = pf_real - pf_base if (math.isfinite(pf_real) and math.isfinite(pf_base)) else float("nan")

        print(f"  raw={n_raw}  A-filtered={n_after_a}"
              f"  base_PF={_f(pf_base)}  real_PF={_f(pf_real)}"
              f"  delta={_f(delta, '+.3f')}")

        wf_rows.append({
            "window":       idx,
            "test_start":   str(date_lo),
            "test_end":     str(date_hi),
            "test_bars":    len(test_df),
            "raw_signals":  n_raw,
            "a_filtered":   n_after_a,
            # Baseline
            "base_trades":  m_base.get("n_trades", 0),
            "base_pf":      m_base.get("profit_factor", float("nan")),
            "base_wr":      m_base.get("win_rate", float("nan")),
            "base_exp":     m_base.get("expectancy", float("nan")),
            "base_dd":      m_base.get("max_drawdown", float("nan")),
            # Realistic
            "real_trades":  m_real.get("n_trades", 0),
            "real_pf":      m_real.get("profit_factor", float("nan")),
            "real_wr":      m_real.get("win_rate", float("nan")),
            "real_exp":     m_real.get("expectancy", float("nan")),
            "real_dd":      m_real.get("max_drawdown", float("nan")),
        })

    return wf_rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_asset_summary(asset: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {asset}: no results")
        return

    sep = "-" * 76
    print(f"\n  {asset} — Walk-Forward Summary")
    print(f"  {sep}")
    print(f"  {'Win':>3}  {'Test period':>22}  "
          f"{'Tr':>4}  {'Base PF':>8}  "
          f"{'Re Tr':>5}  {'Real PF':>8}  {'WR':>6}  {'Exp$':>7}  {'DD':>5}")
    print(f"  {sep}")

    for r in rows:
        period = f"{r['test_start']} -> {r['test_end']}"
        print(f"  {r['window']:>3}  {period:>22}  "
              f"{r['base_trades']:>4}  {_f(r['base_pf']):>8}  "
              f"{r['real_trades']:>5}  {_f(r['real_pf']):>8}  "
              f"{_fp(r['real_wr']):>6}  "
              f"{_f(r['real_exp'], '+.2f'):>7}  "
              f"{_f(r['real_dd'], '.1f'):>5}")

    print(f"  {sep}")

    # Aggregates
    valid = [r for r in rows if math.isfinite(r.get("real_pf", float("nan")))]
    if not valid:
        print("  No valid windows")
        return

    avg_base_pf = sum(r["base_pf"] for r in valid if math.isfinite(r["base_pf"])) / len(valid)
    avg_real_pf = sum(r["real_pf"] for r in valid) / len(valid)
    avg_real_wr = sum(r["real_wr"] for r in valid if math.isfinite(r["real_wr"])) / len(valid)
    avg_real_exp = sum(r["real_exp"] for r in valid if math.isfinite(r["real_exp"])) / len(valid)
    total_base  = sum(r["base_trades"] for r in rows)
    total_real  = sum(r["real_trades"] for r in rows)
    delta_pf    = avg_real_pf - avg_base_pf

    verdict = "EDGE CONFIRMED" if avg_real_pf >= 1.0 else (
              "MARGINAL"       if avg_real_pf >= 0.90 else "NO EDGE")
    marker  = "✓" if avg_real_pf >= 1.0 else ("~" if avg_real_pf >= 0.90 else "✗")

    print(f"\n  {marker}  VERDICT: {verdict}")
    print(f"     Avg Baseline PF : {avg_base_pf:.3f}")
    print(f"     Avg Realistic PF: {avg_real_pf:.3f}  (delta {delta_pf:+.3f} vs baseline)")
    print(f"     Avg WR          : {avg_real_wr*100:.1f}%")
    print(f"     Avg Expectancy  : ${avg_real_exp:.2f}")
    print(f"     Total trades    : {total_real}  (baseline {total_base})")
    print()


def _print_cross_asset_summary(all_results: dict[str, list[dict]]) -> None:
    print(f"\n{'='*76}")
    print(f"  CROSS-ASSET REALISTIC WF SUMMARY")
    print(f"{'='*76}")
    print(f"  {'Asset':<14}  {'Avg Base PF':>12}  {'Avg Real PF':>12}  "
          f"{'Delta':>8}  {'Trades':>7}  {'Verdict':>16}")
    print(f"  {'-'*72}")

    all_rows_flat = []
    for asset, rows in all_results.items():
        if not rows:
            continue
        valid = [r for r in rows if math.isfinite(r.get("real_pf", float("nan")))]
        if not valid:
            continue

        avg_base = sum(r["base_pf"] for r in valid if math.isfinite(r["base_pf"])) / len(valid)
        avg_real = sum(r["real_pf"] for r in valid) / len(valid)
        delta    = avg_real - avg_base
        trades   = sum(r["real_trades"] for r in rows)
        verdict  = "EDGE" if avg_real >= 1.0 else ("MARGINAL" if avg_real >= 0.90 else "NO EDGE")
        marker   = "✓" if avg_real >= 1.0 else ("~" if avg_real >= 0.90 else "✗")

        print(f"  {asset:<14}  {avg_base:>12.3f}  {avg_real:>12.3f}  "
              f"{delta:>+8.3f}  {trades:>7}  {marker} {verdict:>14}")

        all_rows_flat.append({"asset": asset, "base_pf": avg_base,
                               "real_pf": avg_real, "delta": delta, "trades": trades})

    print(f"  {'-'*72}")

    if all_rows_flat:
        avg_b = sum(r["base_pf"] for r in all_rows_flat) / len(all_rows_flat)
        avg_r = sum(r["real_pf"] for r in all_rows_flat) / len(all_rows_flat)
        print(f"  {'AVERAGE':<14}  {avg_b:>12.3f}  {avg_r:>12.3f}  "
              f"{avg_r - avg_b:>+8.3f}")

    print(f"{'='*76}\n")


def _save_results(all_results: dict[str, list[dict]]) -> None:
    rows = []
    for asset, wf_rows in all_results.items():
        for r in wf_rows:
            rows.append({"asset": asset, **r})
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "walk_forward_v2_results.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "N/A"
    return format(v, fmt)

def _fp(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "N/A"
    return f"{v*100:.1f}%"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realistic walk-forward validation — engine_v2"
    )
    parser.add_argument("--asset", default=None,
                        help="Single asset (e.g. AVAXUSDT). Default: Tier-1.")
    parser.add_argument("--all", action="store_true",
                        help="Run all 10 candidate assets.")
    args = parser.parse_args()

    if args.all:
        assets = ALL_ASSETS
    elif args.asset:
        assets = [args.asset.upper()]
    else:
        assets = TIER1_ASSETS

    print(f"\n{'='*76}")
    print(f"  ARKAD MRK — Realistic Walk-Forward (engine_v2)")
    print(f"  Entry model : next 1H bar open + {config.SLIPPAGE_PCT*100:.2f}% slip")
    print(f"  A-filter    : 4 x 15m bars, min_aligned=2  (signal filtering only)")
    print(f"  B-pullback  : DISABLED  (was intrabar lookahead)")
    print(f"  SL/TP       : {config.STOP_LOSS_ATR_MULT}x/{config.TAKE_PROFIT_ATR_MULT}x ATR")
    print(f"  WF windows  : {config.WF_TRAIN_FRACTIONS}  (10% test each)")
    print(f"  Assets      : {', '.join(assets)}")
    print(f"{'='*76}\n")

    all_results: dict[str, list[dict]] = {}

    for asset in assets:
        path_1h = os.path.join(DATA_DIR, f"{asset}_1h_4y.csv")
        path_5m = os.path.join(DATA_DIR, f"{asset}_5m_4y.csv")

        if not os.path.exists(path_1h):
            print(f"  [SKIP] {asset}: missing {path_1h}")
            continue

        print(f"\n  {asset}")
        print(f"  {'-'*60}")

        # Load 1H, run full feature pipeline
        df_raw = load_ohlcv(path_1h)
        df     = generate_features(df_raw)
        df     = add_regime_columns(df)
        df     = add_session_column(df)
        df     = add_labels(df)
        df.dropna(inplace=True)
        print(f"  1H data : {len(df_raw):,} bars  "
              f"({df_raw.index[0].date()} -> {df_raw.index[-1].date()})")

        # Load 15m
        df_15m = None
        if os.path.exists(path_5m):
            try:
                df_15m = load_5m_as_15m(path_5m)
                print(f"  15m data: {len(df_15m):,} bars  "
                      f"({df_15m.index[0].date()} -> {df_15m.index[-1].date()})")
            except Exception as e:
                print(f"  15m load failed: {e}")
        else:
            print(f"  15m data: not found — A-filter will be skipped")

        rows = run_wf_asset(asset, df, df_15m)
        all_results[asset] = rows
        _print_asset_summary(asset, rows)

    _print_cross_asset_summary(all_results)
    _save_results(all_results)


if __name__ == "__main__":
    main()

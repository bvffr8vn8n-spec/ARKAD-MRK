"""
experiments/spot_check_hybrid.py

FAIR spot-check for March 16-21, 2026.

Strategy:
  1. Load full 4-year historical CSV (same data as walk-forward)
  2. Fetch fresh Bybit candles from last CSV date → March 21
  3. Concatenate → one clean dataset
  4. TRAIN on everything BEFORE March 16
  5. TEST  on March 16 → March 21
  6. engine_v2 + realistic_execution=True + A-filter + NO B-pullback

This is equivalent to WF Win3 conditions (full history train, OOS test).

Usage
-----
    python experiments/spot_check_hybrid.py
    python experiments/spot_check_hybrid.py --asset AVAXUSDT
"""

import argparse
import math
import os
import sys
import warnings
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import pandas as pd

import config
from data.loader import load_ohlcv
from data.bybit_loader import fetch_klines
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column,
    apply_trend_filter, apply_vol_filter,
)
from features.execution_15m import load_5m_as_15m, annotate_signals_A
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine_v2 import run_backtest_v2, compute_metrics_v2

TIER1   = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

TRAIN_END  = datetime(2026, 3, 16, 0, 0)   # exclusive train boundary
TEST_START = datetime(2026, 3, 16, 0, 0)
TEST_END   = datetime(2026, 3, 21, 0, 0)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_and_extend_1h(asset: str) -> pd.DataFrame:
    """Load historical CSV + extend to TEST_END via Bybit."""
    path = os.path.join(DATA_DIR, f"{asset}_1h_4y.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    hist = load_ohlcv(path)
    last_ts = hist.index[-1]
    print(f"  CSV 1H  : {len(hist):,} bars  ({hist.index[0].date()} -> {last_ts.date()})")

    # Fetch Bybit from last CSV bar+1h to TEST_END
    fetch_start = last_ts.to_pydatetime() + timedelta(hours=1)
    if fetch_start >= TEST_END:
        print(f"  CSV already covers test window — no fetch needed")
        return hist

    print(f"  Fetching Bybit 1H: {fetch_start.date()} -> {TEST_END.date()} ...")
    live = fetch_klines(
        symbol=asset, category="linear", interval="60",
        start=fetch_start, end=TEST_END,
    )
    print(f"  Bybit 1H: {len(live):,} bars  ({live.index[0].date()} -> {live.index[-1].date()})")

    combined = pd.concat([hist, live])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    print(f"  Combined: {len(combined):,} bars  ({combined.index[0].date()} -> {combined.index[-1].date()})")
    return combined


def _load_and_extend_15m(asset: str) -> pd.DataFrame | None:
    """Load 5m CSV resampled to 15m + extend to TEST_END via Bybit."""
    path_5m = os.path.join(DATA_DIR, f"{asset}_5m_4y.csv")
    if not os.path.exists(path_5m):
        print(f"  15m     : no 5m CSV found — A-filter will be skipped")
        return None

    try:
        hist_15m = load_5m_as_15m(path_5m)
        last_ts  = hist_15m.index[-1]
        print(f"  CSV 15m : {len(hist_15m):,} bars  ({hist_15m.index[0].date()} -> {last_ts.date()})")
    except Exception as e:
        print(f"  15m load failed: {e}")
        return None

    fetch_start = last_ts.to_pydatetime() + timedelta(minutes=15)
    if fetch_start >= TEST_END:
        return hist_15m

    print(f"  Fetching Bybit 15m: {fetch_start.date()} -> {TEST_END.date()} ...")
    try:
        live_15m = fetch_klines(
            symbol=asset, category="linear", interval="15",
            start=fetch_start, end=TEST_END,
        )
        combined = pd.concat([hist_15m, live_15m])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        print(f"  Combined 15m: {len(combined):,} bars")
        return combined
    except Exception as e:
        print(f"  Bybit 15m fetch failed: {e} — using CSV only")
        return hist_15m


# ── Main per-asset logic ───────────────────────────────────────────────────────

def run_asset(asset: str) -> None:
    print(f"\n{'='*68}")
    print(f"  {asset}  |  TRAIN: all data < {TRAIN_END.date()}  |  TEST: {TEST_START.date()} -> {TEST_END.date()}")
    print(f"{'='*68}")

    # ── 1. Load + extend ──────────────────────────────────────────────────────
    df_raw = _load_and_extend_1h(asset)
    df_15m = _load_and_extend_15m(asset)

    # ── 2. Full feature pipeline on combined dataset ──────────────────────────
    df = generate_features(df_raw)
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    train_end_ts  = pd.Timestamp(TRAIN_END)
    test_start_ts = pd.Timestamp(TEST_START)
    test_end_ts   = pd.Timestamp(TEST_END)

    train = df[df.index < train_end_ts]
    test  = df[(df.index >= test_start_ts) & (df.index < test_end_ts)]

    print(f"\n  TRAIN   : {len(train):,} bars  ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"  TEST    : {len(test):,} bars   ({test.index[0].date() if len(test) > 0 else '?'} -> {test.index[-1].date() if len(test) > 0 else '?'})")

    if len(train) < 500:
        print(f"  WARN: only {len(train)} train bars — results may be unreliable")
    if len(test) == 0:
        print("  SKIP: no test bars in range")
        return

    # ── 4. Train model on full pre-March-16 history ───────────────────────────
    print(f"\n  Training model on {len(train):,} bars ...")
    feature_cols = get_feature_columns(df)
    model  = fit_model(train, feature_cols)

    # ── 5. Score test window ──────────────────────────────────────────────────
    scored   = apply_signals(model, feature_cols, test)
    filtered = apply_trend_filter(scored)
    filtered["signal"] = filtered["signal_trend_filtered"]
    filtered = apply_vol_filter(filtered)
    filtered["signal"] = filtered["signal_vol_filtered"]
    n_raw = int((filtered["signal"] != 0).sum())

    # ── 6. A-filter via 15m ───────────────────────────────────────────────────
    has_15m = df_15m is not None and len(df_15m) > 0
    if has_15m:
        ann = annotate_signals_A(
            filtered, df_15m,
            k_bars=getattr(config, "A_FILTER_BARS", 4),
            min_aligned=getattr(config, "A_FILTER_MIN_ALIGNED", 2),
        )
        real_sig = filtered.copy()
        real_sig["signal"] = ann["signal_15m_A"]
    else:
        real_sig = filtered.copy()

    n_after_a = int((real_sig["signal"] != 0).sum())

    print(f"  Raw signals    : {n_raw}")
    print(f"  After A-filter : {n_after_a}  ({'A-filter active' if has_15m else 'NO 15m — A-filter skipped'})")

    # ── 7. Realistic backtest (engine_v2, next 1H open, no B-pullback) ────────
    trades, eq = run_backtest_v2(
        real_sig,
        df_15m=None,               # no B-pullback
        realistic_execution=True,  # entry at next bar open
        symbol=asset,
    )
    m = compute_metrics_v2(trades, eq)

    # ── 8. Print results ──────────────────────────────────────────────────────
    n_trades = m.get("n_trades", 0)
    pf       = m.get("profit_factor", float("nan"))
    wr       = m.get("win_rate", float("nan"))
    exp      = m.get("expectancy", float("nan"))
    avg_r    = m.get("avg_r", float("nan"))
    dd       = m.get("max_drawdown", float("nan"))

    print(f"\n  ── Backtest (realistic engine_v2, A-filter, next 1H open) ──")
    print(f"  Trades       : {n_trades}")
    print(f"  Win rate     : {wr*100:.1f}%" if math.isfinite(wr) else "  Win rate     : N/A")
    print(f"  Profit factor: {pf:.3f}" if math.isfinite(pf) else "  Profit factor: N/A")
    print(f"  Expectancy   : ${exp:+.2f}" if math.isfinite(exp) else "  Expectancy   : N/A")
    print(f"  Avg R        : {avg_r:+.3f}" if math.isfinite(avg_r) else "  Avg R        : N/A")
    print(f"  Max DD       : {dd:.1f}%" if math.isfinite(dd) else "  Max DD       : N/A")

    # Trade log
    if trades:
        print(f"\n  Trade log:")
        print(f"  {'Entry time':<22}  {'Dir':>5}  {'Entry':>9}  {'SL':>9}  "
              f"{'TP':>9}  {'Exit':>9}  {'Reason':>6}  {'R':>6}  {'PnL':>8}")
        print(f"  {'-'*96}")
        for t in trades:
            direction = "LONG" if t.get("direction") == 1 else "SHORT"
            r_val     = t.get("R", t.get("r_multiple", float("nan")))
            r_str     = f"{r_val:+.2f}" if isinstance(r_val, float) and math.isfinite(r_val) else "N/A"
            print(f"  {str(t.get('entry_time','')):<22}  {direction:>5}  "
                  f"{t.get('entry_price',0):>9.4f}  "
                  f"{t.get('stop_price',0):>9.4f}  "
                  f"{t.get('tp_price',0):>9.4f}  "
                  f"{t.get('exit_price',0):>9.4f}  "
                  f"{str(t.get('exit_reason','?')):>6}  "
                  f"{r_str:>6}  "
                  f"${t.get('pnl',0):>+7.2f}")

    # ── 9. Comparison vs WF ───────────────────────────────────────────────────
    WF_PF = {"AVAXUSDT": 1.221, "ADAUSDT": 1.206, "SOLUSDT": 1.041, "XRPUSDT": 1.015}
    wf_pf = WF_PF.get(asset)
    print(f"\n  ── Comparison vs walk-forward ──")
    print(f"  WF avg PF (3 windows, 2024-25) : {wf_pf:.3f}" if wf_pf else "  WF PF: N/A")
    if math.isfinite(pf) and wf_pf:
        delta = pf - wf_pf
        verdict = "CONSISTENT" if abs(delta) < 0.4 else ("BETTER" if delta > 0 else "WORSE")
        print(f"  Spot PF (Mar 16-21)            : {pf:.3f}  (delta {delta:+.3f}  {verdict})")
        if n_trades < 5:
            print(f"  NOTE: only {n_trades} trades — too small sample for strong conclusions")
        if pf >= 1.0:
            print(f"  VERDICT: Edge holds this week")
        elif pf >= 0.85:
            print(f"  VERDICT: Marginal — within normal variance for a 5-day window")
        else:
            print(f"  VERDICT: Underperformed this week — check market conditions")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="Single asset, e.g. AVAXUSDT")
    args = parser.parse_args()

    assets = [args.asset.upper()] if args.asset else TIER1

    print(f"\n{'='*68}")
    print(f"  ARKAD MRK — Hybrid Spot-Check (fair conditions)")
    print(f"  Train: full 4y CSV + Bybit extension < {TRAIN_END.date()}")
    print(f"  Test : {TEST_START.date()} -> {TEST_END.date()}")
    print(f"  Engine: v2 | realistic | A-filter | no B-pullback")
    print(f"{'='*68}")

    for asset in assets:
        try:
            run_asset(asset)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {e}")
        except Exception as e:
            print(f"\n  [ERROR] {asset}: {e}")
            import traceback; traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()

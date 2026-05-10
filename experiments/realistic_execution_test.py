"""
experiments/realistic_execution_test.py

Side-by-side comparison:
    Mode A — Naive (old engine):  entry at signal bar close  (biased baseline)
    Mode B — Realistic 1H:        entry at next 1H bar open  (no 15m data)
    Mode C — Realistic 15m-AB:    entry at next 15m bar open after A+B confirmation

Runs on all Tier-1 assets using the same model and thresholds as the main pipeline.
Answers: Does the strategy retain edge after realistic execution?

Usage
-----
    python experiments/realistic_execution_test.py --asset AVAXUSDT
    python experiments/realistic_execution_test.py          (all Tier-1 assets)
"""

import argparse
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column,
    apply_trend_filter, apply_vol_filter,
)
from features.execution_15m import load_5m_as_15m
from models.classifier import fit_model, get_feature_columns, apply_signals
from backtest.engine_v2 import run_backtest_v2, compute_metrics_v2, trades_to_dataframe

# ── Config ────────────────────────────────────────────────────────────────────

TIER1_ASSETS = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Walk-forward: use last 20% of data as test window
TEST_FRACTION = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_signals(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline -> signals on test set (last TEST_FRACTION of rows)."""
    df = generate_features(df_raw.copy())
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    split   = int(len(df) * (1 - TEST_FRACTION))
    train   = df.iloc[:split]
    test    = df.iloc[split:]

    feature_cols = get_feature_columns(df)
    model        = fit_model(train, feature_cols)

    scored   = apply_signals(model, feature_cols, test)
    filtered = apply_trend_filter(scored)
    filtered["signal"] = filtered["signal_trend_filtered"]
    filtered = apply_vol_filter(filtered)
    filtered["signal"] = filtered["signal_vol_filtered"]

    return filtered


def _fmt(v, fmt=".3f") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "N/A"
    return format(v, fmt)


def _print_comparison(asset: str, results: list[tuple[str, dict]]) -> None:
    col_w = 16
    lbl_w = 22
    sep   = "-" * (lbl_w + col_w * len(results) + 2)

    print(f"\n  {asset}")
    print(f"  {sep}")
    header = f"  {'Metric':<{lbl_w}}" + "".join(f"{name:>{col_w}}" for name, _ in results)
    print(header)
    print(f"  {sep}")

    rows = [
        ("Trades",        "n_trades",      "d"),
        ("Win rate",      "win_rate",      ".1%"),
        ("Profit factor", "profit_factor", ".3f"),
        ("Avg R",         "avg_r",         "+.3f"),
        ("Expectancy $",  "expectancy",    "+.2f"),
        ("Max DD %",      "max_drawdown",  ".1f"),
        ("exit_tp",       "exit_tp",       "d"),
        ("exit_stop",     "exit_stop",     "d"),
        ("exit_time",     "exit_time",     "d"),
    ]

    def _fmtv(v, fmt):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "N/A"
        try:
            return format(v, fmt)
        except (TypeError, ValueError):
            return str(v)

    for label, key, fmt in rows:
        row = f"  {label:<{lbl_w}}"
        for _, m in results:
            row += f"{_fmtv(m.get(key, 'N/A'), fmt):>{col_w}}"
        print(row)

    print(f"  {sep}")

    # PF delta vs naive
    pf_naive = results[0][1].get("profit_factor", math.nan)
    for name, m in results[1:]:
        pf = m.get("profit_factor", math.nan)
        if math.isfinite(pf) and math.isfinite(pf_naive):
            delta = pf - pf_naive
            marker = "✓" if pf >= 1.0 else "✗"
            print(f"  {marker}  {name}  PF={pf:.3f}  delta_vs_naive={delta:+.3f}")
    print()


def run_asset(asset: str) -> None:
    path_1h = os.path.join(DATA_DIR, f"{asset}_1h_4y.csv")
    path_5m = os.path.join(DATA_DIR, f"{asset}_5m_4y.csv")

    if not os.path.exists(path_1h):
        print(f"  [SKIP] {asset}: missing 1H data ({path_1h})")
        return

    print(f"\n{'='*60}")
    print(f"  Running: {asset}")
    print(f"{'='*60}")

    # Load 1H + build signals
    df_raw    = load_ohlcv(path_1h)
    signals   = _build_signals(df_raw)
    n_signals = int((signals["signal"] != 0).sum())
    print(f"  Test window: {signals.index[0].date()} -> {signals.index[-1].date()}"
          f"  ({len(signals):,} bars,  {n_signals} raw signals)")

    # Load 15m data if available
    df_15m = None
    if os.path.exists(path_5m):
        try:
            df_15m = load_5m_as_15m(path_5m)
            print(f"  15m data:  {len(df_15m):,} bars  "
                  f"({df_15m.index[0].date()} -> {df_15m.index[-1].date()})")
        except Exception as exc:
            print(f"  15m load failed: {exc}")

    results = []

    # ── Mode A: Naive (same-bar close) ────────────────────────────────────────
    trades_naive, eq_naive = run_backtest_v2(
        signals, df_15m=None,
        realistic_execution=False,
        symbol=asset,
    )
    m_naive = compute_metrics_v2(trades_naive, eq_naive)
    results.append(("Naive (close)", m_naive))
    print(f"  Naive:    {len(trades_naive)} trades")

    # ── Mode B: Realistic 1H (next-bar open, no 15m) ─────────────────────────
    trades_r1h, eq_r1h = run_backtest_v2(
        signals, df_15m=None,
        realistic_execution=True,
        symbol=asset,
    )
    m_r1h = compute_metrics_v2(trades_r1h, eq_r1h)
    results.append(("Realistic 1H", m_r1h))
    print(f"  Real-1H:  {len(trades_r1h)} trades")

    # ── Mode C: Realistic 15m-AB (next 15m bar open after A+B) ───────────────
    if df_15m is not None:
        trades_rab, eq_rab = run_backtest_v2(
            signals, df_15m=df_15m,
            realistic_execution=True,
            symbol=asset,
        )
        m_rab = compute_metrics_v2(trades_rab, eq_rab)
        results.append(("Realistic 15m-AB", m_rab))
        print(f"  Real-AB:  {len(trades_rab)} trades  "
              f"(A-pass={m_rab.get('a_pass', '?')}  "
              f"pullback={m_rab.get('b_pullback', '?')})")

    _print_comparison(asset, results)

    # Save trade logs
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "realistic_exec_logs"
    )
    os.makedirs(out_dir, exist_ok=True)

    trades_to_dataframe(trades_naive).to_csv(
        os.path.join(out_dir, f"{asset}_naive.csv"), index=False
    )
    trades_to_dataframe(trades_r1h).to_csv(
        os.path.join(out_dir, f"{asset}_realistic_1h.csv"), index=False
    )
    if df_15m is not None and trades_rab:
        trades_to_dataframe(trades_rab).to_csv(
            os.path.join(out_dir, f"{asset}_realistic_ab.csv"), index=False
        )

    print(f"  Trade logs saved to: {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realistic execution comparison for ARKAD MRK strategy"
    )
    parser.add_argument(
        "--asset", default=None,
        help="Single asset to test (e.g. AVAXUSDT). Default: all Tier-1 assets."
    )
    args = parser.parse_args()

    assets = [args.asset.upper()] if args.asset else TIER1_ASSETS

    print(f"\n{'='*60}")
    print(f"  ARKAD MRK — Realistic Execution Test")
    print(f"  Modes: Naive | Realistic-1H | Realistic-15m-AB")
    print(f"  Slippage: {config.SLIPPAGE_PCT*100:.2f}% per side")
    print(f"  Commission: {config.COMMISSION_PCT*100:.2f}% per side")
    print(f"  ATR mult — SL: {config.STOP_LOSS_ATR_MULT}x  TP: {config.TAKE_PROFIT_ATR_MULT}x")
    print(f"{'='*60}")

    for asset in assets:
        run_asset(asset)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
experiments/spot_check_mar2026.py

Fetches live 1H + 15m data from Bybit for 2026-03-19 to 2026-03-21,
trains the model on warmup history, and runs engine_v2 backtest on that window.

Usage
-----
    python experiments/spot_check_mar2026.py
    python experiments/spot_check_mar2026.py --asset AVAXUSDT
"""

import argparse
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
from data.bybit_loader import fetch_klines
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import (
    add_regime_columns, add_session_column,
    apply_trend_filter, apply_vol_filter,
)
from features.execution_15m import load_5m_as_15m, annotate_signals_A
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine_v2 import run_backtest_v2, compute_metrics_v2, trades_to_dataframe

TIER1 = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]

# Test window
TEST_START = datetime(2026, 3, 16, 0, 0)
TEST_END   = datetime(2026, 3, 21, 0, 0)

# Warmup: 350 x 1H bars before test start = ~14.5 days
WARMUP_BARS = 350
WARMUP_START = TEST_START - timedelta(hours=WARMUP_BARS)

# 15m warmup covers same period at finer resolution
WARMUP_15M_START = TEST_START - timedelta(hours=WARMUP_BARS)


def fetch_1h(symbol: str) -> pd.DataFrame:
    print(f"  Fetching 1H data ({WARMUP_START.date()} -> {TEST_END.date()}) ...")
    df = fetch_klines(
        symbol=symbol, category="linear", interval="60",
        start=WARMUP_START, end=TEST_END,
    )
    print(f"  Got {len(df)} bars  ({df.index[0]} -> {df.index[-1]})")
    return df


def fetch_15m(symbol: str) -> pd.DataFrame | None:
    print(f"  Fetching 15m data ...")
    try:
        df = fetch_klines(
            symbol=symbol, category="linear", interval="15",
            start=WARMUP_15M_START, end=TEST_END,
        )
        print(f"  Got {len(df)} 15m bars")
        return df
    except Exception as e:
        print(f"  15m fetch failed: {e}")
        return None


def run_asset(asset: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {asset}  |  {TEST_START.date()} -> {TEST_END.date()}")
    print(f"{'='*64}")

    # ── Fetch data ────────────────────────────────────────────────
    df_raw = fetch_1h(asset)
    df_15m = fetch_15m(asset)

    # ── Build features on full window ─────────────────────────────
    df = generate_features(df_raw)
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    # ── Split: train = warmup, test = Mar 19-21 ───────────────────
    test_start_ts = pd.Timestamp(TEST_START)
    test_end_ts   = pd.Timestamp(TEST_END)

    train = df[df.index < test_start_ts]
    test  = df[(df.index >= test_start_ts) & (df.index < test_end_ts)]

    print(f"\n  Train bars : {len(train):,}  ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"  Test  bars : {len(test):,}   ({test.index[0].date()} -> {test.index[-1].date()})")

    if len(train) < 50:
        print("  SKIP: not enough train bars")
        return
    if len(test) == 0:
        print("  SKIP: no test bars in range (data may not cover Mar 19-21 yet)")
        return

    # ── Train model ───────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    model   = fit_model(train, feature_cols)
    scored  = apply_signals(model, feature_cols, test)

    # ── Apply filters ─────────────────────────────────────────────
    filtered = apply_trend_filter(scored)
    filtered["signal"] = filtered["signal_trend_filtered"]
    filtered = apply_vol_filter(filtered)
    filtered["signal"] = filtered["signal_vol_filtered"]

    n_raw = int((filtered["signal"] != 0).sum())

    # ── Apply A-filter if 15m available ───────────────────────────
    if df_15m is not None and len(df_15m) > 0:
        ann = annotate_signals_A(
            filtered, df_15m,
            k_bars=getattr(config, "A_FILTER_BARS", 4),
            min_aligned=getattr(config, "A_FILTER_MIN_ALIGNED", 2),
        )
        real_sig = filtered.copy()
        real_sig["signal"] = ann["signal_15m_A"]
        n_after_a = int((real_sig["signal"] != 0).sum())
    else:
        real_sig  = filtered.copy()
        n_after_a = n_raw

    print(f"\n  Raw signals     : {n_raw}")
    print(f"  After A-filter  : {n_after_a}")

    # ── Print each signal bar ─────────────────────────────────────
    sig_bars = real_sig[real_sig["signal"] != 0]
    if len(sig_bars) == 0:
        print("\n  No signals in test window.")
    else:
        print(f"\n  Signal bars:")
        print(f"  {'Time':<22}  {'Dir':>5}  {'Close':>10}  {'ATR':>8}  {'BuyP':>6}  {'SellP':>6}")
        print(f"  {'-'*64}")
        for ts, row in sig_bars.iterrows():
            direction = "LONG" if row["signal"] == 1 else "SHORT"
            atr_val = row.get('atr', float('nan'))
            atr_str = f"{atr_val:>8.4f}" if isinstance(atr_val, float) else f"{'?':>8}"
            print(f"  {str(ts):<22}  {direction:>5}  "
                  f"{row['close']:>10.4f}  {atr_str}  "
                  f"{row.get('buy_prob', float('nan')):>6.3f}  "
                  f"{row.get('sell_prob', float('nan')):>6.3f}")

    # ── Run realistic backtest ─────────────────────────────────────
    trades, eq = run_backtest_v2(
        real_sig,
        df_15m=None,
        realistic_execution=True,
        symbol=asset,
    )
    m = compute_metrics_v2(trades, eq)

    print(f"\n  Backtest result (realistic, next 1H open):")
    print(f"  Trades      : {m.get('n_trades', 0)}")
    if trades:
        print(f"  Win rate    : {m.get('win_rate', float('nan'))*100:.1f}%")
        print(f"  Profit factor: {m.get('profit_factor', float('nan')):.3f}")
        print(f"  Expectancy  : ${m.get('expectancy', float('nan')):+.2f}")
        print(f"  Max DD      : {m.get('max_drawdown', float('nan')):.1f}%")

        print(f"\n  Trade log:")
        print(f"  {'Entry time':<22}  {'Dir':>5}  {'Entry':>10}  "
              f"{'SL':>10}  {'TP':>10}  {'Exit':>10}  {'Reason':>8}  {'PnL':>8}")
        print(f"  {'-'*92}")
        for t in trades:
            print(f"  {str(t.get('entry_time','')):<22}  "
                  f"{'LONG' if t.get('direction')==1 else 'SHORT':>5}  "
                  f"{t.get('entry_price', 0):>10.4f}  "
                  f"{t.get('stop_price', 0):>10.4f}  "
                  f"{t.get('tp_price', 0):>10.4f}  "
                  f"{t.get('exit_price', 0):>10.4f}  "
                  f"{str(t.get('exit_reason','?')):>8}  "
                  f"${t.get('pnl', 0):>+7.2f}")
    else:
        print("  No trades executed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None)
    args = parser.parse_args()

    assets = [args.asset.upper()] if args.asset else TIER1

    print(f"\nSpot-check: {TEST_START} -> {TEST_END} (UTC)")
    print(f"Assets: {', '.join(assets)}")

    for asset in assets:
        run_asset(asset)

    print("\nDone.")


if __name__ == "__main__":
    main()

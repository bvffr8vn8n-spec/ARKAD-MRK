"""
experiments/trade_mfe_analysis.py

For each trade from the hybrid spot-check:
  - MFE (max favorable excursion): how far price moved IN OUR FAVOR before exit
  - MAE (max adverse excursion):   how far price moved AGAINST us before exit
  - Direction check: was the model's predicted direction correct?

Answers the question: did the price ever "visit" the right zone before reversing?
"""

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
from backtest.engine_v2 import run_backtest_v2

TIER1    = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

TRAIN_END  = datetime(2026, 2, 1, 0, 0)
TEST_START = datetime(2026, 2, 1, 0, 0)
TEST_END   = datetime(2026, 2, 28, 23, 0)


def _load_extend_1h(asset):
    path = os.path.join(DATA_DIR, f"{asset}_1h_4y.csv")
    hist = load_ohlcv(path)
    last = hist.index[-1].to_pydatetime()
    fetch_start = last + timedelta(hours=1)
    if fetch_start < TEST_END:
        live = fetch_klines(symbol=asset, category="linear", interval="60",
                            start=fetch_start, end=TEST_END)
        combined = pd.concat([hist, live])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined
    return hist


def _load_extend_15m(asset):
    path = os.path.join(DATA_DIR, f"{asset}_5m_4y.csv")
    if not os.path.exists(path):
        return None
    try:
        hist = load_5m_as_15m(path)
        last = hist.index[-1].to_pydatetime()
        fetch_start = last + timedelta(minutes=15)
        if fetch_start < TEST_END:
            live = fetch_klines(symbol=asset, category="linear", interval="15",
                                start=fetch_start, end=TEST_END)
            combined = pd.concat([hist, live])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            return combined
        return hist
    except Exception:
        return None


def _mfe_mae(trade: dict, df_1h: pd.DataFrame) -> tuple[float, float, list]:
    """
    Calculate MFE and MAE for a trade from candle data.

    MFE: max move in profitable direction (as % of entry)
    MAE: max move against the trade (as % of entry)

    Returns (mfe_pct, mae_pct, bar_closes_during_trade)
    """
    direction  = 1 if trade["direction"] == "long" else -1
    entry_px   = trade["entry_price"]
    entry_ts   = pd.Timestamp(trade["entry_time"])
    exit_ts    = pd.Timestamp(trade["exit_time"])

    # Candles strictly between entry and exit (inclusive of exit bar)
    mask = (df_1h.index > entry_ts) & (df_1h.index <= exit_ts)
    bars = df_1h[mask]

    if len(bars) == 0:
        return 0.0, 0.0, []

    mfe = 0.0
    mae = 0.0
    closes = []

    for _, bar in bars.iterrows():
        # Favorable: high (long) or low (short) extreme
        if direction == 1:
            fav_extreme = (bar["high"] - entry_px) / entry_px * 100
            adv_extreme = (entry_px - bar["low"])  / entry_px * 100
        else:
            fav_extreme = (entry_px - bar["low"])  / entry_px * 100
            adv_extreme = (bar["high"] - entry_px) / entry_px * 100

        mfe = max(mfe, fav_extreme)
        mae = max(mae, adv_extreme)
        closes.append(bar["close"])

    return mfe, mae, closes


def run_asset(asset: str) -> None:
    print(f"\n{'='*68}")
    print(f"  {asset}")
    print(f"{'='*68}")

    df_raw = _load_extend_1h(asset)
    df_15m = _load_extend_15m(asset)

    df = generate_features(df_raw)
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    train_end_ts  = pd.Timestamp(TRAIN_END)
    test_start_ts = pd.Timestamp(TEST_START)
    test_end_ts   = pd.Timestamp(TEST_END)

    train = df[df.index < train_end_ts]
    test  = df[(df.index >= test_start_ts) & (df.index < test_end_ts)]

    if len(test) == 0:
        print("  No test bars.")
        return

    feature_cols = get_feature_columns(df)
    model  = fit_model(train, feature_cols)
    scored = apply_signals(model, feature_cols, test)

    filtered = apply_trend_filter(scored)
    filtered["signal"] = filtered["signal_trend_filtered"]
    filtered = apply_vol_filter(filtered)
    filtered["signal"] = filtered["signal_vol_filtered"]

    has_15m = df_15m is not None and len(df_15m) > 0
    if has_15m:
        ann = annotate_signals_A(filtered, df_15m,
                                  k_bars=getattr(config, "A_FILTER_BARS", 4),
                                  min_aligned=getattr(config, "A_FILTER_MIN_ALIGNED", 2))
        real_sig = filtered.copy()
        real_sig["signal"] = ann["signal_15m_A"]
    else:
        real_sig = filtered.copy()

    trades, _ = run_backtest_v2(real_sig, df_15m=None,
                                 realistic_execution=True, symbol=asset)

    if not trades:
        print("  No trades this period.")
        return

    # ── Direction analysis ────────────────────────────────────────────────────
    print(f"\n  DIRECTION ANALYSIS")
    print(f"  {'#':<3}  {'Entry time':<22}  {'Model dir':>10}  "
          f"{'Entry':>8}  {'Exit':>8}  {'Move vs dir':>12}  {'Correct?':>9}")
    print(f"  {'-'*76}")

    correct_dir = 0
    for i, t in enumerate(trades, 1):
        direction  = t["direction"]        # "long" or "short"
        entry_px   = t["entry_price"]
        exit_px    = t["exit_price"]
        exit_reason = t["exit_reason"]

        # Net price move in trade direction (positive = favorable)
        if direction == "long":
            net_move = (exit_px - entry_px) / entry_px * 100
        else:
            net_move = (entry_px - exit_px) / entry_px * 100

        # Was the overall exit in the right direction?
        direction_correct = net_move > 0 or exit_reason == "tp"
        if direction_correct:
            correct_dir += 1
        marker = "YES" if direction_correct else "NO"

        print(f"  {i:<3}  {str(t['entry_time']):<22}  {direction.upper():>10}  "
              f"{entry_px:>8.4f}  {exit_px:>8.4f}  "
              f"{net_move:>+11.2f}%  {marker:>9}")

    print(f"\n  Direction correct at exit: {correct_dir}/{len(trades)}")

    # ── MFE / MAE analysis ────────────────────────────────────────────────────
    print(f"\n  MFE / MAE ANALYSIS (bar-by-bar from entry to exit)")
    print(f"  MFE = max price moved IN FAVOR  (were we ever winning?)")
    print(f"  MAE = max price moved AGAINST   (how bad did it get?)")
    print(f"\n  {'#':<3}  {'Entry time':<22}  {'Dir':>6}  {'MFE%':>7}  "
          f"{'MAE%':>7}  {'MFE/SL dist':>12}  {'Was up before stop?':>20}")
    print(f"  {'-'*84}")

    for i, t in enumerate(trades, 1):
        direction = t["direction"]
        entry_px  = t["entry_price"]
        stop_px   = t["stop_price"]
        sl_dist   = abs(entry_px - stop_px) / entry_px * 100

        mfe, mae, closes = _mfe_mae(t, df_raw)

        # Did price reach at least 50% toward TP before reversing?
        tp_px   = t["tp_price"]
        tp_dist = abs(tp_px - entry_px) / entry_px * 100
        halfway = mfe >= (tp_dist * 0.5)

        mfe_vs_sl = mfe / sl_dist if sl_dist > 0 else 0.0

        was_winning = "YES" if mfe > 0.05 else "NO (never moved)"

        print(f"  {i:<3}  {str(t['entry_time']):<22}  {direction.upper():>6}  "
              f"{mfe:>6.2f}%  {mae:>6.2f}%  "
              f"{mfe_vs_sl:>11.2f}x  {was_winning:>20}")

        # Show bar-by-bar close evolution
        if closes:
            if direction == "long":
                close_str = "  Closes: " + "  ".join(
                    f"{c:.4f}{'(+)' if c > entry_px else '(-)'}" for c in closes)
            else:
                close_str = "  Closes: " + "  ".join(
                    f"{c:.4f}{'(+)' if c < entry_px else '(-)'}" for c in closes)
            print(f"        {close_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  SUMMARY FOR {asset}")
    all_mfe = []
    all_mae = []
    for t in trades:
        mfe, mae, _ = _mfe_mae(t, df_raw)
        all_mfe.append(mfe)
        all_mae.append(mae)

    if all_mfe:
        print(f"  Avg MFE : {sum(all_mfe)/len(all_mfe):.2f}%")
        print(f"  Avg MAE : {sum(all_mae)/len(all_mae):.2f}%")
        print(f"  MFE > 0 : {sum(1 for x in all_mfe if x > 0.05)}/{len(all_mfe)} trades "
              f"(price moved in our favor at some point)")
        print(f"  Direction correct: {correct_dir}/{len(trades)}")


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--asset", default=None)
    args = parser.parse_args()
    assets = [args.asset.upper()] if args.asset else TIER1

    for asset in assets:
        try:
            run_asset(asset)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {e}")
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {asset}: {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()

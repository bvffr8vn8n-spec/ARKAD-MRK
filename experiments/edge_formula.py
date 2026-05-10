"""
experiments/edge_formula.py

Answers two questions:
  1. How often is the model correct in direction? (MFE > 0)
  2. What average MFE does the model reach before hitting SL on losing trades?
     -> Use this to derive the minimum TP% needed to be profitable.

Runs all 4 Tier-1 assets across a configurable date range.
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
TEST_END   = datetime(2026, 3, 22, 15, 0)


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


def _mfe_mae(trade, df_1h):
    direction  = 1 if trade["direction"] == "long" else -1
    entry_px   = trade["entry_price"]
    entry_ts   = pd.Timestamp(trade["entry_time"])
    exit_ts    = pd.Timestamp(trade["exit_time"])

    mask = (df_1h.index > entry_ts) & (df_1h.index <= exit_ts)
    bars = df_1h[mask]

    mfe, mae = 0.0, 0.0
    for _, bar in bars.iterrows():
        if direction == 1:
            mfe = max(mfe, (bar["high"]  - entry_px) / entry_px * 100)
            mae = max(mae, (entry_px - bar["low"])   / entry_px * 100)
        else:
            mfe = max(mfe, (entry_px - bar["low"])   / entry_px * 100)
            mae = max(mae, (bar["high"] - entry_px)  / entry_px * 100)
    return mfe, mae


def collect_asset(asset):
    print(f"  {asset}...", end=" ", flush=True)
    try:
        df_raw = _load_extend_1h(asset)
        df_15m = _load_extend_15m(asset)

        df = generate_features(df_raw)
        df = add_regime_columns(df)
        df = add_session_column(df)
        df = add_labels(df)
        df.dropna(inplace=True)

        train = df[df.index < pd.Timestamp(TRAIN_END)]
        test  = df[(df.index >= pd.Timestamp(TEST_START)) &
                   (df.index <  pd.Timestamp(TEST_END))]

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

        records = []
        for t in trades:
            mfe, mae = _mfe_mae(t, df_raw)
            records.append({
                "asset":      asset,
                "entry_time": t["entry_time"],
                "direction":  t["direction"],
                "entry_px":   t["entry_price"],
                "exit_px":    t["exit_price"],
                "exit_reason":t["exit_reason"],
                "pnl":        t["pnl"],
                "R":          t["R"],
                "mfe":        mfe,
                "mae":        mae,
                "sl_dist":    abs(t["entry_price"] - t["stop_price"]) / t["entry_price"] * 100,
                "tp_dist":    abs(t["entry_price"] - t["tp_price"])   / t["entry_price"] * 100,
            })

        print(f"{len(records)} trades")
        return records
    except Exception as e:
        print(f"ERROR: {e}")
        return []


def analyze(all_trades):
    n = len(all_trades)
    if n == 0:
        print("No trades.")
        return

    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    tps    = [t for t in all_trades if t["exit_reason"] == "tp"]
    stops  = [t for t in all_trades if t["exit_reason"] == "stop"]
    times  = [t for t in all_trades if t["exit_reason"] == "time"]

    mfe_all    = [t["mfe"] for t in all_trades]
    mfe_wins   = [t["mfe"] for t in wins]
    mfe_losses = [t["mfe"] for t in losses]
    mfe_stops  = [t["mfe"] for t in stops]

    mfe_gt0    = [t for t in all_trades if t["mfe"] > 0.05]
    mfe_gt05   = [t for t in all_trades if t["mfe"] > 0.5]
    mfe_gt1    = [t for t in all_trades if t["mfe"] > 1.0]

    avg_sl_dist = sum(t["sl_dist"] for t in all_trades) / n
    avg_tp_dist = sum(t["tp_dist"] for t in all_trades) / n

    print(f"\n{'='*64}")
    print(f"  EDGE FORMULA ANALYSIS  |  {TEST_START.date()} -> {TEST_END.date()}")
    print(f"  Assets: {', '.join(TIER1)}")
    print(f"{'='*64}")

    print(f"\n  ── 1. ОБЩАЯ СТАТИСТИКА ──────────────────────────────────")
    print(f"  Всего сделок       : {n}")
    print(f"  TP (победы)        : {len(tps)}  ({len(tps)/n*100:.1f}%)")
    print(f"  Stop (поражения)   : {len(stops)}  ({len(stops)/n*100:.1f}%)")
    print(f"  Time (нейтральные) : {len(times)}  ({len(times)/n*100:.1f}%)")
    print(f"  Avg SL дистанция   : {avg_sl_dist:.2f}%  (от цены входа)")
    print(f"  Avg TP дистанция   : {avg_tp_dist:.2f}%  (от цены входа)")

    print(f"\n  ── 2. НАПРАВЛЕНИЕ — БЫЛА ЛИ ПРОГРАММА ПРАВА? ───────────")
    print(f"  Цена двигалась в нашу сторону (MFE > 0.05%)")
    print(f"  Всего            : {len(mfe_gt0)}/{n}  ({len(mfe_gt0)/n*100:.1f}%)")
    print(f"  Из победных TP   : {sum(1 for t in tps if t['mfe']>0.05)}/{len(tps)}")
    print(f"  Из стопов        : {sum(1 for t in stops if t['mfe']>0.05)}/{len(stops)}  "
          f"({sum(1 for t in stops if t['mfe']>0.05)/len(stops)*100:.1f}% стопов имели движение в нашу сторону)")

    print(f"\n  MFE > 0.5%  : {len(mfe_gt05)}/{n}  ({len(mfe_gt05)/n*100:.1f}%)")
    print(f"  MFE > 1.0%  : {len(mfe_gt1)}/{n}  ({len(mfe_gt1)/n*100:.1f}%)")

    print(f"\n  ── 3. MFE НА ПРОИГРАННЫХ СТОПАХ ────────────────────────")
    if stops:
        avg_mfe_stop = sum(t["mfe"] for t in stops) / len(stops)
        med_mfe_stop = sorted(t["mfe"] for t in stops)[len(stops)//2]
        pct25 = sorted(t["mfe"] for t in stops)[len(stops)//4]
        pct75 = sorted(t["mfe"] for t in stops)[len(stops)*3//4]
        print(f"  Количество стопов  : {len(stops)}")
        print(f"  Avg MFE перед стопом : {avg_mfe_stop:.2f}%")
        print(f"  Median MFE           : {med_mfe_stop:.2f}%")
        print(f"  25-й перцентиль MFE  : {pct25:.2f}%")
        print(f"  75-й перцентиль MFE  : {pct75:.2f}%")

        # Buckets
        buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)]
        print(f"\n  Распределение MFE на стопах:")
        for lo, hi in buckets:
            cnt = sum(1 for t in stops if lo <= t["mfe"] < hi)
            bar = "#" * cnt
            label = f"{lo:.1f}-{hi:.1f}%" if hi < 99 else f"{lo:.1f}%+"
            print(f"    {label:>10}  {cnt:>4} сделок  {bar}")

    print(f"\n  ── 4. ФОРМУЛА МИНИМАЛЬНОГО TP ───────────────────────────")
    wr   = len(tps) / n
    lr   = 1 - wr
    sl   = avg_sl_dist
    tp   = avg_tp_dist

    print(f"\n  Текущие параметры:")
    print(f"    Win rate       : {wr*100:.1f}%")
    print(f"    Avg SL dist    : {sl:.2f}%")
    print(f"    Avg TP dist    : {tp:.2f}%")
    print(f"    R:R ratio      : {tp/sl:.2f}")

    # Current expectancy
    exp_curr = wr * tp - lr * sl
    print(f"\n  Текущий expectancy : {exp_curr:+.3f}%  "
          f"({'ПРИБЫЛЬНО' if exp_curr > 0 else 'УБЫТОЧНО'})")

    # Min TP to break even: wr * TP = lr * SL  -> TP = lr * SL / wr
    tp_breakeven = lr * sl / wr if wr > 0 else float("inf")
    print(f"\n  Минимальный TP для break-even при WR={wr*100:.1f}%:")
    print(f"    TP_min = (1 - WR) × SL / WR")
    print(f"    TP_min = {lr:.3f} × {sl:.2f}% / {wr:.3f} = {tp_breakeven:.2f}%")
    print(f"    Текущий TP = {tp:.2f}%  -> "
          f"{'OK (выше min)' if tp > tp_breakeven else 'ПРОБЛЕМА (ниже min)'}")

    # What TP is reachable? Based on avg MFE on ALL trades
    avg_mfe_all = sum(mfe_all) / n
    avg_mfe_stops_val = sum(t["mfe"] for t in stops) / len(stops) if stops else 0
    print(f"\n  Средний MFE по всем сделкам : {avg_mfe_all:.2f}%")
    print(f"  Средний MFE на стопах       : {avg_mfe_stops_val:.2f}%")
    print(f"\n  Если выставить TP = avg_MFE_all = {avg_mfe_all:.2f}%:")
    exp_new = wr * avg_mfe_all - lr * sl
    # Recalculate WR assuming TP at avg_mfe (rough estimate)
    reachable_pct = len([t for t in all_trades if t["mfe"] >= avg_mfe_all]) / n
    print(f"    Сделок достигших этого уровня : {reachable_pct*100:.1f}%")
    print(f"    Expectancy при новом TP       : {exp_new:+.3f}%")

    # Optimal TP sweep
    print(f"\n  ── 5. SWEEP: TP уровни vs прибыльность ─────────────────")
    print(f"  {'TP%':>6}  {'Достигают%':>12}  {'Expectancy%':>13}  {'Verdict':>10}")
    print(f"  {'-'*46}")
    for tp_try in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        reach_n  = sum(1 for t in all_trades if t["mfe"] >= tp_try)
        reach_pct = reach_n / n
        exp_try  = reach_pct * tp_try - (1 - reach_pct) * sl
        verdict  = "PROFIT" if exp_try > 0 else "loss"
        marker   = "***" if exp_try > 0 and tp_try <= avg_mfe_all else ""
        print(f"  {tp_try:>5.1f}%  {reach_pct*100:>11.1f}%  {exp_try:>+12.3f}%  "
              f"{verdict:>10}  {marker}")

    print(f"\n  NOTE: 'Достигают%' = % сделок где MFE >= TP (т.е. TP был бы заполнен)")
    print(f"  SL дистанция фиксирована на avg={sl:.2f}%\n")


def main():
    print(f"\nCollecting trades: {TEST_START.date()} -> {TEST_END.date()}")
    all_trades = []
    for asset in TIER1:
        records = collect_asset(asset)
        all_trades.extend(records)

    analyze(all_trades)
    print("Done.")


if __name__ == "__main__":
    main()

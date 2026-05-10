"""
experiments/engine_scaled_validation.py
Sanity-check: нативный engine с exit_mode="scaled" должен дать те же
результаты что и пост-процессорная симуляция в tp1_sweep_2025.py.

Прогон 2025 года, baseline TP1=0.65R.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from features.generator import generate_features, add_labels
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from features.execution_15m import load_5m_as_15m, annotate_signals_AB
from backtest.engine import run_backtest

TEST_START = pd.Timestamp("2025-01-01")
TEST_END   = pd.Timestamp("2025-12-31 23:59:59")

ASSETS = [
    {"symbol": "AVAXUSDT", "h1": "data/AVAXUSDT_1h_4y.csv", "m5": "data/AVAXUSDT_5m_4y.csv"},
    {"symbol": "ADAUSDT",  "h1": "data/ADAUSDT_1h_4y.csv",  "m5": "data/ADAUSDT_5m_4y.csv"},
    {"symbol": "SOLUSDT",  "h1": "data/SOLUSDT_1h_4y.csv",  "m5": "data/SOLUSDT_5m_4y.csv"},
    {"symbol": "XRPUSDT",  "h1": "data/XRPUSDT_1h_4y.csv",  "m5": "data/XRPUSDT_5m_4y.csv"},
]


def _load_1h(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _run_asset(cfg, exit_mode):
    df = _load_1h(cfg["h1"])
    f = generate_features(df.copy())
    f = add_labels(f)
    f = add_regime_columns(f)
    f = add_session_column(f)
    f = f.dropna()

    train = f[f.index < TEST_START]
    test  = f[(f.index >= TEST_START) & (f.index <= TEST_END)]
    if len(train) < 200 or len(test) == 0:
        return [], None

    cols = get_feature_columns(train)
    model = fit_model(train, cols)
    sig = apply_signals(model, cols, test.copy())

    df15 = load_5m_as_15m(cfg["m5"])
    ann  = annotate_signals_AB(sig, df15)
    ann["signal"] = ann["signal_15m_A"]

    trades, equity = run_backtest(ann, exit_mode=exit_mode)
    for t in trades:
        t["symbol"] = cfg["symbol"]
    return trades, equity


def _stats(trades):
    rs = np.array([t.get("R", 0) for t in trades])
    N = len(rs)
    if N == 0:
        return {}
    wins = rs > 0
    losses = rs < 0
    pf = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")
    return {
        "N": N,
        "WR": wins.sum() / N * 100,
        "Avg R": rs.mean(),
        "Sum R": rs.sum(),
        "PF":    pf,
        "Median R": np.median(rs),
        "Max R": rs.max(),
        "Min R": rs.min(),
    }


def main():
    W = 86
    print("═" * W)
    print("  ARKAD MRK — NATIVE ENGINE VALIDATION (single vs scaled)")
    print("  Test: 2025-01-01 → 2025-12-31  |  4 актива")
    print("═" * W)

    for mode in ["single", "scaled"]:
        all_trades = []
        for cfg in ASSETS:
            trs, _ = _run_asset(cfg, mode)
            all_trades.extend(trs)

        s = _stats(all_trades)
        print(f"\n[exit_mode = '{mode}']")
        for k, v in s.items():
            if isinstance(v, float):
                if k.startswith("Sum") or k.startswith("Avg") or k.startswith("Median") \
                   or k.startswith("Max") or k.startswith("Min"):
                    print(f"  {k:<12}: {v:>+10.4f}R")
                elif k == "WR":
                    print(f"  {k:<12}: {v:>+9.2f}%")
                elif k == "PF":
                    print(f"  {k:<12}: {v:>+10.4f}")
                else:
                    print(f"  {k:<12}: {v}")
            else:
                print(f"  {k:<12}: {v}")

        # Распределение исходов в scaled-mode
        if mode == "scaled":
            cats = {"tp3": 0, "tp": 0, "stop_post_tp1": 0, "pure_stop": 0,
                    "time": 0, "tp2_time": 0, "tp1_time": 0}
            for t in all_trades:
                if t["exit_reason"] == "tp":
                    cats["tp3"] += 1
                elif t["exit_reason"] == "stop":
                    if t.get("tp1_hit"):
                        cats["stop_post_tp1"] += 1
                    else:
                        cats["pure_stop"] += 1
                elif t["exit_reason"] == "time":
                    if t.get("tp2_hit"):
                        cats["tp2_time"] += 1
                    elif t.get("tp1_hit"):
                        cats["tp1_time"] += 1
                    else:
                        cats["time"] += 1
            print(f"\n  Распределение исходов (scaled):")
            print(f"    TP3 (full path)         : {cats['tp3']}")
            print(f"    Stop после TP1 (BE)     : {cats['stop_post_tp1']}")
            print(f"    Pure stop (без TP1)     : {cats['pure_stop']}")
            print(f"    TP2 → time на остатке   : {cats['tp2_time']}")
            print(f"    TP1 → time на остатке   : {cats['tp1_time']}")
            print(f"    Time exit (без TP1)     : {cats['time']}")

    print(f"\n{'═' * W}")
    print(f"  Сравнение с пост-процессорной симуляцией (tp1_sweep_2025.py baseline TP1=0.65R)")
    print(f"  Ожидаем Sum R ≈ +274.68R, WR ≈ 74.4%, PF ≈ 1.891")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()

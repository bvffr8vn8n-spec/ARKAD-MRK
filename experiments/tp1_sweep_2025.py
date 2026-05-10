"""
experiments/tp1_sweep_2025.py
TP1 sweep на ПОЛНОМ 2025 году (Jan 1 – Dec 31, 2025) — 12 месяцев OOS.

Train: 2022-03 → 2024-12 (~33 месяца)
Test : 2025-01-01 → 2025-12-31 (12 месяцев)

5m данные берём из _5m_4y.csv (покрывают 2022-2026).
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

TP1_VALUES = [0.30, 0.40, 0.50, 0.65, 0.80]
TP2_R = 1.00
TP3_R = 1.67
SIZE_TP1 = 0.50
SIZE_TP2 = 0.25
SIZE_TP3 = 0.25
RISK_USD = 18.0


def _load_1h(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace(t, df_1h, tp1_R):
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return {"final_R": 0, "first_action": "Time", "outcome": "no_atr"}

    if direction == 1:
        tp1, tp2, tp3 = entry + tp1_R*sl_dist, entry + TP2_R*sl_dist, entry + TP3_R*sl_dist
        sl_lvl, be    = entry - sl_dist, entry
    else:
        tp1, tp2, tp3 = entry - tp1_R*sl_dist, entry - TP2_R*sl_dist, entry - TP3_R*sl_dist
        sl_lvl, be    = entry + sl_dist, entry

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]
    if len(window) == 0:
        return {"final_R": 0, "first_action": "Time", "outcome": "no_window"}

    phase = "pre_tp1"
    locked_r = 0.0
    open_size = 1.0
    first_action = None
    tp1_bar = tp2_bar = tp3_bar = be_bar = sl_bar = None

    for i in range(len(window)):
        b = window.iloc[i]
        hi, lo = float(b["high"]), float(b["low"])

        if phase == "pre_tp1":
            if direction == 1:
                sl_hit  = lo <= sl_lvl
                tp1_hit = hi >= tp1
            else:
                sl_hit  = hi >= sl_lvl
                tp1_hit = lo <= tp1
            if sl_hit:
                sl_bar = i + 1
                first_action = "SL"
                locked_r += open_size * (-1.0)
                open_size = 0.0
                break
            if tp1_hit:
                tp1_bar = i + 1
                first_action = "TP1"
                locked_r += SIZE_TP1 * tp1_R
                open_size -= SIZE_TP1
                phase = "post_tp1"
                if direction == 1:
                    if hi >= tp2 and tp2_bar is None:
                        tp2_bar = i + 1
                        locked_r += SIZE_TP2 * TP2_R
                        open_size -= SIZE_TP2
                    if hi >= tp3:
                        tp3_bar = i + 1
                        locked_r += SIZE_TP3 * TP3_R
                        open_size = 0.0
                        break
                else:
                    if lo <= tp2 and tp2_bar is None:
                        tp2_bar = i + 1
                        locked_r += SIZE_TP2 * TP2_R
                        open_size -= SIZE_TP2
                    if lo <= tp3:
                        tp3_bar = i + 1
                        locked_r += SIZE_TP3 * TP3_R
                        open_size = 0.0
                        break
                continue

        else:
            if direction == 1:
                be_hit  = lo <= be
                tp2_hit = hi >= tp2 and tp2_bar is None
                tp3_hit = hi >= tp3
            else:
                be_hit  = hi >= be
                tp2_hit = lo <= tp2 and tp2_bar is None
                tp3_hit = lo <= tp3
            if be_hit:
                be_bar = i + 1
                open_size = 0.0
                break
            if tp2_hit:
                tp2_bar = i + 1
                locked_r += SIZE_TP2 * TP2_R
                open_size -= SIZE_TP2
            if tp3_hit:
                tp3_bar = i + 1
                locked_r += SIZE_TP3 * TP3_R
                open_size = 0.0
                break

    if open_size > 0:
        last_close = float(window.iloc[-1]["close"])
        r_close = (last_close - entry) * direction / sl_dist
        locked_r += open_size * r_close
        if first_action is None:
            first_action = "Time"

    if first_action == "SL":
        outcome = "pure_stop"
    elif tp3_bar is not None:
        outcome = "tp3"
    elif tp2_bar is not None and be_bar is not None:
        outcome = "tp2_be"
    elif tp2_bar is not None:
        outcome = "tp2_time"
    elif tp1_bar is not None and be_bar is not None:
        outcome = "tp1_be"
    elif tp1_bar is not None:
        outcome = "tp1_time"
    else:
        outcome = "time_no_tp1"

    return {"final_R": locked_r, "first_action": first_action, "outcome": outcome}


def _run_asset(cfg, tp1_values):
    df = _load_1h(cfg["h1"])
    f = generate_features(df.copy())
    f = add_labels(f)
    f = add_regime_columns(f)
    f = add_session_column(f)
    f = f.dropna()

    train = f[f.index < TEST_START]
    test  = f[(f.index >= TEST_START) & (f.index <= TEST_END)]
    if len(train) < 200 or len(test) == 0:
        return {tp: [] for tp in tp1_values}

    cols = get_feature_columns(train)
    model = fit_model(train, cols)
    sig = apply_signals(model, cols, test.copy())

    df15 = load_5m_as_15m(cfg["m5"])
    ann  = annotate_signals_AB(sig, df15)
    ann["signal"] = ann["signal_15m_A"]

    trades, _ = run_backtest(ann)
    results = {tp: [] for tp in tp1_values}
    for t in trades:
        t["symbol"] = cfg["symbol"]
        for tp in tp1_values:
            info = _trace(t, df, tp)
            info["symbol"] = cfg["symbol"]
            info["entry_date"] = t["entry_date"]
            results[tp].append(info)
    return results


def _stats(trades):
    rs = np.array([t["final_R"] for t in trades])
    N = len(rs)
    wins = rs > 0
    losses = rs < 0
    pf = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")
    return {
        "N": N,
        "wr": wins.sum()/N if N else 0,
        "avg_r": rs.mean() if N else 0,
        "sum_r": rs.sum() if N else 0,
        "pf": pf,
        "median": np.median(rs) if N else 0,
    }


def _outcome_breakdown(trades):
    counts = {}
    for t in trades:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    return counts


def main():
    W = 92
    print("═" * W)
    print("  ARKAD MRK — TP1 SWEEP на ПОЛНОМ 2025 году")
    print(f"  Train: 2022-03 → 2024-12  |  Test: {TEST_START.date()} → {TEST_END.date()}")
    print("═" * W)

    print(f"\n  Прогоняем {len(ASSETS)} актива × {len(TP1_VALUES)} значений TP1...")
    all_results = {tp: [] for tp in TP1_VALUES}
    asset_counts = {}
    for cfg in ASSETS:
        res = _run_asset(cfg, TP1_VALUES)
        for tp in TP1_VALUES:
            all_results[tp].extend(res[tp])
        n = len(res[TP1_VALUES[0]])
        asset_counts[cfg["symbol"]] = n
        print(f"    {cfg['symbol']:<10}  {n} сделок")
    total = sum(asset_counts.values())
    print(f"  ─────────────────")
    print(f"    Всего          {total} сделок")

    # ── Сводная таблица ──────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  СВОДНАЯ ТАБЛИЦА — все {total} сделок при разных TP1 (12 месяцев OOS)")
    print(f"{'═' * W}")
    print(f"  {'TP1':>6}  {'N':>4}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>10}  "
          f"{'PF':>7}  {'Median':>8}  {'PnL$':>10}")
    print(f"  {'─' * (W-2)}")

    rows = []
    for tp in TP1_VALUES:
        s = _stats(all_results[tp])
        pnl = s["sum_r"] * RISK_USD
        rows.append((tp, s, pnl))
        marker = " ← BASELINE" if tp == 0.65 else ""
        print(f"  {tp:>5.2f}R  {s['N']:>4}  {s['wr']*100:>6.1f}%  "
              f"{s['avg_r']:>+8.4f}R  {s['sum_r']:>+9.2f}R  "
              f"{s['pf']:>7.3f}  {s['median']:>+7.4f}R  "
              f"${pnl:>+8.0f}{marker}")

    best = max(rows, key=lambda r: r[1]["sum_r"])
    print(f"\n  → Лучший по Sum R:  TP1 = {best[0]:.2f}R  →  +{best[1]['sum_r']:.2f}R "
          f"(${best[2]:+,.0f})")

    base = next(r for r in rows if r[0] == 0.65)[1]
    print(f"\n  Дельта vs baseline (TP1=0.65R, Sum R = {base['sum_r']:.2f}R):")
    for tp, s, pnl in rows:
        d_r = s["sum_r"] - base["sum_r"]
        d_usd = d_r * RISK_USD
        marker = "  (baseline)" if tp == 0.65 else ""
        print(f"    TP1={tp:.2f}R:  ΔR = {d_r:+8.2f}R  (${d_usd:+,.0f}){marker}")

    # ── Распределение исходов ────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  РАСПРЕДЕЛЕНИЕ ИСХОДОВ по каждому TP1 (количество)")
    print(f"{'═' * W}")
    cats = ["tp3", "tp2_be", "tp2_time", "tp1_be", "tp1_time", "pure_stop", "time_no_tp1"]
    cat_labels = {
        "tp3":         "TP3 full",
        "tp2_be":      "TP2→BE",
        "tp2_time":    "TP2→Time",
        "tp1_be":      "TP1→BE",
        "tp1_time":    "TP1→Time",
        "pure_stop":   "Pure stop",
        "time_no_tp1": "Time(noTP1)",
    }
    print(f"  {'TP1':<7}", end="")
    for c in cats:
        print(f"{cat_labels[c]:>13}", end="")
    print()
    print("  " + "─" * (W - 2))
    for tp in TP1_VALUES:
        bd = _outcome_breakdown(all_results[tp])
        print(f"  {tp:.2f}R  ", end="")
        for c in cats:
            n = bd.get(c, 0)
            print(f"{n:>13}", end="")
        print()

    # ── Per-asset для лучшего ────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  PER-ASSET для оптимального TP1={best[0]:.2f}R")
    print(f"{'═' * W}")
    by_asset = {}
    for t in all_results[best[0]]:
        by_asset.setdefault(t["symbol"], []).append(t)
    print(f"  {'Symbol':<12}  {'N':>4}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>9}  "
          f"{'PF':>7}  {'PnL$':>10}")
    for sym, trs in sorted(by_asset.items()):
        s = _stats(trs)
        pnl = s["sum_r"] * RISK_USD
        print(f"  {sym:<12}  {s['N']:>4}  {s['wr']*100:>6.1f}%  "
              f"{s['avg_r']:>+8.4f}R  {s['sum_r']:>+8.2f}R  "
              f"{s['pf']:>7.3f}  ${pnl:>+8.0f}")

    # ── Per-asset для baseline ──────────────────────────────────────────────
    print(f"\n  PER-ASSET для baseline TP1=0.65R")
    print(f"  {'─' * (W - 4)}")
    by_asset = {}
    for t in all_results[0.65]:
        by_asset.setdefault(t["symbol"], []).append(t)
    print(f"  {'Symbol':<12}  {'N':>4}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>9}  "
          f"{'PF':>7}  {'PnL$':>10}")
    for sym, trs in sorted(by_asset.items()):
        s = _stats(trs)
        pnl = s["sum_r"] * RISK_USD
        print(f"  {sym:<12}  {s['N']:>4}  {s['wr']*100:>6.1f}%  "
              f"{s['avg_r']:>+8.4f}R  {s['sum_r']:>+8.2f}R  "
              f"{s['pf']:>7.3f}  ${pnl:>+8.0f}")

    # ── Сравнение с Jan-Apr 2026 (ранее) ────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ROBUSTNESS CHECK: TP1 OPT в обоих окнах")
    print(f"{'═' * W}")
    print(f"  Окно                      | TP1=0.40R Sum R    | TP1=0.65R Sum R    | Лидер")
    print(f"  ──────────────────────────|--------------------|--------------------|----------")
    s40 = next(r for r in rows if r[0] == 0.40)[1]
    s65 = next(r for r in rows if r[0] == 0.65)[1]
    leader = "0.40R" if s40["sum_r"] > s65["sum_r"] else "0.65R"
    print(f"  Jan-Apr 2026 (197 trades) | +43.16R  ($777)    | +35.46R  ($638)    | 0.40R")
    print(f"  Full 2025  ({total:>4} trades) | "
          f"{s40['sum_r']:+7.2f}R  (${s40['sum_r']*RISK_USD:+5.0f})    | "
          f"{s65['sum_r']:+7.2f}R  (${s65['sum_r']*RISK_USD:+5.0f})    | {leader}")

    print(f"\n{'═' * W}\n")


if __name__ == "__main__":
    main()

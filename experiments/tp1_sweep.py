"""
experiments/tp1_sweep.py
Параметрический sweep по TP1 (0.30 / 0.40 / 0.50 / 0.65 / 0.80R).

Для каждого значения TP1 — честная bar-by-bar симуляция scaled exit:
  - 50% позиции закрывается на TP1
  - 25% на TP2 = 1.00R (фиксировано)
  - 25% на TP3 = 1.67R (фиксировано)
  - После TP1 → BE стоп на остатке

Сравниваем: total R, WR, PF, распределение исходов.
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

TEST_START = pd.Timestamp("2026-01-01")
TEST_END   = pd.Timestamp("2026-04-26 23:59:59")

ASSETS = [
    {"symbol": "AVAXUSDT", "hist_1h": "data/AVAXUSDT_1h_4y.csv",
     "new_1h": "data/AVAXUSDT_2026_1h.csv", "new_5m": "data/AVAXUSDT_2026_5m.csv"},
    {"symbol": "ADAUSDT",  "hist_1h": "data/ADAUSDT_1h_4y.csv",
     "new_1h": "data/ADAUSDT_2026_1h.csv",  "new_5m": "data/ADAUSDT_2026_5m.csv"},
    {"symbol": "SOLUSDT",  "hist_1h": "data/SOLUSDT_1h_4y.csv",
     "new_1h": "data/SOLUSDT_2026_1h.csv",  "new_5m": "data/SOLUSDT_2026_5m.csv"},
    {"symbol": "XRPUSDT",  "hist_1h": "data/XRPUSDT_1h_4y.csv",
     "new_1h": "data/XRPUSDT_2026_1h.csv",  "new_5m": "data/XRPUSDT_2026_5m.csv"},
]

TP1_VALUES = [0.30, 0.40, 0.50, 0.65, 0.80]
TP2_R = 1.00
TP3_R = 1.67
SIZE_TP1 = 0.50
SIZE_TP2 = 0.25
SIZE_TP3 = 0.25
RISK_USD = 18.0


def _load_1h(hist, new):
    h = pd.read_csv(hist, parse_dates=["date"]).set_index("date")
    n = pd.read_csv(new,  parse_dates=["date"]).set_index("date")
    df = pd.concat([h, n])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace(t, df_1h, tp1_R):
    """Bar-by-bar симуляция для одной сделки с заданным TP1."""
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
            # Консервативно: SL приоритетнее TP1 в одном баре
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
                # В этом же баре могут сработать TP2 / TP3
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

        else:  # post_tp1
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

    # Time exit на остатке
    if open_size > 0:
        last_close = float(window.iloc[-1]["close"])
        r_close = (last_close - entry) * direction / sl_dist
        locked_r += open_size * r_close
        if first_action is None:
            first_action = "Time"

    # Outcome
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
    df = _load_1h(cfg["hist_1h"], cfg["new_1h"])
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

    df15 = load_5m_as_15m(cfg["new_5m"])
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


def _compute_stats(trades):
    """Считает агрегированные метрики по списку трейдов."""
    rs = np.array([t["final_R"] for t in trades])
    N  = len(rs)
    wins   = rs > 0
    losses = rs < 0
    pf     = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")
    return {
        "N":       N,
        "wr":      wins.sum() / N if N > 0 else 0,
        "avg_r":   rs.mean() if N > 0 else 0,
        "sum_r":   rs.sum() if N > 0 else 0,
        "pf":      pf,
        "median":  np.median(rs) if N > 0 else 0,
        "max_r":   rs.max() if N > 0 else 0,
        "min_r":   rs.min() if N > 0 else 0,
    }


def _outcome_breakdown(trades):
    """Считает распределение по исходам."""
    counts = {}
    for t in trades:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    return counts


def main():
    W = 92
    print("═" * W)
    print("  ARKAD MRK — TP1 PARAMETRIC SWEEP  (Jan 1 – Apr 26, 2026)")
    print("  TP2=1.00R/25%, TP3=1.67R/25%, BE стоп после TP1")
    print("═" * W)

    # Прогон
    print(f"\n  Прогоняем {len(ASSETS)} актива × {len(TP1_VALUES)} значений TP1...")
    all_results = {tp: [] for tp in TP1_VALUES}
    for cfg in ASSETS:
        res = _run_asset(cfg, TP1_VALUES)
        for tp in TP1_VALUES:
            all_results[tp].extend(res[tp])
        print(f"    {cfg['symbol']:<10}  {len(res[TP1_VALUES[0]])} сделок")

    # ── Сводная таблица метрик ──────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  СВОДНАЯ ТАБЛИЦА — все 197 сделок при разных TP1")
    print(f"{'═' * W}")
    print(f"  {'TP1':>6}  {'N':>4}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>9}  "
          f"{'PF':>7}  {'Median':>8}  {'PnL$':>10}")
    print(f"  {'─' * (W-2)}")

    rows = []
    for tp in TP1_VALUES:
        s = _compute_stats(all_results[tp])
        pnl = s["sum_r"] * RISK_USD
        rows.append((tp, s, pnl))
        marker = " ← BASELINE" if tp == 0.65 else ""
        print(f"  {tp:>5.2f}R  {s['N']:>4}  {s['wr']*100:>6.1f}%  "
              f"{s['avg_r']:>+8.4f}R  {s['sum_r']:>+8.2f}R  "
              f"{s['pf']:>7.3f}  {s['median']:>+7.4f}R  "
              f"${pnl:>+8.0f}{marker}")

    # ── Лучший вариант ──────────────────────────────────────────────────────
    best = max(rows, key=lambda r: r[1]["sum_r"])
    print(f"\n  → Лучший по Sum R:  TP1 = {best[0]:.2f}R  →  +{best[1]['sum_r']:.2f}R "
          f"(${best[2]:+,.0f})")

    # ── Дельта vs baseline (0.65R) ──────────────────────────────────────────
    base_s = next(r for r in rows if r[0] == 0.65)[1]
    print(f"\n  Дельта vs baseline (0.65R) Sum R = {base_s['sum_r']:.2f}R:")
    for tp, s, pnl in rows:
        d_r   = s["sum_r"] - base_s["sum_r"]
        d_usd = d_r * RISK_USD
        marker = "  (baseline)" if tp == 0.65 else ""
        print(f"    TP1={tp:.2f}R:  ΔR = {d_r:+7.2f}R  ({d_usd:+.0f}$){marker}")

    # ── Распределение исходов по каждому TP1 ────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  РАСПРЕДЕЛЕНИЕ ИСХОДОВ по каждому TP1")
    print(f"{'═' * W}")
    cats = ["tp3", "tp2_be", "tp2_time", "tp1_be", "tp1_time", "pure_stop", "time_no_tp1"]
    cat_labels = {
        "tp3":         "TP3 full",
        "tp2_be":      "TP2→BE",
        "tp2_time":    "TP2→Time",
        "tp1_be":      "TP1→BE",
        "tp1_time":    "TP1→Time",
        "pure_stop":   "Pure stop",
        "time_no_tp1": "Time (no TP1)",
    }

    print(f"  {'TP1':<7}", end="")
    for c in cats:
        print(f"{cat_labels[c]:>12}", end="")
    print()
    print("  " + "─" * (W - 2))
    for tp in TP1_VALUES:
        bd = _outcome_breakdown(all_results[tp])
        print(f"  {tp:.2f}R  ", end="")
        for c in cats:
            n = bd.get(c, 0)
            print(f"{n:>12}", end="")
        print()

    # ── Per-asset разбивка для лучшего варианта ─────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  PER-ASSET для оптимального TP1={best[0]:.2f}R")
    print(f"{'═' * W}")
    by_asset = {}
    for t in all_results[best[0]]:
        by_asset.setdefault(t["symbol"], []).append(t)
    print(f"  {'Symbol':<12}  {'N':>4}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>9}  "
          f"{'PF':>7}  {'PnL$':>10}")
    for sym, trs in sorted(by_asset.items()):
        s = _compute_stats(trs)
        pnl = s["sum_r"] * RISK_USD
        print(f"  {sym:<12}  {s['N']:>4}  {s['wr']*100:>6.1f}%  "
              f"{s['avg_r']:>+8.4f}R  {s['sum_r']:>+8.2f}R  "
              f"{s['pf']:>7.3f}  ${pnl:>+8.0f}")

    print(f"\n{'═' * W}\n")


if __name__ == "__main__":
    main()

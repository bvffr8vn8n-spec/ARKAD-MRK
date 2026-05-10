"""
experiments/first_action_analysis.py
Анализ ПЕРВОГО действия каждой сделки в scaled exit.

После того как сделка открылась (entry зафиксирован, SL и TP1/TP2/TP3 выставлены),
проходим бар за баром и фиксируем САМОЕ ПЕРВОЕ событие, которое произошло.

Возможные "первые действия":
  TP1_first  — первой коснулась TP1 (+0.65R)  → залок 50%, BE стоп активен
  SL_first   — первой коснулась SL (-1.0R)    → полный убыток
  Time_first — ни TP1 ни SL не были достигнуты → time exit на 24-м баре

Затем для каждой группы прослеживаем дальнейший путь и считаем итоговый R.
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

TP1_R = 0.65
TP2_R = 1.00
TP3_R = 1.67
SIZE_TP1 = 0.50
SIZE_TP2 = 0.25
SIZE_TP3 = 0.25


def _load_1h(hist, new):
    h = pd.read_csv(hist, parse_dates=["date"]).set_index("date")
    n = pd.read_csv(new,  parse_dates=["date"]).set_index("date")
    df = pd.concat([h, n])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace(t, df_1h):
    """
    Bar-by-bar trace.

    Возвращает:
      first_action : 'TP1' | 'SL' | 'Time'
      first_bar    : int, на каком баре произошло
      final_R      : итоговый R-multiple (после учёта пути после первого действия)
      final_outcome: словесное описание финального исхода
    """
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)

    if direction == 1:
        tp1, tp2, tp3 = entry + TP1_R*sl_dist, entry + TP2_R*sl_dist, entry + TP3_R*sl_dist
        sl_lvl, be    = entry - sl_dist, entry
    else:
        tp1, tp2, tp3 = entry - TP1_R*sl_dist, entry - TP2_R*sl_dist, entry - TP3_R*sl_dist
        sl_lvl, be    = entry + sl_dist, entry

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]

    # Шаг 1 — найти первое действие
    first_action = "Time"
    first_bar    = None
    for i, (ts, b) in enumerate(window.iterrows(), 1):
        hi, lo = float(b["high"]), float(b["low"])
        if direction == 1:
            sl_hit  = lo <= sl_lvl
            tp1_hit = hi >= tp1
        else:
            sl_hit  = hi >= sl_lvl
            tp1_hit = lo <= tp1
        if sl_hit and tp1_hit:
            # Одновременно — консервативно SL первым
            first_action = "SL"
            first_bar    = i
            break
        if sl_hit:
            first_action = "SL"
            first_bar    = i
            break
        if tp1_hit:
            first_action = "TP1"
            first_bar    = i
            break

    # Шаг 2 — финализировать R
    if first_action == "SL":
        # Полный убыток на всей позиции
        return {
            "first_action": "SL",
            "first_bar":    first_bar,
            "final_R":      -1.0,
            "outcome":      "Pure stop",
            "tp2_bar":      None,
            "tp3_bar":      None,
            "be_bar":       None,
        }

    if first_action == "Time":
        # Цена не дошла ни до TP1 ни до SL за весь holding period
        last = window.iloc[-1] if len(window) > 0 else None
        if last is not None:
            cl = float(last["close"])
            r  = (cl - entry) * direction / sl_dist if sl_dist > 0 else 0
        else:
            r = 0
        return {
            "first_action": "Time",
            "first_bar":    None,
            "final_R":      r,
            "outcome":      "Time exit (без касания TP1/SL)",
            "tp2_bar":      None,
            "tp3_bar":      None,
            "be_bar":       None,
        }

    # first_action == TP1: продолжаем с залоком 0.325R и BE-стопом на остатке
    locked_r = SIZE_TP1 * TP1_R   # +0.325
    open_size = 1.0 - SIZE_TP1    # = 0.5
    tp2_bar = tp3_bar = be_bar = None

    # Сначала проверим тот же бар, в котором сработал TP1 — мог ли там же TP2/TP3?
    bar0 = window.iloc[first_bar - 1]
    hi0, lo0 = float(bar0["high"]), float(bar0["low"])
    if direction == 1:
        if hi0 >= tp2:
            tp2_bar = first_bar
            locked_r += SIZE_TP2 * TP2_R
            open_size -= SIZE_TP2
        if hi0 >= tp3:
            tp3_bar = first_bar
            locked_r += SIZE_TP3 * TP3_R
            open_size = 0.0
    else:
        if lo0 <= tp2:
            tp2_bar = first_bar
            locked_r += SIZE_TP2 * TP2_R
            open_size -= SIZE_TP2
        if lo0 <= tp3:
            tp3_bar = first_bar
            locked_r += SIZE_TP3 * TP3_R
            open_size = 0.0

    # Если ещё открыто — продолжаем с бара first_bar+1
    if open_size > 0:
        for i in range(first_bar, len(window)):
            b = window.iloc[i]
            hi, lo = float(b["high"]), float(b["low"])
            if direction == 1:
                be_hit  = lo <= be
                tp2_hit = hi >= tp2 and tp2_bar is None
                tp3_hit = hi >= tp3 and tp3_bar is None
            else:
                be_hit  = hi >= be
                tp2_hit = lo <= tp2 and tp2_bar is None
                tp3_hit = lo <= tp3 and tp3_bar is None

            if be_hit:
                be_bar = i + 1
                # остаток выходит по 0R
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

    # Если осталось открытое — time exit
    if open_size > 0:
        last = window.iloc[-1]
        cl   = float(last["close"])
        r_cl = (cl - entry) * direction / sl_dist
        locked_r += open_size * r_cl

    # Описание исхода
    if tp3_bar is not None:
        outcome = "TP3 (full path)"
    elif tp2_bar is not None and be_bar is not None:
        outcome = "TP2 → BE на остатке"
    elif tp2_bar is not None:
        outcome = "TP2 → Time на остатке"
    elif be_bar is not None:
        outcome = "TP1 → BE на остатке"
    else:
        outcome = "TP1 → Time на остатке"

    return {
        "first_action": "TP1",
        "first_bar":    first_bar,
        "final_R":      locked_r,
        "outcome":      outcome,
        "tp2_bar":      tp2_bar,
        "tp3_bar":      tp3_bar,
        "be_bar":       be_bar,
    }


def _run_asset(cfg):
    df = _load_1h(cfg["hist_1h"], cfg["new_1h"])
    f = generate_features(df.copy())
    f = add_labels(f)
    f = add_regime_columns(f)
    f = add_session_column(f)
    f = f.dropna()

    train = f[f.index < TEST_START]
    test  = f[(f.index >= TEST_START) & (f.index <= TEST_END)]
    if len(train) < 200 or len(test) == 0:
        return []

    cols = get_feature_columns(train)
    model = fit_model(train, cols)
    sig = apply_signals(model, cols, test.copy())

    df15 = load_5m_as_15m(cfg["new_5m"])
    ann  = annotate_signals_AB(sig, df15)
    ann["signal"] = ann["signal_15m_A"]

    trades, _ = run_backtest(ann)
    for t in trades:
        t["symbol"] = cfg["symbol"]
        t.update(_trace(t, df))
    return trades


def main():
    W = 84
    print("═" * W)
    print("  ARKAD MRK — FIRST ACTION после открытия сделки (Scaled Exit)")
    print("  Что произошло первым: TP1 коснулся, SL коснулся, или Time exit?")
    print("═" * W)

    all_trades = []
    for cfg in ASSETS:
        all_trades.extend(_run_asset(cfg))
    N = len(all_trades)

    # ── ПЕРВОЕ ДЕЙСТВИЕ ──────────────────────────────────────────────────────
    by_first = {"TP1": [], "SL": [], "Time": []}
    for t in all_trades:
        by_first[t["first_action"]].append(t)

    print(f"\n[ПЕРВОЕ ДЕЙСТВИЕ ПОСЛЕ ВХОДА — распределение по {N} сделкам]")
    print("─" * W)
    print(f"  {'Первое действие':<30}  {'Сделок':>7}  {'%':>5}  {'Avg R':>9}  {'Sum R':>9}")
    for fa, lab in [("TP1", "TP1 коснулся (+0.65R, лок 50%)"),
                    ("SL",  "SL коснулся (полный -1R)"),
                    ("Time", "Ни TP1 ни SL → Time exit")]:
        grp = by_first[fa]
        n = len(grp)
        if n == 0:
            print(f"  {lab:<30}  {n:>7}  {n/N*100:>4.1f}%  {'—':>9}  {'—':>9}")
            continue
        avg_r = np.mean([t["final_R"] for t in grp])
        sum_r = sum(t["final_R"] for t in grp)
        print(f"  {lab:<30}  {n:>7}  {n/N*100:>4.1f}%  {avg_r:>+8.4f}R  {sum_r:>+8.2f}R")

    # ── ДЕТАЛИЗАЦИЯ ВЕТКИ TP1_FIRST ─────────────────────────────────────────
    tp1_grp = by_first["TP1"]
    if tp1_grp:
        sub_outcomes = {}
        for t in tp1_grp:
            sub_outcomes.setdefault(t["outcome"], []).append(t)

        print(f"\n[ИЗ {len(tp1_grp)} СДЕЛОК С 'TP1 ПЕРВЫМ' — что было дальше с остатком]")
        print("─" * W)
        print(f"  {'Финальный исход':<32}  {'N':>4}  {'%':>5}  {'Avg R':>9}  {'Sum R':>9}")
        for out, grp in sorted(sub_outcomes.items(), key=lambda x: -len(x[1])):
            n = len(grp)
            avg = np.mean([t["final_R"] for t in grp])
            tot = sum(t["final_R"] for t in grp)
            print(f"  {out:<32}  {n:>4}  {n/len(tp1_grp)*100:>4.1f}%  "
                  f"{avg:>+8.4f}R  {tot:>+8.2f}R")

    # ── СКОРОСТЬ ПЕРВОГО ДЕЙСТВИЯ (баров от входа) ──────────────────────────
    print(f"\n[СКОРОСТЬ ПЕРВОГО ДЕЙСТВИЯ — медиана баров]")
    print("─" * W)
    for fa, lab in [("TP1", "TP1 коснулся"), ("SL", "SL коснулся")]:
        grp = by_first[fa]
        if not grp:
            continue
        bars = [t["first_bar"] for t in grp if t["first_bar"] is not None]
        print(f"  {lab:<25}  N={len(bars):>3}  медиана={int(np.median(bars))} ч  "
              f"среднее={np.mean(bars):.1f} ч  "
              f"мин={min(bars)} ч  макс={max(bars)} ч")

    # ── ИТОГОВАЯ СТАТИСТИКА ─────────────────────────────────────────────────
    rs = np.array([t["final_R"] for t in all_trades])
    wins = rs > 0
    losses = rs < 0
    pf = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")

    print(f"\n[ИТОГОВАЯ СТАТИСТИКА SCALED EXIT по 197 сделкам]")
    print("─" * W)
    print(f"  Win rate (R > 0)     : {wins.sum()} / {N} = {wins.sum()/N*100:.1f}%")
    print(f"  Loss rate (R < 0)    : {losses.sum()} / {N} = {losses.sum()/N*100:.1f}%")
    print(f"  Avg R                : {rs.mean():+.4f}R")
    print(f"  Sum R                : {rs.sum():+.2f}R")
    print(f"  Profit factor        : {pf:.3f}")
    print(f"  Median R             : {np.median(rs):+.4f}R")

    # ── ВЫВОД ────────────────────────────────────────────────────────────────
    n_tp1  = len(by_first["TP1"])
    n_sl   = len(by_first["SL"])
    n_time = len(by_first["Time"])

    tp1_to_loss = sum(1 for t in by_first["TP1"] if t["final_R"] < 0)
    print(f"\n[КЛЮЧЕВЫЕ ВЫВОДЫ]")
    print("─" * W)
    print(f"  • Из {N} сделок: TP1-first={n_tp1} ({n_tp1/N*100:.0f}%), "
          f"SL-first={n_sl} ({n_sl/N*100:.0f}%), Time-first={n_time}")
    print(f"  • После TP1 убыточных трейдов: {tp1_to_loss} (BE-стоп защищает на 100%)")
    print(f"  • Всего убытков ровно от чистых стопов: {n_sl}")
    print(f"  • Edge формируется здесь: TP1-first группа залочила "
          f"{sum(t['final_R'] for t in by_first['TP1']):+.2f}R")
    print(f"    SL-first группа отняла {sum(t['final_R'] for t in by_first['SL']):+.2f}R")
    print(f"    Чистый итог: {rs.sum():+.2f}R")
    print()


if __name__ == "__main__":
    main()

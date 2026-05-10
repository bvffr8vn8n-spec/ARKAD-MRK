"""
experiments/scaled_exit_path.py
Честная bar-by-bar симуляция Scaled Exit (TP1=0.65R/50%, TP2=1.0R/25%,
TP3=1.67R/25%, BE стоп после TP1) поверх 197 сделок Jan-Apr 2026.

Не MFE-based — реальный walk через 1H бары с проверкой порядка событий.

Для каждой сделки определяет окончательный категорированный исход:
  TP3              — все три тейка сработали
  TP2 + BE         — TP1 и TP2 сработали, остаток вышел по BE
  TP1 + BE         — TP1 сработал, остаток вышел по BE
  TP1 + Time       — TP1 сработал, остаток вышел по времени (плюсовой)
  Pure Stop        — SL без касания TP1 (полный -1R)
  Time (no TP1)    — Время вышло, не достигнут даже TP1
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


def _scaled_path(t, df_1h):
    """
    Bar-by-bar walk. Возвращает категорию + бары событий + итоговый R.

    Правила:
      Перед TP1 (полная позиция):
        SL hit  → category = pure_stop, r = -1.0
        TP1 hit → залок 50% × 0.65, BE стоп активируется, продолжаем
      После TP1 (остаток 50%):
        BE стоп hit → 25% TP2 уже могло быть закрыто
        TP2 hit → залок 25% × 1.0, продолжаем
        TP3 hit → залок 25% × 1.67, полный выход

      Внутрибарный порядок (консервативно, как в engine):
        Перед TP1: SL приоритетнее TP1 в одном баре
        После TP1: BE стоп приоритетнее TP2/TP3 в одном баре
    """
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)

    if direction == 1:
        tp1 = entry + TP1_R * sl_dist
        tp2 = entry + TP2_R * sl_dist
        tp3 = entry + TP3_R * sl_dist
        sl_lvl = entry - sl_dist
        be     = entry
    else:
        tp1 = entry - TP1_R * sl_dist
        tp2 = entry - TP2_R * sl_dist
        tp3 = entry - TP3_R * sl_dist
        sl_lvl = entry + sl_dist
        be     = entry

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]

    # State
    phase = "pre_tp1"   # pre_tp1 | post_tp1
    tp1_bar = tp2_bar = tp3_bar = sl_bar = be_bar = None
    locked_r = 0.0
    open_size = 1.0   # ещё открыто 100% позиции
    final_close = float(window["close"].iloc[-1]) if len(window) > 0 else entry

    for i, (ts, b) in enumerate(window.iterrows(), 1):
        hi, lo, cl = float(b["high"]), float(b["low"]), float(b["close"])

        if phase == "pre_tp1":
            if direction == 1:
                sl_hit  = lo <= sl_lvl
                tp1_hit = hi >= tp1
            else:
                sl_hit  = hi >= sl_lvl
                tp1_hit = lo <= tp1
            if sl_hit:
                # SL приоритетнее в одном баре
                sl_bar = i
                locked_r += open_size * (-1.0)
                open_size = 0.0
                break
            if tp1_hit:
                tp1_bar = i
                locked_r += SIZE_TP1 * TP1_R
                open_size -= SIZE_TP1
                phase = "post_tp1"
                # В этом же баре могут сработать TP2/TP3 — проверяем дальше
                if direction == 1:
                    tp2_hit = hi >= tp2
                    tp3_hit = hi >= tp3
                else:
                    tp2_hit = lo <= tp2
                    tp3_hit = lo <= tp3
                if tp2_hit:
                    tp2_bar = i
                    locked_r += SIZE_TP2 * TP2_R
                    open_size -= SIZE_TP2
                if tp3_hit:
                    tp3_bar = i
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
            # BE стоп приоритетнее (консервативно)
            if be_hit:
                be_bar = i
                # остаток выходит по 0R
                open_size = 0.0
                break
            if tp2_hit:
                tp2_bar = i
                locked_r += SIZE_TP2 * TP2_R
                open_size -= SIZE_TP2
            if tp3_hit:
                tp3_bar = i
                locked_r += SIZE_TP3 * TP3_R
                open_size = 0.0
                break

    # Если позиция не закрылась — time exit на close последнего бара
    if open_size > 0:
        diff = (final_close - entry) * direction
        r = diff / sl_dist if sl_dist > 0 else 0
        locked_r += open_size * r

    # Категоризация
    if sl_bar is not None and tp1_bar is None:
        cat = "pure_stop"
    elif tp3_bar is not None:
        cat = "tp3_full"
    elif tp2_bar is not None and be_bar is not None:
        cat = "tp2_be"
    elif tp2_bar is not None:
        cat = "tp2_time"
    elif tp1_bar is not None and be_bar is not None:
        cat = "tp1_be"
    elif tp1_bar is not None:
        cat = "tp1_time"
    else:
        cat = "time_no_tp1"

    return {
        "cat":      cat,
        "tp1_bar":  tp1_bar,
        "tp2_bar":  tp2_bar,
        "tp3_bar":  tp3_bar,
        "be_bar":   be_bar,
        "sl_bar":   sl_bar,
        "r_scaled": locked_r,
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
        t.update(_scaled_path(t, df))
    return trades


def main():
    W = 84
    print("═" * W)
    print("  ARKAD MRK — SCALED EXIT: Bar-by-Bar Path Analysis")
    print("  TP1=0.65R/50% | TP2=1.0R/25% | TP3=1.67R/25% | BE стоп после TP1")
    print("═" * W)

    all_trades = []
    for cfg in ASSETS:
        all_trades.extend(_run_asset(cfg))

    N = len(all_trades)
    cats = {
        "tp3_full":    "TP3 (полный)            +0.9925R",
        "tp2_be":      "TP2 потом BE            +0.5750R",
        "tp2_time":    "TP2 потом time exit     варьируется",
        "tp1_be":      "TP1 потом BE            +0.3250R",
        "tp1_time":    "TP1 потом time exit     варьируется",
        "pure_stop":   "Чистый стоп (без TP1)   -1.0000R",
        "time_no_tp1": "Time exit без TP1       варьируется",
    }
    counts = {c: 0 for c in cats}
    for t in all_trades:
        counts[t["cat"]] += 1

    print(f"\n[РАСПРЕДЕЛЕНИЕ {N} СДЕЛОК ПО КАТЕГОРИЯМ]")
    print("─" * W)
    print(f"  {'Категория':<40}  {'N':>4}  {'%':>5}")
    for c, lab in cats.items():
        n = counts[c]
        print(f"  {lab:<40}  {n:>4}  {n/N*100:>4.1f}%")

    # ── Сводная статистика ───────────────────────────────────────────────────
    rs = np.array([t["r_scaled"] for t in all_trades])
    wins = rs > 0
    losses = rs < 0
    breakevens = rs == 0
    pf = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")

    print(f"\n[ОБЩИЕ МЕТРИКИ SCALED EXIT — bar-by-bar]")
    print("─" * W)
    print(f"  Win rate (R > 0)         : {wins.sum()} / {N} = {wins.sum()/N*100:.1f}%")
    print(f"  Loss rate (R < 0)        : {losses.sum()} / {N} = {losses.sum()/N*100:.1f}%")
    print(f"  Break-even (R = 0)       : {breakevens.sum()} / {N} = {breakevens.sum()/N*100:.1f}%")
    print(f"  Avg R                    : {rs.mean():+.4f}R")
    print(f"  Sum R                    : {rs.sum():+.2f}R")
    print(f"  Profit factor            : {pf:.3f}")
    print(f"  Median R                 : {np.median(rs):+.4f}R")

    # ── Сколько баров в среднем до TP1 / TP2 / TP3 / SL ─────────────────────
    print(f"\n[ВРЕМЯ ДО СОБЫТИЙ — медиана баров от входа]")
    print("─" * W)
    for evt, key in [("TP1 коснулся", "tp1_bar"),
                     ("TP2 коснулся", "tp2_bar"),
                     ("TP3 коснулся", "tp3_bar"),
                     ("BE стоп сработал", "be_bar"),
                     ("SL без TP1", "sl_bar")]:
        bars = [t[key] for t in all_trades if t[key] is not None]
        if bars:
            print(f"  {evt:<22} N={len(bars):>3}  медиана={int(np.median(bars))} ч  "
                  f"среднее={np.mean(bars):.1f} ч")

    # ── Что случается ПОСЛЕ TP1 — детально ───────────────────────────────────
    after_tp1 = [t for t in all_trades if t["tp1_bar"] is not None]
    n_after  = len(after_tp1)
    n_be     = sum(1 for t in after_tp1 if t["be_bar"] is not None)
    n_tp3    = sum(1 for t in after_tp1 if t["tp3_bar"] is not None)
    n_tp2only= sum(1 for t in after_tp1
                   if t["tp2_bar"] is not None and t["tp3_bar"] is None and t["be_bar"] is None)

    print(f"\n[ПОСЛЕ КАСАНИЯ TP1] что было с остатком позиции (50%)?")
    print("─" * W)
    print(f"  Всего трейдов с TP1               : {n_after} ({n_after/N*100:.1f}%)")
    print(f"  ├─ Дошли до TP3 (полный путь)     : {n_tp3} ({n_tp3/n_after*100:.1f}%)")
    print(f"  ├─ Закрыли TP2, остаток time      : {n_tp2only} ({n_tp2only/n_after*100:.1f}%)")
    print(f"  └─ Откатилось → BE стоп сработал  : {n_be} ({n_be/n_after*100:.1f}%)")

    # ── Сравнение со старым MFE-based анализом ───────────────────────────────
    pnl_per_r = 18.0  # риск $18 при риске 1.5×ATR на $1000 позиции (примерно)
    sum_r_orig = 197 * 0.0866  # Avg R оригинала
    print(f"\n[СРАВНЕНИЕ С ОРИГИНАЛЬНОЙ СИСТЕМОЙ]")
    print("─" * W)
    print(f"  Оригинал (1 TP=1.67R):  Avg R = +0.087R | PF = 1.25 | Sum = +{sum_r_orig:.1f}R")
    print(f"  Scaled bar-by-bar:      Avg R = {rs.mean():+.4f}R | PF = {pf:.3f} | "
          f"Sum = {rs.sum():+.1f}R")
    print(f"  Дельта в R:                  {rs.sum() - sum_r_orig:+.1f}R")
    print(f"  Дельта в $ (риск $18/R):     ${(rs.sum() - sum_r_orig) * pnl_per_r:+,.0f}")

    print(f"\n{'═' * W}")


if __name__ == "__main__":
    main()

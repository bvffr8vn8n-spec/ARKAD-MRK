"""
experiments/sl_group_mfe.py
Для 57 сделок группы 'SL-first' (стоп сработал ДО касания TP1) считаем,
насколько максимально цена УХОДИЛА В НАШУ СТОРОНУ перед разворотом к SL.

MFE_R = максимальное движение по позиции в R-единицах (где 1R = расстояние до SL).
Поскольку TP1 = 0.65R, MFE для этой группы по определению < 0.65R.
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


def _load_1h(hist, new):
    h = pd.read_csv(hist, parse_dates=["date"]).set_index("date")
    n = pd.read_csv(new,  parse_dates=["date"]).set_index("date")
    df = pd.concat([h, n])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace_sl_first(t, df_1h):
    """Если сделка — SL-first, возвращает MFE до SL в R-единицах. Иначе None."""
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return None

    if direction == 1:
        tp1_lvl = entry + TP1_R * sl_dist
        sl_lvl  = entry - sl_dist
    else:
        tp1_lvl = entry - TP1_R * sl_dist
        sl_lvl  = entry + sl_dist

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]

    best_favourable_R = 0.0
    sl_first = None     # True если SL коснулся раньше TP1

    for i, (ts, b) in enumerate(window.iterrows(), 1):
        hi, lo = float(b["high"]), float(b["low"])

        # MFE на этом баре
        if direction == 1:
            best_in_bar = hi
            fav_R = (best_in_bar - entry) / sl_dist
        else:
            best_in_bar = lo
            fav_R = (entry - best_in_bar) / sl_dist
        if fav_R > best_favourable_R:
            best_favourable_R = fav_R

        # Проверяем SL и TP1 (консервативно: SL первым в одном баре)
        if direction == 1:
            sl_hit  = lo <= sl_lvl
            tp1_hit = hi >= tp1_lvl
        else:
            sl_hit  = hi >= sl_lvl
            tp1_hit = lo <= tp1_lvl

        if sl_hit and tp1_hit:
            sl_first = True   # SL "первым" по консервативному правилу
            break
        if sl_hit:
            sl_first = True
            break
        if tp1_hit:
            sl_first = False
            break

    if sl_first is None or sl_first is False:
        return None     # не SL-first сделка

    # Для MFE нам важно "сколько ушло в плюс ДО касания SL"
    # best_favourable_R уже учитывает все бары вплоть до бара со стопом включительно;
    # но это ОК — high бара со стопом тоже мог быть в плюсе перед обвалом
    return {
        "symbol":      t["symbol"],
        "entry_date":  t["entry_date"],
        "direction":   t["direction"],
        "mfe_R":       min(best_favourable_R, TP1_R - 0.0001),
        "raw_mfe_R":   best_favourable_R,
        "exit_date":   t["exit_date"],
        "bars_held":   len(window),
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
    out = []
    for t in trades:
        t["symbol"] = cfg["symbol"]
        info = _trace_sl_first(t, df)
        if info is not None:
            out.append(info)
    return out


def main():
    W = 84
    print("═" * W)
    print("  ARKAD MRK — Группа 'SL-first': MFE до разворота к SL")
    print("  Насколько в плюс уходила цена перед тем как пойти в стоп?")
    print("═" * W)

    sl_first = []
    for cfg in ASSETS:
        sl_first.extend(_run_asset(cfg))

    N = len(sl_first)
    if N == 0:
        print("Нет SL-first сделок.")
        return

    mfes = np.array([t["mfe_R"] for t in sl_first])

    print(f"\n[ОБЩАЯ СТАТИСТИКА MFE для {N} 'SL-first' сделок]")
    print("─" * W)
    print(f"  Минимум     : {mfes.min():+.4f}R   (цена сразу пошла в SL без отскока)")
    print(f"  Максимум    : {mfes.max():+.4f}R   (почти до TP1, но развернулась)")
    print(f"  Среднее MFE : {mfes.mean():+.4f}R")
    print(f"  Медиана MFE : {np.median(mfes):+.4f}R")
    print(f"  Std         : {mfes.std():+.4f}R")

    # ── Распределение по корзинам ───────────────────────────────────────────
    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65]
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}R" for i in range(len(bins)-1)]
    counts, _ = np.histogram(mfes, bins=bins)

    print(f"\n[РАСПРЕДЕЛЕНИЕ MFE по корзинам]")
    print("─" * W)
    print(f"  {'MFE (в R)':<15}  {'Сделок':>7}  {'%':>6}  {'Гистограмма':<40}")
    mx = max(counts) if counts.max() > 0 else 1
    for lab, c in zip(labels, counts):
        bar = "█" * int(c / mx * 40)
        pct = c / N * 100
        print(f"  {lab:<15}  {c:>7}  {pct:>5.1f}%  {bar}")

    # ── Топ-10 'обиднее всего' (близко к TP1, но не дошло) ─────────────────
    sorted_by_mfe = sorted(sl_first, key=lambda x: x["mfe_R"], reverse=True)
    print(f"\n[TOP-10 'почти-победителей' (MFE ближе всего к TP1=0.65R)]")
    print("─" * W)
    print(f"  {'Символ':<10}  {'Дата':<16}  {'Dir':<5}  {'MFE_R':>8}  "
          f"{'% к TP1':>8}  {'Bars':>5}")
    for t in sorted_by_mfe[:10]:
        pct = t["mfe_R"] / TP1_R * 100
        print(f"  {t['symbol']:<10}  {str(t['entry_date'])[:16]:<16}  "
              f"{t['direction']:<5}  {t['mfe_R']:>+8.4f}  {pct:>7.1f}%  "
              f"{t['bars_held']:>5}")

    # ── Bottom-10 (сразу в SL без отскока) ──────────────────────────────────
    print(f"\n[BOTTOM-10 'сразу в SL без отскока' (MFE ближе всего к 0)]")
    print("─" * W)
    print(f"  {'Символ':<10}  {'Дата':<16}  {'Dir':<5}  {'MFE_R':>8}  "
          f"{'% к TP1':>8}  {'Bars':>5}")
    for t in sorted_by_mfe[-10:]:
        pct = t["mfe_R"] / TP1_R * 100
        print(f"  {t['symbol']:<10}  {str(t['entry_date'])[:16]:<16}  "
              f"{t['direction']:<5}  {t['mfe_R']:>+8.4f}  {pct:>7.1f}%  "
              f"{t['bars_held']:>5}")

    # ── Сегментация по "глубине отката" ────────────────────────────────────
    n_zero      = sum(1 for m in mfes if m < 0.05)
    n_minor     = sum(1 for m in mfes if 0.05 <= m < 0.20)
    n_medium    = sum(1 for m in mfes if 0.20 <= m < 0.40)
    n_close_tp1 = sum(1 for m in mfes if m >= 0.40)

    print(f"\n[СЕГМЕНТАЦИЯ ПО ГЛУБИНЕ MFE]")
    print("─" * W)
    print(f"  Сразу в SL (MFE < 0.05R)        : {n_zero}  ({n_zero/N*100:.1f}%) — "
          f"чистый разворот после входа")
    print(f"  Лёгкий отскок (0.05-0.20R)      : {n_minor}  ({n_minor/N*100:.1f}%)")
    print(f"  Средний отскок (0.20-0.40R)     : {n_medium}  ({n_medium/N*100:.1f}%)")
    print(f"  Почти до TP1 (0.40-0.65R)       : {n_close_tp1}  ({n_close_tp1/N*100:.1f}%) "
          f"— "
          f"могли бы стать TP1 при чуть более ранней реакции")

    print()


if __name__ == "__main__":
    main()

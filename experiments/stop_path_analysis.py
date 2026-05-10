"""
experiments/stop_path_analysis.py
Анализ "пути" каждого стопа: дошла ли цена до TP1/TP2 ДО того как снесла SL?

Для каждой сделки с exit_reason='stop' проходим бар за баром и смотрим:
  - В каком баре впервые достигся TP1 (0.65R favourable)?
  - В каком баре впервые достигся TP2 (1.0R favourable)?
  - В каком баре сработал SL (1.0R adverse)?

Если TP1_bar < SL_bar → в scaled exit мы бы успели зафиксировать 50%×0.65R = +0.325R
                       и переехать стоп в BE до пробоя SL.
Если TP1_bar > SL_bar или TP1 не достигнут → "чистый" стоп −1R.

Для каждого 1H бара МЫ НЕ ЗНАЕМ внутрибарный порядок: если в одном баре
high дошёл до TP1 И low дошёл до SL — мы предполагаем что SL раньше
(консервативно, как в backtest engine).
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


def _load_1h(hist, new):
    h = pd.read_csv(hist, parse_dates=["date"]).set_index("date")
    n = pd.read_csv(new,  parse_dates=["date"]).set_index("date")
    df = pd.concat([h, n])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _path_analysis(t, df_1h):
    """
    Возвращает dict с информацией о пути цены в трейде.

    Ключи:
      tp1_bar : бар (1-based), на котором впервые достигнут TP1; None если не достигнут
      tp2_bar : то же для TP2
      sl_bar  : бар, на котором сработал SL; None если не стоп
      reached_tp1_before_sl : bool
      reached_tp2_before_sl : bool
    """
    entry  = t["entry_price"]
    sl     = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)

    if direction == 1:
        tp1_level = entry + TP1_R * sl_dist
        tp2_level = entry + TP2_R * sl_dist
        sl_level  = entry - sl_dist
    else:
        tp1_level = entry - TP1_R * sl_dist
        tp2_level = entry + TP2_R * sl_dist * (-1)   # = entry - TP2_R*sl_dist
        sl_level  = entry + sl_dist

    # Окно баров СРАЗУ ПОСЛЕ входа (вход = close сигнального бара)
    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]

    tp1_bar = tp2_bar = sl_bar = None
    for i, (ts, b) in enumerate(window.iterrows(), 1):
        if direction == 1:
            tp1_hit = b["high"] >= tp1_level
            tp2_hit = b["high"] >= tp2_level
            sl_hit  = b["low"]  <= sl_level
        else:
            tp1_hit = b["low"]  <= tp1_level
            tp2_hit = b["low"]  <= tp2_level
            sl_hit  = b["high"] >= sl_level

        # Консервативно: если в одном баре оба, считаем что SL первым
        if tp1_bar is None and tp1_hit and not sl_hit:
            tp1_bar = i
        if tp2_bar is None and tp2_hit and not sl_hit:
            tp2_bar = i
        if sl_bar is None and sl_hit:
            sl_bar = i
            break

        # Также допускаем что TP1 был достигнут в более ранних барах
        if tp1_bar is None and tp1_hit and sl_hit:
            # одновременный бар — TP1 не засчитываем (консервативно)
            pass

    return {
        "tp1_bar": tp1_bar,
        "tp2_bar": tp2_bar,
        "sl_bar":  sl_bar,
        "reached_tp1_before_sl": tp1_bar is not None and (sl_bar is None or tp1_bar < sl_bar),
        "reached_tp2_before_sl": tp2_bar is not None and (sl_bar is None or tp2_bar < sl_bar),
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
        t.update(_path_analysis(t, df))
    return trades


def main():
    W = 78
    print("═" * W)
    print("  ARKAD MRK — Stop Path Analysis  (Jan 1 – Apr 26, 2026)")
    print("  Дошёл ли каждый стоп до TP1/TP2 ДО того как снёс SL?")
    print("═" * W)

    all_trades = []
    for cfg in ASSETS:
        all_trades.extend(_run_asset(cfg))

    stops = [t for t in all_trades if t["exit_reason"] == "stop"]
    N = len(stops)

    # ── Категоризация стопов ────────────────────────────────────────────────
    cat_pure_stop = []      # ни TP1 ни TP2 не были достигнуты
    cat_tp1_only  = []      # коснулся TP1, потом откатился к SL
    cat_tp2_only  = []      # коснулся TP2, потом откатился к SL (TP1 тоже)
    cat_both_same_bar = []  # TP1 и SL в одном баре

    for t in stops:
        tp1 = t["tp1_bar"]
        tp2 = t["tp2_bar"]
        sl  = t["sl_bar"]
        if tp2 is not None and tp2 < sl:
            cat_tp2_only.append(t)
        elif tp1 is not None and tp1 < sl:
            cat_tp1_only.append(t)
        else:
            cat_pure_stop.append(t)

    print(f"\n[ВСЕ {N} СТОПОВ] КАТЕГОРИЗАЦИЯ ПО ПУТИ ЦЕНЫ")
    print("─" * W)
    print(f"  Категория                                 Сделок     %      Что бы дал scaled exit")
    print(f"  {'─' * 76}")
    pct = lambda c: f"{c/N*100:.1f}%"
    print(f"  Чистый стоп (не дошёл до TP1)              "
          f"{len(cat_pure_stop):>3}    {pct(len(cat_pure_stop)):>5}     -1.000R (полный стоп)")
    print(f"  Дошёл до TP1, потом SL                     "
          f"{len(cat_tp1_only):>3}    {pct(len(cat_tp1_only)):>5}     +0.325R (50%×0.65R, остаток BE)")
    print(f"  Дошёл до TP2, потом SL                     "
          f"{len(cat_tp2_only):>3}    {pct(len(cat_tp2_only)):>5}     +0.575R (50%×0.65 + 25%×1.0)")

    # ── Подсчёт R для scaled exit на стопах ──────────────────────────────────
    r_pure  = -1.000
    r_tp1   = 0.50 * 0.65 + 0.50 * 0.0  # TP1 + BE на остатке
    r_tp2   = 0.50 * 0.65 + 0.25 * 1.0 + 0.25 * 0.0  # TP1 + TP2 + BE на остатке

    total_r_scaled = (
        len(cat_pure_stop) * r_pure
        + len(cat_tp1_only)  * r_tp1
        + len(cat_tp2_only)  * r_tp2
    )
    total_r_orig = N * r_pure  # все стопы = -1R в оригинале

    print(f"\n[СУММА R НА СТОПАХ — ОРИГИНАЛ vs SCALED]")
    print("─" * W)
    print(f"  Оригинал (все 93 стопа = -1R):     {total_r_orig:+.3f}R")
    print(f"  Scaled exit на тех же 93 стопах:   {total_r_scaled:+.3f}R")
    print(f"  ── Спасено благодаря BE-стопу:     {total_r_orig - total_r_scaled:+.3f}R "
          f"(= ~${(total_r_orig - total_r_scaled) * -18:.0f} при риске $18/трейд)")

    # ── Детализация по сколько баров ушло до TP1/SL ─────────────────────────
    tp1_then_sl = cat_tp1_only + cat_tp2_only
    if tp1_then_sl:
        print(f"\n[ДЛЯ {len(tp1_then_sl)} 'СПАСЁННЫХ' СТОПОВ] СКОЛЬКО БАРОВ ДО TP1, ПОТОМ ДО SL")
        print("─" * W)
        print(f"  {'Символ':<10} {'Дата':<16} {'Dir':<5} {'TP1@bar':>8} {'TP2@bar':>8} "
              f"{'SL@bar':>7}  {'PnL_orig':>9}")
        for t in sorted(tp1_then_sl, key=lambda x: x["tp1_bar"] or 99):
            tp2_str = str(t["tp2_bar"]) if t["tp2_bar"] is not None else "—"
            print(f"  {t['symbol']:<10} {str(t['entry_date'])[:16]:<16} "
                  f"{t['direction']:<5} {t['tp1_bar']:>8} {tp2_str:>8} "
                  f"{t['sl_bar']:>7}  ${t['net_pnl']:>+8.2f}")

    # ── Сводка ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ОТВЕТ НА ВОПРОС")
    print(f"{'═' * W}")
    print(f"  Из {N} стопов:")
    print(f"    {len(cat_pure_stop)} ({len(cat_pure_stop)/N*100:.0f}%) — пошли сразу к SL, "
          f"не коснувшись даже TP1")
    print(f"    {len(cat_tp1_only)} ({len(cat_tp1_only)/N*100:.0f}%) — дошли до TP1 (0.65R), "
          f"потом откатились и сняло SL")
    print(f"    {len(cat_tp2_only)} ({len(cat_tp2_only)/N*100:.0f}%) — дошли до TP2 (1.0R), "
          f"потом откатились и сняло SL")
    print(f"")
    saved = len(cat_tp1_only) + len(cat_tp2_only)
    print(f"  → В scaled exit-стратегии {saved} из {N} стопов ({saved/N*100:.0f}%) превратились")
    print(f"    бы из -1R в +0.325R (или +0.575R) благодаря BE-стопу после TP1.")
    print()


if __name__ == "__main__":
    main()

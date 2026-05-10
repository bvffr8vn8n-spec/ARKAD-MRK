"""
experiments/mae_analysis.py
Анализ MAE (Maximum Adverse Excursion) по всем 197 сделкам Jan-Apr 2026.

Для каждого трейда считаем максимальное движение ПРОТИВ позиции
от входа до выхода — насколько близко цена подходила к SL.

  mae_r = расстояние до SL в единицах R (1R = полный SL)
    0.0 = цена ни разу не двинулась против
    0.5 = ушла на половину пути к SL
    1.0 = коснулась SL (= стоп)
    1.0+ = пробой SL (gap)
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


def _load_1h(hist, new):
    h = pd.read_csv(hist, parse_dates=["date"]).set_index("date")
    n = pd.read_csv(new,  parse_dates=["date"]).set_index("date")
    df = pd.concat([h, n])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _hold_window(t, df_1h):
    """Окно баров СТРОГО ПОСЛЕ входа (вход = close сигнального бара)."""
    start = t["entry_date"] + pd.Timedelta(hours=1)
    return df_1h.loc[start:t["exit_date"]]


def _mae(t, df_1h):
    """Максимальное движение против позиции, в R-единицах.
    Считается только от бара СРАЗУ ПОСЛЕ входа (engine entry = close сигнального бара)."""
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)

    window = _hold_window(t, df_1h)
    if len(window) == 0 or sl_dist == 0:
        return 0.0

    if direction == 1:
        worst = window["low"].min()
        adverse = max(entry - worst, 0)
    else:
        worst = window["high"].max()
        adverse = max(worst - entry, 0)

    return adverse / sl_dist


def _mfe(t, df_1h):
    """Максимальное движение ПО позиции, в R-единицах (для контекста)."""
    entry = t["entry_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - t["stop_price"])

    window = _hold_window(t, df_1h)
    if len(window) == 0 or sl_dist == 0:
        return 0.0

    if direction == 1:
        best = window["high"].max()
        favourable = max(best - entry, 0)
    else:
        best = window["low"].min()
        favourable = max(entry - best, 0)

    return favourable / sl_dist  # в R, не в TP


def _run_asset(cfg):
    df = _load_1h(cfg["hist_1h"], cfg["new_1h"])
    f  = generate_features(df.copy())
    f  = add_labels(f)
    f  = add_regime_columns(f)
    f  = add_session_column(f)
    f  = f.dropna()

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
        t["mae_r"] = _mae(t, df)
        t["mfe_r"] = _mfe(t, df)
    return trades


def main():
    W = 78
    print("═" * W)
    print("  ARKAD MRK — MAE Analysis  (Jan 1 – Apr 26, 2026)")
    print("  Насколько близко цена подходила к SL за время трейда")
    print("═" * W)

    all_trades = []
    for cfg in ASSETS:
        all_trades.extend(_run_asset(cfg))

    N = len(all_trades)
    mae = np.array([t["mae_r"] for t in all_trades])
    mfe = np.array([t["mfe_r"] for t in all_trades])
    reasons = [t["exit_reason"] for t in all_trades]

    # ── Общая статистика MAE ────────────────────────────────────────────────
    print(f"\n[ВСЕ {N} СДЕЛОК] СТАТИСТИКА MAE")
    print("─" * W)
    print(f"  Среднее MAE      : {mae.mean():.3f}R   (=в среднем уходили на {mae.mean()*100:.0f}% к SL)")
    print(f"  Медиана MAE      : {np.median(mae):.3f}R")
    print(f"  Std MAE          : {mae.std():.3f}R")
    print(f"  Минимум          : {mae.min():.3f}R   (никуда не дёрнулась)")
    print(f"  Максимум         : {mae.max():.3f}R   (касание/пробой SL)")

    # ── Распределение по корзинам ───────────────────────────────────────────
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 99.0]
    labels = ["0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
              "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
              "≥ 1.00 (SL)"]
    counts, _ = np.histogram(mae, bins=bins)

    print(f"\n[ВСЕ {N}] РАСПРЕДЕЛЕНИЕ MAE ПО КОРЗИНАМ")
    print("─" * W)
    print(f"  {'MAE (в R)':<15}  {'Сделок':>7}  {'%':>6}  {'Гистограмма':<40}")
    for lab, c in zip(labels, counts):
        bar = "█" * int(c / max(counts) * 40)
        pct = c / N * 100
        print(f"  {lab:<15}  {c:>7}  {pct:>5.1f}%  {bar}")

    # ── Сегментация: победители vs стопы ───────────────────────────────────
    wins  = [t for t in all_trades if t["net_pnl"] > 0]
    losses_sl = [t for t in all_trades if t["exit_reason"] == "stop"]
    losses_time = [t for t in all_trades if t["exit_reason"] == "time" and t["net_pnl"] <= 0]

    win_mae = np.array([t["mae_r"] for t in wins])
    sl_mae  = np.array([t["mae_r"] for t in losses_sl])

    print(f"\n[СЕГМЕНТАЦИЯ] MAE ПО ИСХОДУ ТРЕЙДА")
    print("─" * W)
    print(f"  ┌─ ПОБЕДИТЕЛИ (net_pnl > 0): {len(wins)} сделок")
    print(f"  │    avg MAE : {win_mae.mean():.3f}R  ← перед тем как пойти в TP")
    print(f"  │    медиана : {np.median(win_mae):.3f}R")
    print(f"  │    max MAE : {win_mae.max():.3f}R")
    print(f"  │    >= 0.5R : {(win_mae >= 0.5).sum()} ({(win_mae >= 0.5).sum()/len(wins)*100:.1f}%)")
    print(f"  │    >= 0.8R : {(win_mae >= 0.8).sum()} ({(win_mae >= 0.8).sum()/len(wins)*100:.1f}%)")
    print(f"  │")
    print(f"  ├─ СТОПЫ (exit=stop): {len(losses_sl)} сделок")
    print(f"  │    avg MAE : {sl_mae.mean():.3f}R")
    print(f"  │    max MAE : {sl_mae.max():.3f}R   (пробой = gap stop)")
    print(f"  │")
    print(f"  └─ TIME EXIT в минус: {len(losses_time)} сделок")

    # ── Скоринг для BE-стопа ────────────────────────────────────────────────
    # Какой % победителей "пощупал" SL близко перед разворотом?
    print(f"\n[ВАЖНО ДЛЯ SCALED EXIT] СКОЛЬКО ПОБЕДИТЕЛЕЙ ЕДВА НЕ СНЕСЛО SL")
    print("─" * W)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for thr in thresholds:
        n = (win_mae >= thr).sum()
        print(f"  Победители с MAE ≥ {thr:.2f}R : {n:>3} из {len(wins)} ({n/len(wins)*100:.1f}%)")

    # ── Топ-10 самых "болезненных" победителей ───────────────────────────────
    print(f"\n[TOP-10 ПОБЕДИТЕЛЕЙ С НАИБОЛЕЕ ГЛУБОКИМ ПРОСЕДАНИЕМ]")
    print("─" * W)
    sorted_wins = sorted(wins, key=lambda t: t["mae_r"], reverse=True)[:10]
    print(f"  {'Символ':<10}  {'Дата':<16}  {'Dir':<5}  {'MAE_R':>7}  {'MFE_R':>7}  {'PnL$':>8}  {'Exit':<5}")
    for t in sorted_wins:
        print(f"  {t['symbol']:<10}  {str(t['entry_date'])[:16]:<16}  "
              f"{t['direction']:<5}  {t['mae_r']:>7.3f}  {t['mfe_r']:>7.3f}  "
              f"{t['net_pnl']:>+8.2f}  {t['exit_reason']:<5}")

    # ── Сводка ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  СВОДКА")
    print(f"{'═' * W}")
    print(f"  В среднем по всем 197 сделкам цена уходила на {mae.mean()*100:.0f}% пути к SL")
    print(f"  Медианная сделка прошла {np.median(mae)*100:.0f}% пути к SL")
    print(f"  {(win_mae < 0.65).sum()} из {len(wins)} победителей ({(win_mae < 0.65).sum()/len(wins)*100:.0f}%) "
          f"НЕ ДОХОДИЛИ до уровня TP1 (0.65R) на просадке")
    print(f"  → BE-стоп после TP1 безопасен в {(win_mae < 0.65).sum()/len(wins)*100:.0f}% случаев")
    print()


if __name__ == "__main__":
    main()

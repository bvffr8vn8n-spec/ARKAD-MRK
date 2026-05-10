"""
experiments/multi_tp_analysis.py
Тест стратегии частичного тейка (Scaled Exit):
  TP1 = 0.65R  → закрываем 50% позиции
  TP2 = 1.00R  → закрываем 25% позиции
  TP3 = 1.67R  → закрываем оставшиеся 25% (оригинальный TP)
  После срабатывания TP1 → стоп переносится в безубыток (BE = 0R)

Сравниваем с оригинальной стратегией (единый TP на 1.67R).
Данные: симуляция Jan 1 – Apr 26, 2026 (197 сделок).
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
from backtest.metrics import compute_metrics

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

# ── Параметры частичного тейка ──────────────────────────────────────────────
TP1_R     = 0.65   # первый тейк в единицах R
TP2_R     = 1.00   # второй тейк
TP3_R     = 1.67   # третий тейк (оригинальный TP)
TP1_SIZE  = 0.50   # доля позиции на TP1
TP2_SIZE  = 0.25   # доля позиции на TP2
TP3_SIZE  = 0.25   # доля позиции на TP3

# Пороги MFE%TP для определения достижения тейков
TP1_THRESH = TP1_R / TP3_R * 100   # 38.92%
TP2_THRESH = TP2_R / TP3_R * 100   # 59.88%

POSITION_SIZE_PCT = 0.10
EQUITY_START      = 10_000.0
COMMISSION_PCT    = 0.001
SLIPPAGE_PCT      = 0.0002


def _load_1h(hist_path, new_path):
    df_hist = pd.read_csv(hist_path, parse_dates=["date"]).set_index("date")
    df_new  = pd.read_csv(new_path,  parse_dates=["date"]).set_index("date")
    df = pd.concat([df_hist, df_new])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _mfe_stats(t, df_1h):
    entry, tp, sl = t["entry_price"], t["tp_price"], t["stop_price"]
    d  = 1 if t["direction"] == "long" else -1
    window = df_1h.loc[t["entry_date"]:t["exit_date"]]
    if len(window) == 0:
        return 0.0, 0.0
    if d == 1:
        best   = window["high"].max()
        mfe_p  = max(best - entry, 0)
        tp_d   = tp - entry
    else:
        best   = window["low"].min()
        mfe_p  = max(entry - best, 0)
        tp_d   = entry - tp
    mfe_pct_tp = min(mfe_p / tp_d * 100, 100.0) if tp_d > 0 else 0.0
    return mfe_p, mfe_pct_tp


def _scaled_exit_r(t, mfe_pct_tp, equity_at_entry):
    """
    Рассчитывает итоговый PnL в $ при стратегии частичного тейка.
    Логика (BE stop after TP1):
      - MFE%TP >= TP3 (100%) : TP1 + TP2 + TP3 все сработали
      - MFE%TP >= TP2 (59.9%): TP1 + TP2 сработали, остаток вышел по BE
      - MFE%TP >= TP1 (38.9%): TP1 сработал, остаток вышел по BE
      - MFE%TP <  TP1         : стоп-лосс по всей позиции (или time exit)
    """
    entry   = t["entry_price"]
    tp      = t["tp_price"]
    sl      = t["stop_price"]
    d       = 1 if t["direction"] == "long" else -1
    reason  = t["exit_reason"]

    # Рассчитываем 1R в долларах
    atr_dist = abs(entry - sl)          # = 1.5 × ATR
    one_r_pts = atr_dist / 1.5 * 1.67  # полный TP в пунктах
    pos_size  = (equity_at_entry * POSITION_SIZE_PCT) / entry  # кол-во монет
    one_r_usd = pos_size * atr_dist / 1.5 * 1.67  # TP в $, это и есть 1R (TP_R * ATR)

    # Уточнение: 1R = ATR * (TP_mult/SL_mult) * pos_size ... проще через entry/sl/tp
    # Используем: TP_R=1.67, SL_R=1.0 → 1R_USD = pos_size * (entry - sl) (= SL distance)
    one_r_usd = pos_size * atr_dist  # 1.0R в долларах

    comm = (equity_at_entry * POSITION_SIZE_PCT) * (COMMISSION_PCT + SLIPPAGE_PCT) * 2

    if mfe_pct_tp >= 99.9:
        # TP3 достигнут — все три тейка
        r_total = TP1_SIZE * TP1_R + TP2_SIZE * TP2_R + TP3_SIZE * TP3_R
    elif mfe_pct_tp >= TP2_THRESH:
        # TP2 достигнут, остаток по BE
        r_total = TP1_SIZE * TP1_R + TP2_SIZE * TP2_R + TP3_SIZE * 0.0
    elif mfe_pct_tp >= TP1_THRESH:
        # TP1 достигнут, остаток по BE
        r_total = TP1_SIZE * TP1_R + (TP2_SIZE + TP3_SIZE) * 0.0
    else:
        # Нет ни одного тейка
        if reason == "stop":
            r_total = -1.0  # полный стоп
        else:
            # time exit: используем фактическую цену выхода
            exit_p = t["exit_price"]
            r_total = (exit_p - entry) * d / atr_dist  # в R-единицах

    pnl = r_total * one_r_usd - comm
    return pnl, r_total


def _run_asset(cfg):
    sym = cfg["symbol"]
    df_all = _load_1h(cfg["hist_1h"], cfg["new_1h"])

    df_feat = generate_features(df_all.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    train = df_feat[df_feat.index < TEST_START]
    test  = df_feat[(df_feat.index >= TEST_START) & (df_feat.index <= TEST_END)]

    if len(train) < 200 or len(test) == 0:
        return []

    feat_cols = get_feature_columns(train)
    model     = fit_model(train, feat_cols)
    signals   = apply_signals(model, feat_cols, test.copy())

    df_15m    = load_5m_as_15m(cfg["new_5m"])
    annotated = annotate_signals_AB(signals, df_15m)
    annotated["signal"] = annotated["signal_15m_A"]

    trades, _ = run_backtest(annotated)

    equity = EQUITY_START
    enriched = []
    for t in trades:
        t["symbol"] = sym
        _, mfe_tp = _mfe_stats(t, df_all)
        t["mfe_pct_tp"] = mfe_tp
        pnl_scaled, r_scaled = _scaled_exit_r(t, mfe_tp, equity)
        t["pnl_scaled"] = pnl_scaled
        t["r_scaled"]   = r_scaled
        # оригинальный R
        atr_dist = abs(t["entry_price"] - t["stop_price"])
        pos = (equity * POSITION_SIZE_PCT) / t["entry_price"]
        t["r_original"] = t["net_pnl"] / (pos * atr_dist) if atr_dist > 0 else 0.0
        # обновляем equity (для обоих трекаем на базе оригинала для простоты)
        equity = max(equity + t["net_pnl"], 1.0)
        enriched.append(t)

    return enriched


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("  ARKAD MRK — Scaled Exit Analysis  (Jan 1 – Apr 26, 2026)")
    print(f"  TP1={TP1_R}R → {int(TP1_SIZE*100)}%  |  TP2={TP2_R}R → {int(TP2_SIZE*100)}%"
          f"  |  TP3={TP3_R}R → {int(TP3_SIZE*100)}%  |  BE stop after TP1")
    print("=" * 75)

    all_trades = []
    for cfg in ASSETS:
        print(f"  Запуск {cfg['symbol']}...", end=" ", flush=True)
        trades = _run_asset(cfg)
        all_trades.extend(trades)
        print(f"{len(trades)} сделок")

    all_trades.sort(key=lambda t: t.get("entry_date", pd.Timestamp.min))
    N = len(all_trades)

    # ── Классификация по достигнутым тейкам ─────────────────────────────────
    hits_tp3 = sum(1 for t in all_trades if t["mfe_pct_tp"] >= 99.9)
    hits_tp2 = sum(1 for t in all_trades if TP2_THRESH <= t["mfe_pct_tp"] < 99.9)
    hits_tp1 = sum(1 for t in all_trades if TP1_THRESH <= t["mfe_pct_tp"] < TP2_THRESH)
    hits_sl  = sum(1 for t in all_trades if t["mfe_pct_tp"] < TP1_THRESH and t["exit_reason"] == "stop")
    hits_time= sum(1 for t in all_trades if t["mfe_pct_tp"] < TP1_THRESH and t["exit_reason"] == "time")

    # ── Оригинальные метрики ─────────────────────────────────────────────────
    orig_pnl  = sum(t["net_pnl"] for t in all_trades)
    orig_wins = sum(1 for t in all_trades if t["net_pnl"] > 0)
    orig_gw   = sum(t["net_pnl"] for t in all_trades if t["net_pnl"] > 0)
    orig_gl   = abs(sum(t["net_pnl"] for t in all_trades if t["net_pnl"] <= 0))
    orig_pf   = orig_gw / orig_gl if orig_gl > 0 else float("inf")
    orig_wr   = orig_wins / N

    # ── Scaled метрики ───────────────────────────────────────────────────────
    sc_pnl    = sum(t["pnl_scaled"] for t in all_trades)
    sc_wins   = sum(1 for t in all_trades if t["pnl_scaled"] > 0)
    sc_gw     = sum(t["pnl_scaled"] for t in all_trades if t["pnl_scaled"] > 0)
    sc_gl     = abs(sum(t["pnl_scaled"] for t in all_trades if t["pnl_scaled"] <= 0))
    sc_pf     = sc_gw / sc_gl if sc_gl > 0 else float("inf")
    sc_wr     = sc_wins / N

    # ── Avg R ────────────────────────────────────────────────────────────────
    avg_r_orig = sum(t["r_original"] for t in all_trades) / N
    avg_r_sc   = sum(t["r_scaled"]   for t in all_trades) / N

    # ── Вывод ─────────────────────────────────────────────────────────────────
    SEP = "─" * 65

    print(f"\n{SEP}")
    print(f"  РАСПРЕДЕЛЕНИЕ ИСХОДОВ ПО УРОВНЯМ ЦЕНЫ")
    print(SEP)
    print(f"  Достигли TP3 (полный TP, 100% пути) : {hits_tp3:>3}  ({hits_tp3/N*100:.1f}%)")
    print(f"  Достигли TP2 (60–99% пути)          : {hits_tp2:>3}  ({hits_tp2/N*100:.1f}%)")
    print(f"  Достигли TP1 (39–59% пути)          : {hits_tp1:>3}  ({hits_tp1/N*100:.1f}%)")
    print(f"  Стоп без TP1 (< 39% пути)           : {hits_sl:>3}  ({hits_sl/N*100:.1f}%)")
    print(f"  Время без TP1 (< 39% пути)          : {hits_time:>3}  ({hits_time/N*100:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  ИТОГО                               : {N}")

    print(f"\n{SEP}")
    print(f"  СРАВНЕНИЕ: ОРИГИНАЛ vs SCALED EXIT")
    print(SEP)
    print(f"  {'Метрика':<28}  {'Оригинал':>12}  {'Scaled Exit':>12}  {'Дельта':>10}")
    print(f"  {'─'*62}")

    rows = [
        ("Сделок",          f"{N}",               f"{N}",               "—"),
        ("Win Rate",         f"{orig_wr*100:.1f}%",f"{sc_wr*100:.1f}%", f"{(sc_wr-orig_wr)*100:+.1f}pp"),
        ("Profit Factor",    f"{orig_pf:.4f}",     f"{sc_pf:.4f}",      f"{sc_pf-orig_pf:+.4f}"),
        ("Итоговый PnL $",   f"${orig_pnl:+.2f}",  f"${sc_pnl:+.2f}",   f"${sc_pnl-orig_pnl:+.2f}"),
        ("Avg R/сделка",     f"{avg_r_orig:+.4f}R",f"{avg_r_sc:+.4f}R", f"{avg_r_sc-avg_r_orig:+.4f}R"),
        ("Gross Wins $",     f"${orig_gw:+.2f}",   f"${sc_gw:+.2f}",    f"${sc_gw-orig_gw:+.2f}"),
        ("Gross Losses $",   f"-${orig_gl:.2f}",   f"-${sc_gl:.2f}",    f"${orig_gl-sc_gl:+.2f}"),
    ]
    for name, v1, v2, delta in rows:
        print(f"  {name:<28}  {v1:>12}  {v2:>12}  {delta:>10}")

    # ── R-профиль по группам ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  R-ПРОФИЛЬ ПО ГРУППАМ ИСХОДОВ")
    print(SEP)
    g_tp3  = [t for t in all_trades if t["mfe_pct_tp"] >= 99.9]
    g_tp2  = [t for t in all_trades if TP2_THRESH <= t["mfe_pct_tp"] < 99.9]
    g_tp1  = [t for t in all_trades if TP1_THRESH <= t["mfe_pct_tp"] < TP2_THRESH]
    g_sl   = [t for t in all_trades if t["mfe_pct_tp"] < TP1_THRESH and t["exit_reason"] == "stop"]
    g_time = [t for t in all_trades if t["mfe_pct_tp"] < TP1_THRESH and t["exit_reason"] == "time"]

    groups = [
        ("TP3 достигнут (50%+25%+25%)", g_tp3, 0.50*0.65+0.25*1.0+0.25*1.67),
        ("TP2 достигнут (50%+25%+BE)",  g_tp2, 0.50*0.65+0.25*1.0),
        ("TP1 достигнут (50%+BE+BE)",   g_tp1, 0.50*0.65),
        ("Стоп (нет TP1)",              g_sl,  -1.0),
        ("Время (нет TP1)",             g_time, None),
    ]
    print(f"  {'Группа':<35} {'N':>4}  {'R (scaled)':>11}  {'R (orig)':>10}")
    print(f"  {'─'*62}")
    for name, grp, r_fixed in groups:
        if not grp:
            continue
        r_orig_avg = sum(t["r_original"] for t in grp) / len(grp)
        r_sc_avg   = r_fixed if r_fixed is not None else sum(t["r_scaled"] for t in grp)/len(grp)
        print(f"  {name:<35} {len(grp):>4}  {r_sc_avg:>+10.4f}R  {r_orig_avg:>+9.4f}R")

    # ── Полная таблица сделок ─────────────────────────────────────────────────
    W = 130
    print(f"\n{'='*W}")
    print(f"  ПОЛНЫЙ СПИСОК СДЕЛОК — SCALED EXIT")
    print(f"{'='*W}")
    hdr = (f"{'#':>3}  {'Символ':<10}  {'Вход':^16}  {'Dir':^5}  "
           f"{'EntryP':>9}  {'Стоп':>9}  {'TP':>9}  "
           f"{'Выход':^16}  {'Рез':^4}  {'MFE%TP':>7}  "
           f"{'Уровень':^10}  {'PnL_orig':>9}  {'PnL_scaled':>10}  {'R_sc':>7}")
    print(hdr)
    print("-" * W)

    # Что было достигнуто
    def tp_level(mfe):
        if mfe >= 99.9:         return "TP1+TP2+TP3"
        elif mfe >= TP2_THRESH: return "TP1+TP2+BE "
        elif mfe >= TP1_THRESH: return "TP1+BE     "
        else:                   return "—          "

    sc_total_pnl = 0.0
    for i, t in enumerate(all_trades, 1):
        ed   = str(t["entry_date"])[:16]
        xd   = str(t["exit_date"])[:16]
        dirn = "LONG" if t["direction"] == "long" else "SHRT"
        mfe  = t["mfe_pct_tp"]
        lvl  = tp_level(mfe)
        orig = t["net_pnl"]
        sc   = t["pnl_scaled"]
        sc_total_pnl += sc
        r_sc = t["r_scaled"]
        print(
            f"{i:>3}  {t['symbol']:<10}  {ed:^16}  {dirn:^5}  "
            f"{t['entry_price']:>9.4f}  {t['stop_price']:>9.4f}  {t['tp_price']:>9.4f}  "
            f"{xd:^16}  {t['exit_reason']:^4}  {mfe:>6.1f}%  "
            f"{lvl:^10}  {orig:>+9.2f}  {sc:>+10.2f}  {r_sc:>+6.3f}R"
        )

    print("-" * W)
    print(f"\n  ИТОГО ОРИГИНАЛ: PnL=${orig_pnl:+.2f}  WR={orig_wr*100:.1f}%  PF={orig_pf:.3f}")
    print(f"  ИТОГО SCALED:  PnL=${sc_pnl:+.2f}  WR={sc_wr*100:.1f}%  PF={sc_pf:.3f}")

    # ── Вывод ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ВЫВОД")
    print(f"{'='*65}")

    be_wr_orig   = 1 / (1 + 1.67)   # 37.5%
    # Scaled: avg R on win vs avg R on loss
    sc_win_r  = sum(t["r_scaled"] for t in all_trades if t["pnl_scaled"] > 0)
    sc_loss_r = abs(sum(t["r_scaled"] for t in all_trades if t["pnl_scaled"] <= 0))
    avg_win_r = sc_win_r  / sc_wins  if sc_wins > 0 else 0
    avg_los_r = sc_loss_r / (N - sc_wins) if (N - sc_wins) > 0 else 0
    realized_rr_sc = avg_win_r / avg_los_r if avg_los_r > 0 else float("inf")
    be_wr_sc = avg_los_r / (avg_win_r + avg_los_r) if (avg_win_r + avg_los_r) > 0 else 0.5

    print(f"\n  Оригинал (единый TP на 1.67R):")
    print(f"    Теор. R:R = 1.67 | Breakeven WR = {be_wr_orig*100:.1f}%")
    print(f"    Фактический PF = {orig_pf:.3f} | WR = {orig_wr*100:.1f}% | Exp = ${orig_pnl/N:+.2f}/сделку")

    print(f"\n  Scaled Exit (TP1=0.65R/50%, TP2=1R/25%, TP3=1.67R/25%, BE после TP1):")
    print(f"    Реализованный avg win = +{avg_win_r:.4f}R | avg loss = -{avg_los_r:.4f}R")
    print(f"    Реализованный R:R = {realized_rr_sc:.3f} | Breakeven WR = {be_wr_sc*100:.1f}%")
    print(f"    WR = {sc_wr*100:.1f}% (выше BE на {(sc_wr - be_wr_sc)*100:+.1f}pp)")
    print(f"    Фактический PF = {sc_pf:.3f} | Exp = ${sc_pnl/N:+.2f}/сделку")
    print(f"    Delta PnL vs оригинал: ${sc_pnl - orig_pnl:+.2f}")

    print(f"\n  КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:")
    print(f"    1. TP1 (0.65R) достигается в {(hits_tp3+hits_tp2+hits_tp1)/N*100:.0f}% сделок")
    print(f"       → BE stop срабатывает в {(hits_tp3+hits_tp2+hits_tp1)/N*100:.0f}% случаев")
    print(f"    2. TP3 (полный TP) достигается в {hits_tp3/N*100:.0f}% сделок")
    print(f"       (у оригинальных winners = {orig_wins/N*100:.0f}%)")
    print(f"    3. WR scaled={sc_wr*100:.1f}% vs BE={be_wr_sc*100:.1f}% → "
          f"{'EDGE ЕСТЬ' if sc_wr > be_wr_sc else 'EDGE НЕТ'} при scaled exit")
    print(f"    4. Реализованный avg R при scaled: {avg_r_sc:+.4f}R vs оригинал {avg_r_orig:+.4f}R")

    if sc_pf > orig_pf:
        print(f"\n  РЕКОМЕНДАЦИЯ: Scaled Exit УЛУЧШАЕТ результат (+{sc_pf-orig_pf:.4f} PF)")
        print(f"    BE stop снижает размер проигрышей, сохраняя часть прибыли.")
    else:
        print(f"\n  РЕКОМЕНДАЦИЯ: Scaled Exit УХУДШАЕТ результат ({sc_pf-orig_pf:.4f} PF)")
        print(f"    Ранние частичные закрытия срезают большие выигрышные трейды.")
    print()


if __name__ == "__main__":
    main()

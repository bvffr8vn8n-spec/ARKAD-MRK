"""
experiments/compound_2025.py
Симуляция с РЕИНВЕСТОМ (compounding) на полном 2025 году.
Стартовый капитал $1000, размер позиции 1% / 2% / 5% / 10% от текущего equity.

Сделки берутся из tp1_sweep на 2025 (TP1=0.65R baseline) и сортируются
хронологически. Каждая сделка использует свежий equity на момент входа.
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

TP1_R = 0.65
TP2_R = 1.00
TP3_R = 1.67
SIZE_TP1 = 0.50
SIZE_TP2 = 0.25
SIZE_TP3 = 0.25

START_EQUITY = 1000.0
POSITION_PCTS = [0.01, 0.02, 0.05, 0.10]
COMMISSION = 0.001    # 0.1% таker
SLIPPAGE   = 0.0002   # 2 bps per side


def _load_1h(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace(t, df_1h):
    """Bar-by-bar симуляция scaled exit. Возвращает final_R."""
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return 0.0

    if direction == 1:
        tp1 = entry + TP1_R * sl_dist
        tp2 = entry + TP2_R * sl_dist
        tp3 = entry + TP3_R * sl_dist
        sl_lvl, be = entry - sl_dist, entry
    else:
        tp1 = entry - TP1_R * sl_dist
        tp2 = entry - TP2_R * sl_dist
        tp3 = entry - TP3_R * sl_dist
        sl_lvl, be = entry + sl_dist, entry

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]
    if len(window) == 0:
        return 0.0

    phase = "pre_tp1"
    locked_r = 0.0
    open_size = 1.0
    tp2_hit_flag = False

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
                locked_r += open_size * (-1.0)
                return locked_r
            if tp1_hit:
                locked_r += SIZE_TP1 * TP1_R
                open_size -= SIZE_TP1
                phase = "post_tp1"
                if direction == 1:
                    if hi >= tp2 and not tp2_hit_flag:
                        tp2_hit_flag = True
                        locked_r += SIZE_TP2 * TP2_R
                        open_size -= SIZE_TP2
                    if hi >= tp3:
                        locked_r += SIZE_TP3 * TP3_R
                        return locked_r
                else:
                    if lo <= tp2 and not tp2_hit_flag:
                        tp2_hit_flag = True
                        locked_r += SIZE_TP2 * TP2_R
                        open_size -= SIZE_TP2
                    if lo <= tp3:
                        locked_r += SIZE_TP3 * TP3_R
                        return locked_r
                continue

        else:
            if direction == 1:
                be_hit  = lo <= be
                tp2_hit = hi >= tp2 and not tp2_hit_flag
                tp3_hit = hi >= tp3
            else:
                be_hit  = hi >= be
                tp2_hit = lo <= tp2 and not tp2_hit_flag
                tp3_hit = lo <= tp3
            if be_hit:
                return locked_r
            if tp2_hit:
                tp2_hit_flag = True
                locked_r += SIZE_TP2 * TP2_R
                open_size -= SIZE_TP2
            if tp3_hit:
                locked_r += SIZE_TP3 * TP3_R
                return locked_r

    # Time exit
    if open_size > 0:
        last_close = float(window.iloc[-1]["close"])
        r_close = (last_close - entry) * direction / sl_dist
        locked_r += open_size * r_close
    return locked_r


def _run_asset(cfg):
    df = _load_1h(cfg["h1"])
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

    df15 = load_5m_as_15m(cfg["m5"])
    ann  = annotate_signals_AB(sig, df15)
    ann["signal"] = ann["signal_15m_A"]

    trades, _ = run_backtest(ann)
    out = []
    for t in trades:
        t["symbol"] = cfg["symbol"]
        t["final_R"] = _trace(t, df)
        # 1R distance в % (для пересчёта в $ при разном position size)
        sl_dist = abs(t["entry_price"] - t["stop_price"])
        t["sl_pct"] = sl_dist / t["entry_price"] if t["entry_price"] > 0 else 0
        out.append(t)
    return out


def _simulate_compound(trades, position_pct, start_equity=1000.0):
    """
    Прогон с реинвестом. Каждая сделка использует current_equity × position_pct.
    Возвращает: финальный equity, equity_curve, max_dd, peak.
    """
    equity = start_equity
    peak   = start_equity
    max_dd = 0.0
    curve  = []
    blown  = False

    for t in trades:
        pos_value = equity * position_pct
        units     = pos_value / t["entry_price"]
        sl_dist   = abs(t["entry_price"] - t["stop_price"])
        one_r_usd = units * sl_dist  # 1R в долларах для этой сделки

        gross_pnl = t["final_R"] * one_r_usd
        # Комиссия + проскальзывание (грубо учтены через 0.12% от позиции)
        cost = pos_value * (COMMISSION * 2 + SLIPPAGE * 2)
        net_pnl = gross_pnl - cost
        equity += net_pnl

        if equity <= 0:
            blown = True
            equity = 0
            curve.append({"date": t["entry_date"], "equity": equity, "trade_pnl": net_pnl})
            break

        peak = max(peak, equity)
        dd   = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        curve.append({"date": t["entry_date"], "equity": equity, "trade_pnl": net_pnl})

    return {
        "final_equity": equity,
        "peak_equity":  peak,
        "max_dd":       max_dd,
        "curve":        curve,
        "blown":        blown,
        "n_trades":     len(curve),
    }


def main():
    W = 88
    print("═" * W)
    print("  ARKAD MRK — COMPOUND SIMULATION на 2025 году")
    print(f"  Старт: $1000  |  TP1=0.65R baseline  |  Test: {TEST_START.date()} → {TEST_END.date()}")
    print("═" * W)

    print(f"\n  Прогоняем {len(ASSETS)} актива...")
    all_trades = []
    for cfg in ASSETS:
        trs = _run_asset(cfg)
        all_trades.extend(trs)
        print(f"    {cfg['symbol']:<10}  {len(trs)} сделок")
    print(f"  ─────────────────")
    print(f"  Всего: {len(all_trades)} сделок")

    # Сортируем по времени входа — ВАЖНО для compound симуляции
    all_trades.sort(key=lambda t: t["entry_date"])

    # ── Симуляции для каждого position size ────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  РЕЗУЛЬТАТЫ С РЕИНВЕСТОМ — 1228 сделок за 12 месяцев")
    print(f"{'═' * W}")
    print(f"  {'Pos%':>5}  {'Final $':>11}  {'PnL $':>10}  {'PnL %':>8}  "
          f"{'Peak $':>10}  {'Max DD':>7}  {'CAGR':>7}  {'Blown':>7}")
    print(f"  {'─' * (W-2)}")

    results = {}
    for pct in POSITION_PCTS:
        r = _simulate_compound(all_trades, pct, start_equity=START_EQUITY)
        results[pct] = r
        pnl    = r["final_equity"] - START_EQUITY
        pnl_p  = pnl / START_EQUITY * 100
        cagr   = (r["final_equity"] / START_EQUITY) - 1   # 1 year ≈ CAGR
        print(f"  {pct*100:>4.0f}%  ${r['final_equity']:>9,.2f}  "
              f"${pnl:>+8,.2f}  {pnl_p:>+7.1f}%  "
              f"${r['peak_equity']:>8,.2f}  {r['max_dd']*100:>5.1f}%  "
              f"{cagr*100:>+6.1f}%  {'YES' if r['blown'] else 'no':>7}")

    # ── Сравнение с симуляцией БЕЗ реинвеста (фиксированный сайз) ──────────
    print(f"\n{'═' * W}")
    print(f"  СРАВНЕНИЕ: РЕИНВЕСТ vs ФИКСИРОВАННЫЙ САЙЗ")
    print(f"{'═' * W}")
    print(f"  {'Pos%':>5}  {'Compound $':>13}  {'Fixed $':>13}  "
          f"{'Δ vs Fixed':>13}  {'Множитель':>11}")
    print(f"  {'─' * (W-2)}")
    for pct in POSITION_PCTS:
        # Fixed-size: каждая сделка использует $1000 × pct, не зависит от equity
        fixed_pnl = 0.0
        for t in all_trades:
            pos_value = START_EQUITY * pct
            units     = pos_value / t["entry_price"]
            sl_dist   = abs(t["entry_price"] - t["stop_price"])
            one_r_usd = units * sl_dist
            gross = t["final_R"] * one_r_usd
            cost  = pos_value * (COMMISSION * 2 + SLIPPAGE * 2)
            fixed_pnl += gross - cost
        fixed_final = START_EQUITY + fixed_pnl
        comp_final  = results[pct]["final_equity"]
        delta = comp_final - fixed_final
        mult  = comp_final / fixed_final if fixed_final > 0 else 0
        print(f"  {pct*100:>4.0f}%  ${comp_final:>11,.2f}  ${fixed_final:>11,.2f}  "
              f"${delta:>+11,.2f}  {mult:>10.2f}×")

    # ── Кривая equity для 1% и 2% ──────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ХРОНИКА — equity по месяцам (1% и 2%)")
    print(f"{'═' * W}")
    months = pd.date_range("2025-01-01", "2025-12-31", freq="MS").tolist()
    months.append(pd.Timestamp("2025-12-31 23:59:59"))

    print(f"  {'Месяц':<10}  {'1% equity':>12}  {'1% месPnL':>11}  {'1% макс DD%':>11}  "
          f"|  {'2% equity':>12}  {'2% месPnL':>11}")
    print(f"  {'─' * (W-2)}")

    def _equity_at(curve, ts):
        if not curve: return START_EQUITY
        prev_eq = START_EQUITY
        for c in curve:
            if c["date"] > ts:
                return prev_eq
            prev_eq = c["equity"]
        return prev_eq

    for i in range(len(months) - 1):
        m_start = months[i]
        m_end   = months[i + 1] - pd.Timedelta(seconds=1)
        for pct, lbl in [(0.01, "1%"), (0.02, "2%")]:
            curve = results[pct]["curve"]
            eq_end = _equity_at(curve, m_end)
            if i == 0:
                eq_start = START_EQUITY
            else:
                eq_start = _equity_at(curve, months[i] - pd.Timedelta(seconds=1))
            month_pnl = eq_end - eq_start
            if pct == 0.01:
                # макс DD внутри месяца
                month_curve = [c for c in curve if m_start <= c["date"] <= m_end]
                if month_curve:
                    peak_in_m = max((c["equity"] for c in month_curve), default=eq_end)
                    trough_in_m = min((c["equity"] for c in month_curve), default=eq_end)
                    dd_in_m = (peak_in_m - trough_in_m) / peak_in_m * 100 if peak_in_m > 0 else 0
                else:
                    dd_in_m = 0
                row1_eq = eq_end; row1_pnl = month_pnl; row1_dd = dd_in_m
            else:
                row2_eq = eq_end; row2_pnl = month_pnl
        print(f"  {m_start.strftime('%Y-%m'):<10}  ${row1_eq:>10,.2f}  ${row1_pnl:>+9,.2f}  "
              f"{row1_dd:>10.1f}%  |  ${row2_eq:>10,.2f}  ${row2_pnl:>+9,.2f}")

    # ── Финальная сводка ───────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ИТОГИ — $1000 старт, реинвест за 2025 год")
    print(f"{'═' * W}")
    for pct in [0.01, 0.02]:
        r = results[pct]
        pnl    = r["final_equity"] - START_EQUITY
        pnl_p  = pnl / START_EQUITY * 100
        peak_p = (r["peak_equity"] - START_EQUITY) / START_EQUITY * 100
        print(f"\n  Сайз {pct*100:.0f}% позиции:")
        print(f"    Финальный баланс       : ${r['final_equity']:,.2f}")
        print(f"    Чистая прибыль         : ${pnl:+,.2f} ({pnl_p:+.1f}%)")
        print(f"    Пиковый баланс         : ${r['peak_equity']:,.2f} ({peak_p:+.1f}%)")
        print(f"    Максимальная просадка  : {r['max_dd']*100:.2f}%")
        print(f"    Среднемесячная доходн. : {pnl_p/12:.2f}% / месяц")

    print()


if __name__ == "__main__":
    main()

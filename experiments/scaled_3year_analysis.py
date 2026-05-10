"""
experiments/scaled_3year_analysis.py
Полный анализ системы с scaled exit (TP1=0.65R/50%, TP2=1.0R/25%,
TP3=1.67R/25%, BE стоп после TP1) на 3 годах: 2023, 2024, 2025.

Для каждого года модель тренируется на ВСЕЙ истории до начала тестового
периода (walk-forward анчоред expanding-window).

Выдаёт:
  - Per-year summary (WR, PF, R-stats)
  - Per-asset × per-year breakdown
  - Compound simulation $1000 / 2% на каждом годе и комбинированно
  - Распределение исходов (TP3/TP2_BE/TP1_BE/Pure stop)
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

YEARS = [2023, 2024, 2025]

ASSETS = [
    {"symbol": "AVAXUSDT", "h1": "data/AVAXUSDT_1h_4y.csv", "m5": "data/AVAXUSDT_5m_4y.csv"},
    {"symbol": "ADAUSDT",  "h1": "data/ADAUSDT_1h_4y.csv",  "m5": "data/ADAUSDT_5m_4y.csv"},
    {"symbol": "SOLUSDT",  "h1": "data/SOLUSDT_1h_4y.csv",  "m5": "data/SOLUSDT_5m_4y.csv"},
    {"symbol": "XRPUSDT",  "h1": "data/XRPUSDT_1h_4y.csv",  "m5": "data/XRPUSDT_5m_4y.csv"},
]

TP1_R, TP2_R, TP3_R = 0.65, 1.00, 1.67
SIZE_TP1, SIZE_TP2, SIZE_TP3 = 0.50, 0.25, 0.25

START_EQUITY = 1000.0
POS_PCT      = 0.02
COMMISSION   = 0.001
SLIPPAGE     = 0.0002


def _load_1h(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace_scaled(t, df_1h):
    """Bar-by-bar симуляция scaled exit. Возвращает category и final_R."""
    entry = t["entry_price"]
    sl    = t["stop_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return {"R": 0, "outcome": "no_atr"}

    if direction == 1:
        tp1, tp2, tp3 = entry + TP1_R*sl_dist, entry + TP2_R*sl_dist, entry + TP3_R*sl_dist
        sl_lvl, be    = entry - sl_dist, entry
    else:
        tp1, tp2, tp3 = entry - TP1_R*sl_dist, entry - TP2_R*sl_dist, entry - TP3_R*sl_dist
        sl_lvl, be    = entry + sl_dist, entry

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]
    if len(window) == 0:
        return {"R": 0, "outcome": "no_window"}

    phase = "pre_tp1"
    locked_r = 0.0
    open_size = 1.0
    tp1_hit = tp2_hit = tp3_hit = be_hit = sl_hit_flag = False
    closed = False  # позиция полностью закрыта (любым событием)

    for i in range(len(window)):
        b = window.iloc[i]
        hi, lo = float(b["high"]), float(b["low"])
        if phase == "pre_tp1":
            sl_h = (lo <= sl_lvl) if direction == 1 else (hi >= sl_lvl)
            tp1_h = (hi >= tp1)   if direction == 1 else (lo <= tp1)
            if sl_h:
                sl_hit_flag = True
                locked_r += open_size * (-1.0)
                open_size = 0.0
                closed = True
                break
            if tp1_h:
                tp1_hit = True
                locked_r += SIZE_TP1 * TP1_R
                open_size -= SIZE_TP1
                phase = "post_tp1"
                tp2_h = (hi >= tp2) if direction == 1 else (lo <= tp2)
                tp3_h = (hi >= tp3) if direction == 1 else (lo <= tp3)
                if tp2_h and not tp2_hit:
                    tp2_hit = True
                    locked_r += SIZE_TP2 * TP2_R
                    open_size -= SIZE_TP2
                if tp3_h:
                    tp3_hit = True
                    locked_r += SIZE_TP3 * TP3_R
                    open_size = 0.0
                    closed = True
                    break
                continue
        else:
            be_h  = (lo <= be) if direction == 1 else (hi >= be)
            tp2_h = (hi >= tp2) if direction == 1 else (lo <= tp2)
            tp3_h = (hi >= tp3) if direction == 1 else (lo <= tp3)
            if be_h:
                be_hit = True
                # остаток выходит по 0R (BE = entry)
                open_size = 0.0
                closed = True
                break
            if tp2_h and not tp2_hit:
                tp2_hit = True
                locked_r += SIZE_TP2 * TP2_R
                open_size -= SIZE_TP2
            if tp3_h:
                tp3_hit = True
                locked_r += SIZE_TP3 * TP3_R
                open_size = 0.0
                closed = True
                break

    # Time exit на остатке только если позиция НЕ была явно закрыта
    if not closed and open_size > 0:
        last_close = float(window.iloc[-1]["close"])
        r_close = (last_close - entry) * direction / sl_dist
        # Защита: на time-exit без касания SL R не может быть < -1
        r_close = max(r_close, -1.0)
        locked_r += open_size * r_close

    if sl_hit_flag:    outcome = "pure_stop"
    elif tp3_hit:      outcome = "tp3"
    elif tp2_hit and be_hit:   outcome = "tp2_be"
    elif tp2_hit:      outcome = "tp2_time"
    elif tp1_hit and be_hit:   outcome = "tp1_be"
    elif tp1_hit:      outcome = "tp1_time"
    else:              outcome = "time_no_tp1"

    return {"R": locked_r, "outcome": outcome}


def _run(cfg, year):
    """Вернёт сделки за один год для одного актива."""
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end   = pd.Timestamp(f"{year}-12-31 23:59:59")

    df = _load_1h(cfg["h1"])
    f = generate_features(df.copy())
    f = add_labels(f)
    f = add_regime_columns(f)
    f = add_session_column(f)
    f = f.dropna()

    train = f[f.index < test_start]
    test  = f[(f.index >= test_start) & (f.index <= test_end)]
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
        t["year"]   = year
        info = _trace_scaled(t, df)
        t["R"]       = info["R"]
        t["outcome"] = info["outcome"]
        out.append(t)
    return out


def _stats(trades):
    rs = np.array([t["R"] for t in trades])
    N = len(rs)
    if N == 0:
        return {"N": 0, "wr": 0, "avg_r": 0, "sum_r": 0, "pf": 0, "median": 0,
                "max_r": 0, "min_r": 0}
    wins = rs > 0
    losses = rs < 0
    pf = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")
    return {
        "N": N,
        "wr":     wins.sum() / N,
        "avg_r":  rs.mean(),
        "sum_r":  rs.sum(),
        "pf":     pf,
        "median": np.median(rs),
        "max_r":  rs.max(),
        "min_r":  rs.min(),
    }


def _outcome_breakdown(trades):
    counts = {}
    for t in trades:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    return counts


def _simulate_compound(trades, position_pct, start_equity=1000.0):
    """Compound simulation. Trades sorted by entry_date."""
    trades = sorted(trades, key=lambda t: t["entry_date"])
    equity = start_equity
    peak   = start_equity
    max_dd = 0.0
    monthly = {}
    for t in trades:
        pos_value = equity * position_pct
        units = pos_value / t["entry_price"]
        sl_dist = abs(t["entry_price"] - t["stop_price"])
        one_r_usd = units * sl_dist
        gross = t["R"] * one_r_usd
        cost  = pos_value * (COMMISSION * 2 + SLIPPAGE * 2)
        net = gross - cost
        equity += net
        if equity <= 0:
            equity = 0
            break
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

        m_key = pd.Timestamp(t["entry_date"]).strftime("%Y-%m")
        monthly[m_key] = monthly.get(m_key, 0) + net

    return {
        "final_equity": equity,
        "peak_equity":  peak,
        "max_dd":       max_dd,
        "monthly":      monthly,
        "n":            len(trades),
    }


def main():
    W = 96
    print("═" * W)
    print("  ARKAD MRK — 3-YEAR SCALED EXIT ANALYSIS")
    print("  Test years: 2023, 2024, 2025  |  TP1=0.65R/50%, TP2=1.0R/25%, TP3=1.67R/25%, BE")
    print("═" * W)

    # Прогон всех годов и активов
    print(f"\n  Прогоняем {len(ASSETS)} актива × {len(YEARS)} лет...")
    all_by_year   = {y: [] for y in YEARS}
    by_asset_year = {}
    for cfg in ASSETS:
        for y in YEARS:
            trs = _run(cfg, y)
            all_by_year[y].extend(trs)
            by_asset_year[(cfg["symbol"], y)] = trs
            print(f"    {cfg['symbol']:<10}  {y}: {len(trs):>4} сделок")

    # ── Per-year summary ───────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ИТОГ ПО КАЖДОМУ ГОДУ — все 4 актива объединены")
    print(f"{'═' * W}")
    print(f"  {'Год':>5}  {'N':>5}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>10}  "
          f"{'PF':>7}  {'Median':>8}  {'Best R':>9}  {'Worst R':>9}")
    print(f"  {'─' * (W-2)}")

    yearly_stats = {}
    for y in YEARS:
        s = _stats(all_by_year[y])
        yearly_stats[y] = s
        print(f"  {y:>5}  {s['N']:>5}  {s['wr']*100:>6.1f}%  "
              f"{s['avg_r']:>+8.4f}R  {s['sum_r']:>+9.2f}R  "
              f"{s['pf']:>7.3f}  {s['median']:>+7.4f}R  "
              f"{s['max_r']:>+8.4f}R  {s['min_r']:>+8.4f}R")

    # Комбинированно
    all_trades = sum(all_by_year.values(), [])
    sc = _stats(all_trades)
    print(f"  {'─' * (W-2)}")
    print(f"  {'TOTAL':>5}  {sc['N']:>5}  {sc['wr']*100:>6.1f}%  "
          f"{sc['avg_r']:>+8.4f}R  {sc['sum_r']:>+9.2f}R  "
          f"{sc['pf']:>7.3f}  {sc['median']:>+7.4f}R  "
          f"{sc['max_r']:>+8.4f}R  {sc['min_r']:>+8.4f}R")

    # ── Per-asset × per-year breakdown ──────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  PER-ASSET × PER-YEAR — Sum R")
    print(f"{'═' * W}")
    print(f"  {'Symbol':<12}  ", end="")
    for y in YEARS:
        print(f"{y:>15}", end="")
    print(f"  {'TOTAL':>15}")
    print(f"  {'─' * (W-2)}")
    asset_totals = {}
    for cfg in ASSETS:
        sym = cfg["symbol"]
        print(f"  {sym:<12}  ", end="")
        total_r = 0.0
        for y in YEARS:
            s = _stats(by_asset_year[(sym, y)])
            n = s["N"]
            print(f"{s['sum_r']:>+9.2f}R ({n:>3})", end="")
            total_r += s["sum_r"]
        asset_totals[sym] = total_r
        print(f"  {total_r:>+13.2f}R")

    # ── Распределение исходов ────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  РАСПРЕДЕЛЕНИЕ ИСХОДОВ по годам")
    print(f"{'═' * W}")
    cats = ["tp3", "tp2_be", "tp2_time", "tp1_be", "tp1_time", "pure_stop", "time_no_tp1"]
    cat_lab = {
        "tp3":         "TP3 full",
        "tp2_be":      "TP2→BE",
        "tp2_time":    "TP2→Time",
        "tp1_be":      "TP1→BE",
        "tp1_time":    "TP1→Time",
        "pure_stop":   "Stop",
        "time_no_tp1": "Time(noTP1)",
    }
    print(f"  {'Год':<6}", end="")
    for c in cats:
        print(f"{cat_lab[c]:>12}", end="")
    print(f"{'Total':>10}")
    print(f"  {'─' * (W-2)}")
    for y in YEARS:
        bd = _outcome_breakdown(all_by_year[y])
        print(f"  {y:<6}", end="")
        total = 0
        for c in cats:
            n = bd.get(c, 0)
            print(f"{n:>12}", end="")
            total += n
        print(f"{total:>10}")
    bd_all = _outcome_breakdown(all_trades)
    print(f"  {'─' * (W-2)}")
    print(f"  {'TOTAL':<6}", end="")
    for c in cats:
        n = bd_all.get(c, 0)
        print(f"{n:>12}", end="")
    print(f"{len(all_trades):>10}")

    # ── Compound simulation ────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  COMPOUND SIMULATION  (старт ${START_EQUITY:,.0f}, сайз {POS_PCT*100:.0f}%, реинвест)")
    print(f"{'═' * W}")
    print(f"  {'Сценарий':<35}  {'Старт':>10}  {'Финал':>10}  {'PnL %':>8}  {'Max DD':>8}")
    print(f"  {'─' * (W-2)}")

    for y in YEARS:
        sim = _simulate_compound(all_by_year[y], POS_PCT, START_EQUITY)
        pnl_p = (sim["final_equity"] - START_EQUITY) / START_EQUITY * 100
        print(f"  {f'{y} — старт фикс. $1000':<35}  ${START_EQUITY:>8,.2f}  "
              f"${sim['final_equity']:>8,.2f}  {pnl_p:>+7.2f}%  "
              f"{sim['max_dd']*100:>7.2f}%")

    # 3-year combined: реинвестим непрерывно через 2023→2024→2025
    sim_combined = _simulate_compound(all_trades, POS_PCT, START_EQUITY)
    pnl_p = (sim_combined["final_equity"] - START_EQUITY) / START_EQUITY * 100
    print(f"  {'─' * (W-2)}")
    print(f"  {'2023→2024→2025 КОМБО (compound)':<35}  ${START_EQUITY:>8,.2f}  "
          f"${sim_combined['final_equity']:>8,.2f}  {pnl_p:>+7.2f}%  "
          f"{sim_combined['max_dd']*100:>7.2f}%")

    # CAGR
    final = sim_combined["final_equity"]
    cagr = (final / START_EQUITY) ** (1/3) - 1
    print(f"  CAGR (3 года): {cagr*100:+.2f}% годовых")

    # ── Помесячная статистика 2023-2025 (combined) ───────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ПОМЕСЯЧНАЯ ДИНАМИКА (2% сайз, реинвест) — combined 3-year run")
    print(f"{'═' * W}")
    monthly = sim_combined["monthly"]
    keys = sorted(monthly.keys())
    print(f"  {'Месяц':<10}  {'PnL':>9}  {'Cum PnL':>10}  {'Bar':<35}")
    cum = 0
    for k in keys:
        v = monthly[k]
        cum += v
        bar_pos = "█" * int(max(v, 0) * 3)
        bar_neg = "▓" * int(abs(min(v, 0)) * 3)
        bar = bar_neg + bar_pos
        print(f"  {k:<10}  ${v:>+7.2f}  ${cum:>+8.2f}  {bar:<35}")

    # ── Доходность при разных капиталах ─────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  МАСШТАБИРОВАНИЕ — какая прибыль при разных капиталах (3 года, 2% сайз)")
    print(f"{'═' * W}")
    print(f"  {'Капитал':>10}  {'Финал':>15}  {'Прибыль $':>12}  {'Прибыль %':>10}  "
          f"{'CAGR':>8}")
    print(f"  {'─' * (W-2)}")
    for cap in [500, 1000, 5000, 10000, 50000]:
        sim = _simulate_compound(all_trades, POS_PCT, cap)
        pnl = sim["final_equity"] - cap
        pnl_p = pnl / cap * 100
        cagr = (sim["final_equity"] / cap) ** (1/3) - 1
        print(f"  ${cap:>8,}  ${sim['final_equity']:>13,.2f}  "
              f"${pnl:>+10,.2f}  {pnl_p:>+8.2f}%  {cagr*100:>+7.2f}%")

    # ── Финальный summary ──────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ИТОГОВАЯ КАРТИНА")
    print(f"{'═' * W}")
    total_sum_r = sc["sum_r"]
    print(f"  Всего сделок (2023+2024+2025)    : {sc['N']}")
    print(f"  Win rate                          : {sc['wr']*100:.1f}%")
    print(f"  Profit Factor                     : {sc['pf']:.3f}")
    print(f"  Sum R за 3 года                   : {total_sum_r:+.2f}R")
    print(f"  Avg R per trade                   : {sc['avg_r']:+.4f}R")
    print(f"  Median R                          : {sc['median']:+.4f}R")
    print(f"  Compound (старт $1000, 2% сайз) : ${sim_combined['final_equity']:.2f}  "
          f"(+{(sim_combined['final_equity']-START_EQUITY)/START_EQUITY*100:.1f}%)")
    print(f"  CAGR                              : {cagr*100:+.2f}%")
    print(f"  Max DD                            : {sim_combined['max_dd']*100:.2f}%")
    print()


if __name__ == "__main__":
    main()

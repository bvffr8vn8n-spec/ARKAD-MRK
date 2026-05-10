"""
experiments/compound_compare_2025.py
Сравнение compound симуляций ORIGINAL vs SCALED EXIT за 2025 год.

ORIGINAL: один TP=1.67R, SL=1.0R, без BE-стопа
SCALED:   TP1=0.65R/50%, TP2=1.0R/25%, TP3=1.67R/25%, BE-стоп после TP1

Старт: $1000  |  Сайзы: 1% / 2% / 5% / 10%  |  Реинвест: ДА
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

TP1_R, TP2_R, TP3_R = 0.65, 1.00, 1.67
SIZE_TP1, SIZE_TP2, SIZE_TP3 = 0.50, 0.25, 0.25

START_EQUITY  = 1000.0
POSITION_PCTS = [0.01, 0.02, 0.05, 0.10]
COMMISSION = 0.001
SLIPPAGE   = 0.0002


def _load_1h(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _trace_original(t, df_1h):
    """Original: TP=1.67R, SL=-1R, time exit."""
    entry = t["entry_price"]
    sl    = t["stop_price"]
    tp    = t["tp_price"]
    direction = 1 if t["direction"] == "long" else -1
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return 0.0

    if direction == 1:
        sl_lvl, tp_lvl = entry - sl_dist, entry + TP3_R * sl_dist
    else:
        sl_lvl, tp_lvl = entry + sl_dist, entry - TP3_R * sl_dist

    start = t["entry_date"] + pd.Timedelta(hours=1)
    window = df_1h.loc[start:t["exit_date"]]
    if len(window) == 0:
        return 0.0

    for i in range(len(window)):
        b = window.iloc[i]
        hi, lo = float(b["high"]), float(b["low"])
        if direction == 1:
            sl_hit = lo <= sl_lvl
            tp_hit = hi >= tp_lvl
        else:
            sl_hit = hi >= sl_lvl
            tp_hit = lo <= tp_lvl
        if sl_hit:
            return -1.0
        if tp_hit:
            return TP3_R

    last_close = float(window.iloc[-1]["close"])
    return (last_close - entry) * direction / sl_dist


def _trace_scaled(t, df_1h):
    """Scaled exit: TP1+TP2+TP3 + BE стоп после TP1."""
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
                sl_hit = lo <= sl_lvl
                tp1_hit = hi >= tp1
            else:
                sl_hit = hi >= sl_lvl
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
                be_hit = lo <= be
                tp2_hit = hi >= tp2 and not tp2_hit_flag
                tp3_hit = hi >= tp3
            else:
                be_hit = hi >= be
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
        t["R_orig"]   = _trace_original(t, df)
        t["R_scaled"] = _trace_scaled(t, df)
        out.append(t)
    return out


def _simulate(trades, position_pct, r_key, start_equity=1000.0):
    equity = start_equity
    peak = start_equity
    max_dd = 0.0
    curve = []
    blown = False
    win_count = 0
    loss_count = 0

    for t in trades:
        pos_value = equity * position_pct
        units     = pos_value / t["entry_price"]
        sl_dist   = abs(t["entry_price"] - t["stop_price"])
        one_r_usd = units * sl_dist

        gross_pnl = t[r_key] * one_r_usd
        cost = pos_value * (COMMISSION * 2 + SLIPPAGE * 2)
        net_pnl = gross_pnl - cost
        equity += net_pnl

        if net_pnl > 0: win_count += 1
        elif net_pnl < 0: loss_count += 1

        if equity <= 0:
            blown = True
            equity = 0
            curve.append({"date": t["entry_date"], "equity": equity})
            break

        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
        curve.append({"date": t["entry_date"], "equity": equity})

    return {
        "final_equity": equity,
        "peak_equity":  peak,
        "max_dd":       max_dd,
        "curve":        curve,
        "blown":        blown,
        "wins":         win_count,
        "losses":       loss_count,
    }


def main():
    W = 96
    print("═" * W)
    print("  ARKAD MRK — ORIGINAL vs SCALED EXIT  (с реинвестом, $1000 старт, 2025 год)")
    print("═" * W)

    print(f"\n  Прогоняем {len(ASSETS)} актива...")
    all_trades = []
    for cfg in ASSETS:
        trs = _run_asset(cfg)
        all_trades.extend(trs)
        print(f"    {cfg['symbol']:<10}  {len(trs)} сделок")
    print(f"  Всего: {len(all_trades)} сделок")

    all_trades.sort(key=lambda t: t["entry_date"])

    # ── Главная таблица ──────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  СРАВНЕНИЕ — финальный equity при разных сайзах")
    print(f"{'═' * W}")
    print(f"  {'Сайз':>5}  | {'ORIGINAL':<32} | {'SCALED EXIT':<32} | {'ΔPnL':>10}")
    print(f"  {'─' * 5}--|--{'─' * 30}--|--{'─' * 30}--|--{'─' * 10}")
    print(f"  {' ':>5}  | {'Final $':>12} {'PnL %':>7} {'MaxDD':>7} | "
          f"{'Final $':>12} {'PnL %':>7} {'MaxDD':>7} | {' ':>10}")

    rows = []
    for pct in POSITION_PCTS:
        ro = _simulate(all_trades, pct, "R_orig",   START_EQUITY)
        rs = _simulate(all_trades, pct, "R_scaled", START_EQUITY)
        rows.append((pct, ro, rs))
        delta = rs["final_equity"] - ro["final_equity"]
        pct_o = (ro["final_equity"] - START_EQUITY) / START_EQUITY * 100
        pct_s = (rs["final_equity"] - START_EQUITY) / START_EQUITY * 100
        print(f"  {pct*100:>4.0f}%  | "
              f"${ro['final_equity']:>10,.2f} {pct_o:>+6.2f}% {ro['max_dd']*100:>6.2f}% | "
              f"${rs['final_equity']:>10,.2f} {pct_s:>+6.2f}% {rs['max_dd']*100:>6.2f}% | "
              f"${delta:>+8,.2f}")

    # ── Win rate / Loss rate сравнение ──────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  WIN RATE и LOSS RATE")
    print(f"{'═' * W}")
    pct = 0.02   # для статистики (одинакова для всех сайзов)
    ro = _simulate(all_trades, pct, "R_orig", START_EQUITY)
    rs = _simulate(all_trades, pct, "R_scaled", START_EQUITY)
    n = len(all_trades)
    print(f"  ORIGINAL:    {ro['wins']} побед / {ro['losses']} убытков  →  "
          f"WR = {ro['wins']/n*100:.1f}%")
    print(f"  SCALED EXIT: {rs['wins']} побед / {rs['losses']} убытков  →  "
          f"WR = {rs['wins']/n*100:.1f}%")

    # ── Динамика по месяцам — для 1% и 2% ───────────────────────────────────
    months = pd.date_range("2025-01-01", "2025-12-31", freq="MS").tolist()
    months.append(pd.Timestamp("2025-12-31 23:59:59"))

    def _equity_at(curve, ts):
        prev = START_EQUITY
        for c in curve:
            if c["date"] > ts: return prev
            prev = c["equity"]
        return prev

    for pct, lbl in [(0.01, "САЙЗ 1%"), (0.02, "САЙЗ 2%")]:
        ro = _simulate(all_trades, pct, "R_orig",   START_EQUITY)
        rs = _simulate(all_trades, pct, "R_scaled", START_EQUITY)

        print(f"\n{'═' * W}")
        print(f"  {lbl} — equity по месяцам")
        print(f"{'═' * W}")
        print(f"  {'Месяц':<10}  {'ORIG equity':>12}  {'ORIG месPnL':>12}  |  "
              f"{'SCALED equity':>14}  {'SCALED месPnL':>14}  |  {'Δ':>8}")
        print(f"  {'─' * (W-2)}")
        prev_o = prev_s = START_EQUITY
        for i in range(len(months) - 1):
            m_start = months[i]
            m_end = months[i + 1] - pd.Timedelta(seconds=1)
            eq_o = _equity_at(ro["curve"], m_end)
            eq_s = _equity_at(rs["curve"], m_end)
            dpnl_o = eq_o - prev_o
            dpnl_s = eq_s - prev_s
            d = eq_s - eq_o
            print(f"  {m_start.strftime('%Y-%m'):<10}  ${eq_o:>10,.2f}  ${dpnl_o:>+10,.2f}  |  "
                  f"${eq_s:>12,.2f}  ${dpnl_s:>+12,.2f}  |  ${d:>+6,.2f}")
            prev_o, prev_s = eq_o, eq_s

    # ── Итог ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ФИНАЛЬНЫЕ ЦИФРЫ — старт $1000, реинвест, весь 2025")
    print(f"{'═' * W}")
    for pct, ro, rs in rows:
        if pct not in [0.01, 0.02]: continue
        print(f"\n  Сайз {pct*100:.0f}%:")
        print(f"    ORIGINAL:     ${ro['final_equity']:,.2f}  "
              f"(+{(ro['final_equity']-START_EQUITY)/START_EQUITY*100:.2f}%, "
              f"max DD {ro['max_dd']*100:.2f}%)")
        print(f"    SCALED EXIT:  ${rs['final_equity']:,.2f}  "
              f"(+{(rs['final_equity']-START_EQUITY)/START_EQUITY*100:.2f}%, "
              f"max DD {rs['max_dd']*100:.2f}%)")
        print(f"    Дельта:       ${rs['final_equity']-ro['final_equity']:+.2f}  "
              f"({(rs['final_equity']/ro['final_equity']-1)*100:+.1f}% улучшение)")
    print()


if __name__ == "__main__":
    main()

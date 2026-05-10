"""
experiments/native_3year_compare.py
Полный 3-летний прогон через НАТИВНЫЙ engine: single vs scaled.

Test years: 2023, 2024, 2025  |  4 актива  |  Walk-forward (train on history before each year)

Метрики per-year и combined:
  trades, WR, Avg R, Sum R, PF, max DD (по equity curve $1000 базе)
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


def _load_1h(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _run(cfg, year, exit_mode):
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
        return [], pd.Series(dtype=float)

    cols = get_feature_columns(train)
    model = fit_model(train, cols)
    sig = apply_signals(model, cols, test.copy())

    df15 = load_5m_as_15m(cfg["m5"])
    ann  = annotate_signals_AB(sig, df15)
    ann["signal"] = ann["signal_15m_A"]

    trades, equity = run_backtest(ann, exit_mode=exit_mode)
    for t in trades:
        t["symbol"] = cfg["symbol"]
        t["year"]   = year
    return trades, equity


def _max_dd_from_curve(curve: pd.Series) -> float:
    """Max drawdown from equity series, in percentage."""
    if len(curve) == 0:
        return 0.0
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return abs(dd.min()) * 100


def _stats(trades):
    rs = np.array([t.get("R", 0) for t in trades])
    N = len(rs)
    if N == 0:
        return {"N": 0, "WR": 0, "AvgR": 0, "SumR": 0, "PF": 0, "MaxR": 0, "MinR": 0}
    wins = rs > 0
    losses = rs < 0
    pf = rs[wins].sum() / abs(rs[losses].sum()) if losses.any() else float("inf")
    return {
        "N":   N,
        "WR":  wins.sum() / N * 100,
        "AvgR": rs.mean(),
        "SumR": rs.sum(),
        "PF":  pf,
        "MaxR": rs.max(),
        "MinR": rs.min(),
    }


def _simulate_equity(trades, position_pct=0.10, start_equity=10_000.0,
                     commission=0.001, slip=0.0002):
    """
    Прогон equity при заданном sizing. Используем R-multiple каждой сделки
    + position_value × sl_pct для перевода в $.
    Возвращает equity curve и max DD.
    """
    trades = sorted(trades, key=lambda t: t["entry_date"])
    equity = start_equity
    peak = start_equity
    max_dd = 0.0
    curve = []
    for t in trades:
        pos_value = equity * position_pct
        sl_dist = abs(t["entry_price"] - t["stop_price"])
        one_r = (pos_value / t["entry_price"]) * sl_dist
        gross = t["R"] * one_r
        # round-trip cost (estimated; native engine already accounts for it in R approx)
        cost = pos_value * (commission * 2 + slip * 2)
        # However, the engine's R already accounts for commission/slip via net_pnl-based calculation,
        # so we shouldn't double-count. Use the engine's net_pnl scaling instead.
        # Estimate net_pnl_at_this_size = t["net_pnl"] × (pos_value / engine_pos_value)
        # engine_pos_value = config.INITIAL_CAPITAL × POSITION_SIZE_PCT = $1000
        # Scale linearly:
        engine_pos = 10_000.0 * 0.10   # $1000 in engine config
        scaled_pnl = t.get("net_pnl", 0) * (pos_value / engine_pos)
        equity += scaled_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        curve.append({"date": t["entry_date"], "equity": equity})
    return curve, max_dd * 100, equity


def main():
    W = 100
    print("═" * W)
    print("  ARKAD MRK — NATIVE 3-YEAR COMPARE  (single vs scaled)")
    print("  Test years: 2023, 2024, 2025  |  4 assets  |  Native engine")
    print("═" * W)

    all_trades = {"single": {y: [] for y in YEARS}, "scaled": {y: [] for y in YEARS}}

    for cfg in ASSETS:
        for y in YEARS:
            for mode in ["single", "scaled"]:
                trs, _ = _run(cfg, y, mode)
                all_trades[mode][y].extend(trs)
            n_s = len(all_trades["single"][y]) - sum(len(all_trades["single"][yy]) for yy in YEARS if yy < y)
            # simplified status print
        print(f"  {cfg['symbol']:<10}  done")

    # ── Per-year + combined stats ──────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  PER-YEAR METRICS")
    print(f"{'═' * W}")
    print(f"  {'Year':<6}  {'Mode':<8}  {'N':>5}  {'WR':>7}  {'Avg R':>9}  "
          f"{'Sum R':>10}  {'PF':>7}  {'Max DD %':>9}  {'Best R':>8}  {'Worst R':>9}")
    print(f"  {'─' * (W-2)}")

    summary = {"single": {}, "scaled": {}}
    for y in YEARS:
        for mode in ["single", "scaled"]:
            trs = all_trades[mode][y]
            s = _stats(trs)
            curve, dd, _ = _simulate_equity(trs, position_pct=0.10, start_equity=10_000.0)
            summary[mode][y] = {**s, "MaxDD": dd}
            print(f"  {y:<6}  {mode:<8}  {s['N']:>5}  {s['WR']:>6.1f}%  "
                  f"{s['AvgR']:>+8.4f}R  {s['SumR']:>+9.2f}R  "
                  f"{s['PF']:>7.3f}  {dd:>8.2f}%  "
                  f"{s['MaxR']:>+7.3f}R  {s['MinR']:>+8.3f}R")

    # ── Combined 3-year ─────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  COMBINED 3-YEAR (2023 + 2024 + 2025)")
    print(f"{'═' * W}")
    print(f"  {'Mode':<8}  {'N':>5}  {'WR':>7}  {'Avg R':>9}  {'Sum R':>10}  "
          f"{'PF':>7}  {'Max DD %':>9}")
    print(f"  {'─' * (W-2)}")

    for mode in ["single", "scaled"]:
        combined = sum(all_trades[mode].values(), [])
        s = _stats(combined)
        curve, dd, final = _simulate_equity(combined, position_pct=0.10, start_equity=10_000.0)
        cagr = (final / 10_000.0) ** (1/3) - 1
        print(f"  {mode:<8}  {s['N']:>5}  {s['WR']:>6.1f}%  "
              f"{s['AvgR']:>+8.4f}R  {s['SumR']:>+9.2f}R  "
              f"{s['PF']:>7.3f}  {dd:>8.2f}%   "
              f"final=${final:,.0f}  CAGR={cagr*100:+.2f}%")

    # ── Дельты scaled - single ──────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  ДЕЛЬТЫ:  scaled − single")
    print(f"{'═' * W}")
    print(f"  {'Year':<6}  {'Δ trades':>10}  {'Δ WR':>8}  {'Δ Avg R':>10}  "
          f"{'Δ Sum R':>10}  {'Δ PF':>8}  {'Δ MaxDD':>9}")
    print(f"  {'─' * (W-2)}")
    for y in YEARS:
        d_n  = summary["scaled"][y]["N"] - summary["single"][y]["N"]
        d_wr = summary["scaled"][y]["WR"] - summary["single"][y]["WR"]
        d_ar = summary["scaled"][y]["AvgR"] - summary["single"][y]["AvgR"]
        d_sr = summary["scaled"][y]["SumR"] - summary["single"][y]["SumR"]
        d_pf = summary["scaled"][y]["PF"] - summary["single"][y]["PF"]
        d_dd = summary["scaled"][y]["MaxDD"] - summary["single"][y]["MaxDD"]
        print(f"  {y:<6}  {d_n:>+10d}  {d_wr:>+7.1f}pp  {d_ar:>+9.4f}R  "
              f"{d_sr:>+9.2f}R  {d_pf:>+7.3f}  {d_dd:>+8.2f}pp")

    # Combined deltas
    cs = _stats(sum(all_trades["scaled"].values(), []))
    co = _stats(sum(all_trades["single"].values(), []))
    _, dd_s, _ = _simulate_equity(sum(all_trades["scaled"].values(), []),
                                   position_pct=0.10, start_equity=10_000.0)
    _, dd_o, _ = _simulate_equity(sum(all_trades["single"].values(), []),
                                   position_pct=0.10, start_equity=10_000.0)
    print(f"  {'─' * (W-2)}")
    print(f"  {'TOTAL':<6}  {cs['N']-co['N']:>+10d}  {cs['WR']-co['WR']:>+7.1f}pp  "
          f"{cs['AvgR']-co['AvgR']:>+9.4f}R  {cs['SumR']-co['SumR']:>+9.2f}R  "
          f"{cs['PF']-co['PF']:>+7.3f}  {dd_s-dd_o:>+8.2f}pp")

    # ── Per-asset summary (combined 3 years, both modes) ────────────────────
    print(f"\n{'═' * W}")
    print(f"  PER-ASSET 3-YEAR (Sum R)")
    print(f"{'═' * W}")
    print(f"  {'Symbol':<12}  {'single N':>9}  {'single SumR':>12}  "
          f"{'scaled N':>9}  {'scaled SumR':>12}  {'ΔSumR':>9}")
    print(f"  {'─' * (W-2)}")
    for cfg in ASSETS:
        sym = cfg["symbol"]
        sng = [t for t in sum(all_trades["single"].values(), []) if t["symbol"] == sym]
        scl = [t for t in sum(all_trades["scaled"].values(), []) if t["symbol"] == sym]
        sng_sumR = sum(t["R"] for t in sng)
        scl_sumR = sum(t["R"] for t in scl)
        print(f"  {sym:<12}  {len(sng):>9}  {sng_sumR:>+11.2f}R  "
              f"{len(scl):>9}  {scl_sumR:>+11.2f}R  {scl_sumR-sng_sumR:>+8.2f}R")

    print(f"\n{'═' * W}\n")


if __name__ == "__main__":
    main()

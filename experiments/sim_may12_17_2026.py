"""
experiments/sim_may12_17_2026.py
— Simulate trades May 12 → May 17, 2026 using the scaled-exit engine.

Method
------
1. Load 4-year historical 1H data for each Tier-1 asset.
2. Append recent_1h (covers up to 2026-05-18) and merge.
3. Train CalibratedRF on ALL bars before 2026-05-12 (zero look-ahead).
4. Generate signals; apply 15m-AB execution layer (recent_5m → 15m).
5. Run engine.run_backtest with config.BACKTEST_EXIT_MODE = "scaled".
6. Print every trade whose ENTRY date falls in [2026-05-12, 2026-05-17].

Usage:
    python experiments/sim_may12_17_2026.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd

import config
from features.generator import generate_features, add_labels
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from features.execution_15m import load_5m_as_15m, annotate_signals_AB
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


WINDOW_START = pd.Timestamp("2026-05-12")
WINDOW_END   = pd.Timestamp("2026-05-17 23:59:59")
TRAIN_END    = WINDOW_START   # train strictly before window

ASSETS = [
    {"symbol": "AVAXUSDT", "hist_1h": "data/AVAXUSDT_1h_4y.csv",
     "rec_1h":  "data/AVAXUSDT_recent_1h.csv",
     "rec_5m":  "data/AVAXUSDT_recent_5m.csv"},
    {"symbol": "ADAUSDT",  "hist_1h": "data/ADAUSDT_1h_4y.csv",
     "rec_1h":  "data/ADAUSDT_recent_1h.csv",
     "rec_5m":  "data/ADAUSDT_recent_5m.csv"},
    {"symbol": "SOLUSDT",  "hist_1h": "data/SOLUSDT_1h_4y.csv",
     "rec_1h":  "data/SOLUSDT_recent_1h.csv",
     "rec_5m":  "data/SOLUSDT_recent_5m.csv"},
    {"symbol": "XRPUSDT",  "hist_1h": "data/XRPUSDT_1h_4y.csv",
     "rec_1h":  "data/XRPUSDT_recent_1h.csv",
     "rec_5m":  "data/XRPUSDT_recent_5m.csv"},
]


def _load_and_merge(hist_path: str, rec_path: str) -> pd.DataFrame:
    df_hist = pd.read_csv(hist_path, parse_dates=["date"]).set_index("date")
    df_rec  = pd.read_csv(rec_path,  parse_dates=["date"]).set_index("date")
    df = pd.concat([df_hist, df_rec])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _run_asset(cfg: dict) -> list[dict]:
    sym = cfg["symbol"]
    print(f"\n----  {sym}  ----")

    df_all = _load_and_merge(cfg["hist_1h"], cfg["rec_1h"])
    print(f"  1H bars: {df_all.index[0].date()} → {df_all.index[-1].date()}  "
          f"({len(df_all):,})")

    df_feat = generate_features(df_all.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    train = df_feat[df_feat.index < TRAIN_END]
    # extend signal-generation window a few bars past window end so any
    # trades that opened inside the window can be tracked to exit.
    sig_window = df_feat[(df_feat.index >= TRAIN_END)
                         & (df_feat.index <= WINDOW_END + pd.Timedelta(days=3))]

    if len(train) < 200 or len(sig_window) == 0:
        print(f"  SKIP: train={len(train)} sig_window={len(sig_window)}")
        return []

    print(f"  train bars: {len(train):,}  |  sig window bars: {len(sig_window):,}")

    feat_cols = get_feature_columns(train)
    model = fit_model(train, feat_cols)
    signals = apply_signals(model, feat_cols, sig_window.copy())

    raw = int((signals["signal"] != 0).sum())
    print(f"  raw signals (full sig window): {raw}")

    df_15m = load_5m_as_15m(cfg["rec_5m"])
    annotated = annotate_signals_AB(signals, df_15m)
    annotated["signal"] = annotated["signal_15m_A"]
    a_pass = int((annotated["signal"] != 0).sum())
    pb     = int(annotated["entry_price_15m"].notna().sum())
    print(f"  15m-A passed: {a_pass}   |   15m-B pullback fills: {pb}/{a_pass}")

    trades, _eq = run_backtest(annotated, exit_mode="scaled")
    for t in trades:
        t["symbol"] = sym

    # Keep only trades with entry_date inside [May 12, May 17]
    window_trades = [t for t in trades
                     if WINDOW_START <= t["entry_date"] <= WINDOW_END]
    print(f"  trades with entry in window: {len(window_trades)}  "
          f"(total in sig window: {len(trades)})")
    return window_trades


def _print_trade_table(trades: list[dict]) -> None:
    if not trades:
        print("\n  NO TRADES with entry between 2026-05-12 and 2026-05-17.\n")
        return

    trades.sort(key=lambda t: t["entry_date"])

    print(f"\n{'='*120}")
    print(f"  TRADES — entry between 2026-05-12 and 2026-05-17 (scaled exit)")
    print(f"{'='*120}")
    hdr = (f"{'#':>3}  {'Symbol':<10}  {'Entry (UTC)':^19}  {'Dir':^5}  "
           f"{'Entry':>9}  {'Stop':>9}  {'TP3':>9}  "
           f"{'Exit (UTC)':^19}  {'Reason':^6}  {'R':>6}  {'PnL$':>8}")
    print(hdr)
    print("-" * 120)

    wins = 0
    pnl  = 0.0
    for i, t in enumerate(trades, 1):
        ed   = str(t["entry_date"])[:19]
        xd   = str(t["exit_date"])[:19]
        dirn = "LONG" if t["direction"] == "long" else "SHORT"
        r    = t.get("R", 0.0)
        net  = t.get("net_pnl", 0.0)
        if net > 0:
            wins += 1
        pnl += net
        print(f"{i:>3}  {t['symbol']:<10}  {ed:^19}  {dirn:^5}  "
              f"{t['entry_price']:>9.4f}  {t['stop_price']:>9.4f}  "
              f"{t['tp_price']:>9.4f}  {xd:^19}  {t['exit_reason']:^6}  "
              f"{r:>+6.2f}  {net:>+8.2f}")
    print("-" * 120)

    n = len(trades)
    gw = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gl = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] <= 0))
    pf = (gw / gl) if gl > 0 else float("inf")
    print(f"  TOTAL: {n} trades  |  WR={wins/n*100:.1f}%  "
          f"|  PF={pf:.3f}  |  PnL=${pnl:+.2f}\n")

    # Per-asset breakdown
    by_sym: dict[str, list[dict]] = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t)
    print("  By asset:")
    for s, ts in sorted(by_sym.items()):
        ns = len(ts)
        ws = sum(1 for t in ts if t["net_pnl"] > 0)
        ps = sum(t["net_pnl"] for t in ts)
        print(f"    {s:<10}  {ns} trades  WR={ws/ns*100:.0f}%  PnL=${ps:+.2f}")
    print()


def main() -> None:
    print("=" * 70)
    print("  ARKAD MRK — Sim: 2026-05-12 → 2026-05-17  |  scaled exit")
    print(f"  (config.BACKTEST_EXIT_MODE = {config.BACKTEST_EXIT_MODE})")
    print("=" * 70)

    all_trades: list[dict] = []
    for cfg in ASSETS:
        all_trades.extend(_run_asset(cfg))

    _print_trade_table(all_trades)


if __name__ == "__main__":
    main()

"""
experiments/live_sim_apr2026.py
— Simulate what trades would have occurred March 10 – April 22, 2026.

Method
------
1. Load 4-year historical 1H data for each Tier-1 asset.
2. Append recent data (March 10 – April 22, downloaded from Bybit).
3. Train the CalibratedRF model on ALL bars before March 10, 2026.
4. Generate signals for March 10 – April 22 bars.
5. Apply 15m-AB execution layer using downloaded 5m data (resampled to 15m).
6. Run backtest and print every trade with MFE stats.

Usage
-----
    python experiments/live_sim_apr2026.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from features.execution_15m import load_5m_as_15m, annotate_signals_AB
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics

# ── Config ─────────────────────────────────────────────────────────────────────

TEST_START = pd.Timestamp("2026-03-10")
TEST_END   = pd.Timestamp("2026-04-22 23:59:59")

ASSETS = [
    {"symbol": "AVAXUSDT", "hist_1h": "data/AVAXUSDT_1h_4y.csv",
     "rec_1h": "data/AVAXUSDT_recent_1h.csv", "rec_5m": "data/AVAXUSDT_recent_5m.csv"},
    {"symbol": "ADAUSDT",  "hist_1h": "data/ADAUSDT_1h_4y.csv",
     "rec_1h": "data/ADAUSDT_recent_1h.csv",  "rec_5m": "data/ADAUSDT_recent_5m.csv"},
    {"symbol": "SOLUSDT",  "hist_1h": "data/SOLUSDT_1h_4y.csv",
     "rec_1h": "data/SOLUSDT_recent_1h.csv",  "rec_5m": "data/SOLUSDT_recent_5m.csv"},
    {"symbol": "XRPUSDT",  "hist_1h": "data/XRPUSDT_1h_4y.csv",
     "rec_1h": "data/XRPUSDT_recent_1h.csv",  "rec_5m": "data/XRPUSDT_recent_5m.csv"},
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_and_merge(hist_path: str, rec_path: str) -> pd.DataFrame:
    df_hist = pd.read_csv(hist_path, parse_dates=["date"]).set_index("date")
    df_rec  = pd.read_csv(rec_path,  parse_dates=["date"]).set_index("date")
    df = pd.concat([df_hist, df_rec])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _mfe_for_trade(t: dict, df_1h: pd.DataFrame) -> float:
    """
    Calculate MFE as % of TP distance reached.
    For long: (max_high - entry) / (tp - entry) * 100
    For short: (entry - min_low) / (entry - tp) * 100
    Uses 1H bars between entry_date and exit_date.
    """
    entry_date = t["entry_date"]
    exit_date  = t["exit_date"]
    entry      = t["entry_price"]
    tp         = t["tp_price"]
    direction  = 1 if t["direction"] == "long" else -1

    window = df_1h.loc[entry_date:exit_date]
    if len(window) == 0:
        return 0.0

    if direction == 1:
        best = window["high"].max()
        tp_dist = tp - entry
        if tp_dist <= 0:
            return 0.0
        return min((best - entry) / tp_dist * 100, 100.0)
    else:
        best = window["low"].min()
        tp_dist = entry - tp
        if tp_dist <= 0:
            return 0.0
        return min((entry - best) / tp_dist * 100, 100.0)


def _run_asset(cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    sym = cfg["symbol"]
    print(f"\n{'='*60}")
    print(f"  {sym}")
    print(f"{'='*60}")

    # 1. Load and merge
    df_all = _load_and_merge(cfg["hist_1h"], cfg["rec_1h"])
    print(f"  1H data: {df_all.index[0].date()} → {df_all.index[-1].date()}  ({len(df_all):,} bars)")

    # 2. Feature engineering
    df_feat = generate_features(df_all.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    # 3. Split
    train = df_feat[df_feat.index < TEST_START]
    test  = df_feat[(df_feat.index >= TEST_START) & (df_feat.index <= TEST_END)]

    if len(train) < 200 or len(test) == 0:
        print(f"  SKIP: train={len(train)}, test={len(test)}")
        return [], pd.DataFrame()

    print(f"  Train: {len(train):,} bars | Test: {len(test):,} bars")

    # 4. Train model
    feat_cols = get_feature_columns(train)
    model = fit_model(train, feat_cols)

    # 5. Generate signals
    signals = apply_signals(model, feat_cols, test.copy())

    raw_signals = (signals["signal"] != 0).sum()
    print(f"  Raw signals: {raw_signals}  (long={( signals['signal']==1).sum()}, short={(signals['signal']==-1).sum()})")

    # 6. Apply 15m-AB
    df_15m = load_5m_as_15m(cfg["rec_5m"])
    annotated = annotate_signals_AB(signals, df_15m)
    annotated["signal"] = annotated["signal_15m_A"]

    a_pass = (annotated["signal"] != 0).sum()
    a_fail = raw_signals - a_pass
    pb     = annotated["entry_price_15m"].notna().sum()
    print(f"  A filter: {a_pass} passed, {a_fail} blocked | B pullbacks: {pb}/{a_pass}")

    # 7. Backtest
    trades, equity_curve = run_backtest(annotated)

    # Add symbol + MFE to each trade
    for t in trades:
        t["symbol"] = sym
        t["mfe_pct"] = _mfe_for_trade(t, df_all)

    if trades:
        m = compute_metrics(trades, equity_curve)
        print(f"  Trades: {m['n_trades']}  WR={m['win_rate']*100:.1f}%  PF={m['profit_factor']:.3f}  exp=${m['expectancy']:+.2f}")

    return trades, df_all


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  ARKAD MRK — Simulation: March 10 – April 22, 2026")
    print("  Model trained on all data BEFORE March 10 (zero look-ahead)")
    print("=" * 70)

    all_trades = []
    for cfg in ASSETS:
        trades, _ = _run_asset(cfg)
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t.get("entry_date", pd.Timestamp.min))

    if not all_trades:
        print("\n  No trades in this period.")
        return

    # ── Trade table ────────────────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print(f"  ALL TRADES  —  March 10 to April 22, 2026")
    print(f"{'='*105}")
    hdr = (f"{'#':>3}  {'Symbol':<10}  {'Entry date':^16}  {'Dir':^5}  "
           f"{'Entry':>9}  {'Stop':>9}  {'TP':>9}  {'Exit date':^16}  "
           f"{'Reason':^5}  {'PnL$':>7}  {'MFE%TP':>7}")
    print(hdr)
    print("-" * 105)

    april_trades = []
    total_pnl = 0.0
    wins = 0

    for i, t in enumerate(all_trades, 1):
        ed   = str(t["entry_date"])[:16]
        xd   = str(t["exit_date"])[:16]
        dirn = "LONG" if t["direction"] == "long" else "SHORT"
        pnl  = t["net_pnl"]
        mfe  = t["mfe_pct"]
        total_pnl += pnl
        if pnl > 0:
            wins += 1

        # Mark April-only trades
        is_april = t["entry_date"] >= pd.Timestamp("2026-04-01")
        if is_april:
            april_trades.append(t)
            marker = " *"
        else:
            marker = ""

        print(
            f"{i:>3}  {t['symbol']:<10}  {ed:^16}  {dirn:^5}  "
            f"{t['entry_price']:>9.4f}  {t['stop_price']:>9.4f}  {t['tp_price']:>9.4f}  "
            f"{xd:^16}  {t['exit_reason']:^5}  {pnl:>+7.2f}  {mfe:>6.1f}%{marker}"
        )

    print("-" * 105)
    print("  (* = April 2026 entry)")

    n = len(all_trades)
    wr = wins / n * 100
    gross_wins   = sum(t["net_pnl"] for t in all_trades if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in all_trades if t["net_pnl"] <= 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_mfe = sum(t["mfe_pct"] for t in all_trades) / n

    print(f"\n  TOTAL (Mar 10 – Apr 22):  {n} trades  |  WR={wr:.1f}%  |  PF={pf:.3f}  |  PnL=${total_pnl:+.2f}  |  Avg MFE vs TP={avg_mfe:.1f}%")

    # ── April summary ──────────────────────────────────────────────────────────
    if april_trades:
        na   = len(april_trades)
        wa   = sum(1 for t in april_trades if t["net_pnl"] > 0)
        pa   = sum(t["net_pnl"] for t in april_trades)
        gwa  = sum(t["net_pnl"] for t in april_trades if t["net_pnl"] > 0)
        gla  = abs(sum(t["net_pnl"] for t in april_trades if t["net_pnl"] <= 0))
        pfa  = gwa / gla if gla > 0 else float("inf")
        mfea = sum(t["mfe_pct"] for t in april_trades) / na

        print(f"\n  APRIL ONLY:  {na} trades  |  WR={wa/na*100:.1f}%  |  PF={pfa:.3f}  |  PnL=${pa:+.2f}  |  Avg MFE vs TP={mfea:.1f}%")

        print(f"\n  April trades by asset:")
        by_asset: dict[str, list] = {}
        for t in april_trades:
            by_asset.setdefault(t["symbol"], []).append(t)
        for sym, ts in sorted(by_asset.items()):
            ns = len(ts)
            ws = sum(1 for t in ts if t["net_pnl"] > 0)
            ps = sum(t["net_pnl"] for t in ts)
            ms = sum(t["mfe_pct"] for t in ts) / ns
            print(f"    {sym:<12}  {ns} trades  WR={ws/ns*100:.0f}%  PnL=${ps:+.2f}  Avg MFE={ms:.1f}%")

    print()


if __name__ == "__main__":
    main()

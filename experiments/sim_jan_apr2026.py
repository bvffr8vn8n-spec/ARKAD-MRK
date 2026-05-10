"""
experiments/sim_jan_apr2026.py
Simulate Jan 1 – Apr 26, 2026. Model trained on all data before Jan 1.
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
    {"symbol": "AVAXUSDT",
     "hist_1h": "data/AVAXUSDT_1h_4y.csv",
     "new_1h":  "data/AVAXUSDT_2026_1h.csv",
     "new_5m":  "data/AVAXUSDT_2026_5m.csv"},
    {"symbol": "ADAUSDT",
     "hist_1h": "data/ADAUSDT_1h_4y.csv",
     "new_1h":  "data/ADAUSDT_2026_1h.csv",
     "new_5m":  "data/ADAUSDT_2026_5m.csv"},
    {"symbol": "SOLUSDT",
     "hist_1h": "data/SOLUSDT_1h_4y.csv",
     "new_1h":  "data/SOLUSDT_2026_1h.csv",
     "new_5m":  "data/SOLUSDT_2026_5m.csv"},
    {"symbol": "XRPUSDT",
     "hist_1h": "data/XRPUSDT_1h_4y.csv",
     "new_1h":  "data/XRPUSDT_2026_1h.csv",
     "new_5m":  "data/XRPUSDT_2026_5m.csv"},
]


def _load_1h(hist_path, new_path):
    df_hist = pd.read_csv(hist_path, parse_dates=["date"]).set_index("date")
    df_new  = pd.read_csv(new_path,  parse_dates=["date"]).set_index("date")
    df = pd.concat([df_hist, df_new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _mfe_stats(t, df_1h):
    """Return MFE as % of price (from entry) and % of TP distance."""
    entry  = t["entry_price"]
    tp     = t["tp_price"]
    sl     = t["stop_price"]
    d      = 1 if t["direction"] == "long" else -1
    ed     = t["entry_date"]
    xd     = t["exit_date"]

    window = df_1h.loc[ed:xd]
    if len(window) == 0:
        return 0.0, 0.0, entry

    if d == 1:
        best      = window["high"].max()
        mfe_price = max(best - entry, 0)
        tp_dist   = tp - entry
        max_level = best
    else:
        best      = window["low"].min()
        mfe_price = max(entry - best, 0)
        tp_dist   = entry - tp
        max_level = best

    mfe_pct_price = mfe_price / entry * 100          # % от цены входа
    mfe_pct_tp    = min(mfe_price / tp_dist * 100, 100.0) if tp_dist > 0 else 0.0
    return mfe_pct_price, mfe_pct_tp, max_level


def _run_asset(cfg):
    sym = cfg["symbol"]
    print(f"\n{'='*60}")
    print(f"  {sym}")
    print(f"{'='*60}")

    df_all = _load_1h(cfg["hist_1h"], cfg["new_1h"])
    print(f"  1H total: {df_all.index[0].date()} -> {df_all.index[-1].date()}  ({len(df_all):,} bars)")

    df_feat = generate_features(df_all.copy())
    df_feat = add_labels(df_feat)
    df_feat = add_regime_columns(df_feat)
    df_feat = add_session_column(df_feat)
    df_feat = df_feat.dropna()

    train = df_feat[df_feat.index < TEST_START]
    test  = df_feat[(df_feat.index >= TEST_START) & (df_feat.index <= TEST_END)]

    if len(train) < 200 or len(test) == 0:
        print(f"  SKIP: train={len(train)}, test={len(test)}")
        return [], pd.DataFrame()

    print(f"  Train: {len(train):,} bars | Test: {len(test):,} bars")

    feat_cols = get_feature_columns(train)
    model     = fit_model(train, feat_cols)
    signals   = apply_signals(model, feat_cols, test.copy())

    raw = (signals["signal"] != 0).sum()
    print(f"  Raw signals: {raw}  (L={(signals['signal']==1).sum()}, S={(signals['signal']==-1).sum()})")

    df_15m   = load_5m_as_15m(cfg["new_5m"])
    annotated = annotate_signals_AB(signals, df_15m)
    annotated["signal"] = annotated["signal_15m_A"]

    a_pass = (annotated["signal"] != 0).sum()
    pb     = annotated["entry_price_15m"].notna().sum()
    print(f"  A-filter: {a_pass}/{raw} passed | B-pullbacks: {pb}/{a_pass}")

    trades, equity_curve = run_backtest(annotated)

    for t in trades:
        t["symbol"] = sym
        mp, mtp, ml = _mfe_stats(t, df_all)
        t["mfe_pct_price"] = mp
        t["mfe_pct_tp"]    = mtp
        t["max_level"]     = ml

    if trades:
        m = compute_metrics(trades, equity_curve)
        print(f"  Trades: {m['n_trades']}  WR={m['win_rate']*100:.1f}%  PF={m['profit_factor']:.3f}  exp=${m['expectancy']:+.2f}")

    return trades, df_all


def main():
    print("=" * 75)
    print("  ARKAD MRK — Simulation: Jan 1 – Apr 26, 2026")
    print("  Model trained on all data BEFORE Jan 1, 2026  (zero look-ahead)")
    print("=" * 75)

    all_trades = []
    for cfg in ASSETS:
        trades, _ = _run_asset(cfg)
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t.get("entry_date", pd.Timestamp.min))

    if not all_trades:
        print("\n  No trades found.")
        return

    # ── Full trade table ────────────────────────────────────────────────────────
    W = 130
    print(f"\n{'='*W}")
    print(f"  ALL TRADES — Jan 1 to Apr 26, 2026")
    print(f"{'='*W}")
    hdr = (f"{'#':>3}  {'Symbol':<10}  {'Entry':^16}  {'Dir':^5}  "
           f"{'Entry$':>9}  {'Stop':>9}  {'TP':>9}  "
           f"{'Exit':^16}  {'Why':^4}  {'PnL$':>7}  "
           f"{'MFE%цены':>9}  {'MFE%TP':>7}  {'MaxLevel':>10}")
    print(hdr)
    print("-" * W)

    total_pnl = 0.0
    wins      = 0
    jan_t = []; feb_t = []; mar_t = []; apr_t = []

    for i, t in enumerate(all_trades, 1):
        ed   = str(t["entry_date"])[:16]
        xd   = str(t["exit_date"])[:16]
        dirn = "LONG" if t["direction"] == "long" else "SHORT"
        pnl  = t["net_pnl"]
        mp   = t["mfe_pct_price"]
        mtp  = t["mfe_pct_tp"]
        ml   = t["max_level"]
        total_pnl += pnl
        if pnl > 0:
            wins += 1

        mo = t["entry_date"].month
        if mo == 1: jan_t.append(t)
        elif mo == 2: feb_t.append(t)
        elif mo == 3: mar_t.append(t)
        elif mo == 4: apr_t.append(t)

        print(
            f"{i:>3}  {t['symbol']:<10}  {ed:^16}  {dirn:^5}  "
            f"{t['entry_price']:>9.4f}  {t['stop_price']:>9.4f}  {t['tp_price']:>9.4f}  "
            f"{xd:^16}  {t['exit_reason']:^4}  {pnl:>+7.2f}  "
            f"{mp:>8.2f}%  {mtp:>6.1f}%  {ml:>10.4f}"
        )

    print("-" * W)
    n   = len(all_trades)
    wr  = wins / n * 100
    gw  = sum(t["net_pnl"] for t in all_trades if t["net_pnl"] > 0)
    gl  = abs(sum(t["net_pnl"] for t in all_trades if t["net_pnl"] <= 0))
    pf  = gw / gl if gl > 0 else float("inf")
    avg_mp  = sum(t["mfe_pct_price"] for t in all_trades) / n
    avg_mtp = sum(t["mfe_pct_tp"]    for t in all_trades) / n

    print(f"\n  ИТОГО (Jan–Apr 26):  {n} сделок  |  WR={wr:.1f}%  |  PF={pf:.3f}  |"
          f"  PnL=${total_pnl:+.2f}  |  Avg MFE%цены={avg_mp:.2f}%  |  Avg MFE%TP={avg_mtp:.1f}%")

    # ── Monthly breakdown ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  РАЗБИВКА ПО МЕСЯЦАМ")
    print(f"{'─'*60}")
    for label, grp in [("Январь", jan_t), ("Февраль", feb_t), ("Март", mar_t), ("Апрель", apr_t)]:
        if not grp: continue
        ng  = len(grp)
        wg  = sum(1 for t in grp if t["net_pnl"] > 0)
        pg  = sum(t["net_pnl"] for t in grp)
        gwg = sum(t["net_pnl"] for t in grp if t["net_pnl"] > 0)
        glg = abs(sum(t["net_pnl"] for t in grp if t["net_pnl"] <= 0))
        pfg = gwg / glg if glg > 0 else float("inf")
        mmfe= sum(t["mfe_pct_price"] for t in grp) / ng
        mmtp= sum(t["mfe_pct_tp"]    for t in grp) / ng
        print(f"  {label:<9}  {ng:>3} сдел  WR={wg/ng*100:.0f}%  PF={pfg:.3f}  "
              f"PnL=${pg:+.2f}  Avg MFE%цены={mmfe:.2f}%  Avg MFE%TP={mmtp:.1f}%")

    # ── Per-asset summary ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  РАЗБИВКА ПО АКТИВАМ")
    print(f"{'─'*60}")
    by_asset = {}
    for t in all_trades:
        by_asset.setdefault(t["symbol"], []).append(t)
    for sym, ts in sorted(by_asset.items()):
        ns  = len(ts)
        ws  = sum(1 for t in ts if t["net_pnl"] > 0)
        ps  = sum(t["net_pnl"] for t in ts)
        gws = sum(t["net_pnl"] for t in ts if t["net_pnl"] > 0)
        gls = abs(sum(t["net_pnl"] for t in ts if t["net_pnl"] <= 0))
        pfs = gws / gls if gls > 0 else float("inf")
        mmp = sum(t["mfe_pct_price"] for t in ts) / ns
        mmt = sum(t["mfe_pct_tp"]    for t in ts) / ns
        print(f"  {sym:<12}  {ns:>3} сдел  WR={ws/ns*100:.0f}%  PF={pfs:.3f}  "
              f"PnL=${ps:+.2f}  Avg MFE%цены={mmp:.2f}%  Avg MFE%TP={mmt:.1f}%")

    # ── MFE distribution ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  РАСПРЕДЕЛЕНИЕ MFE%цены (насколько цена шла в нужную сторону)")
    print(f"{'─'*60}")
    mfes = sorted([t["mfe_pct_price"] for t in all_trades])
    buckets = [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,100)]
    for lo, hi in buckets:
        cnt = sum(1 for v in mfes if lo <= v < hi)
        bar = "█" * cnt
        print(f"  {lo:.1f}–{hi:.0f}%:  {cnt:>3}  {bar}")

    print(f"\n  Мин MFE%цены: {min(mfes):.3f}%  |  Макс: {max(mfes):.3f}%  |  Медиана: {np.median(mfes):.3f}%")
    print(f"  Сделок с MFE > 1% : {sum(1 for v in mfes if v>1)} / {n}")
    print(f"  Сделок с MFE > 2% : {sum(1 for v in mfes if v>2)} / {n}")
    print()


if __name__ == "__main__":
    main()

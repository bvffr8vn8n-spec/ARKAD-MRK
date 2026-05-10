"""
experiments/multi_asset_selection.py
— Pipeline-identical multi-asset selection with 15m-AB execution layer.

Purpose
-------
Identify which trading pairs produce a stable OOS edge (WF PF > 1.0) using the
validated ARKAD pipeline. No per-asset parameter tuning: same model, same
thresholds, same execution logic for every asset.

Assets evaluated
----------------
  Auto-detected from data/:
    data/{SYMBOL}_1h_4y.csv   — required (1H bars, 4 years)
    data/{SYMBOL}_5m_4y.csv   — optional (5m bars → resampled to 15m)

  Candidate pool (all top-10 USDT perps from Bybit):
    BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT,
    ADAUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT, MATICUSDT

Classification thresholds
--------------------------
  STRONG    : WF avg PF (15m-covered windows) >= 1.0
  PROMISING : WF avg PF >= 0.85
  WEAK      : WF avg PF < 0.85
  BASELINE  : no 15m data — classified on Baseline WF PF

Ranking
-------
  Primary   : WF PF (descending)
  Secondary : Expectancy (descending)

Portfolio simulation (optional, top STRONG assets)
---------------------------------------------------
  Equal capital split across selected symbols.
  Combined P&L from all trade records, equity rebased to INITIAL_CAPITAL total.

Output
------
  experiments/multi_asset_selection.csv
  Console report with ranked table, recommendations, and portfolio summary.

Usage
-----
  python experiments/multi_asset_selection.py
  python experiments/multi_asset_selection.py --no-portfolio
"""

import argparse
import contextlib
import io
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from data.loader import load_ohlcv
from features.generator import generate_features, add_labels
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from features.execution_15m import (
    load_5m_as_15m,
    annotate_signals_AB,
    coverage_stats,
)
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


# ── Asset candidate pool ───────────────────────────────────────────────────────

_CANDIDATES = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "MATICUSDT",
]

# Classification thresholds (applied to WF avg PF on 15m-covered windows)
STRONG_THRESHOLD    = 1.00
PROMISING_THRESHOLD = 0.85

_WF_KEYS = ["n_trades", "profit_factor", "win_rate", "expectancy", "max_drawdown"]


# ── Data discovery ─────────────────────────────────────────────────────────────

def _discover_assets(data_dir: str) -> list[dict]:
    """
    Scan data_dir for available 1H and 5m files.
    Returns a list of asset configs for each candidate that has at least a 1H file.
    """
    assets = []
    for sym in _CANDIDATES:
        f1h = os.path.join(data_dir, f"{sym}_1h_4y.csv")
        f5m = os.path.join(data_dir, f"{sym}_5m_4y.csv")
        if not os.path.exists(f1h):
            print(f"  [SKIP] {sym}: no 1H data at {f1h}")
            continue
        has5m = os.path.exists(f5m)
        assets.append({
            "symbol":  sym,
            "data_1h": f1h,
            "data_5m": f5m if has5m else None,
        })
    return assets


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v, fmt="", fallback="N/A"):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return fallback
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


def _has_15m_coverage(test_df: pd.DataFrame, df_15m: pd.DataFrame,
                      threshold: float = 0.5) -> bool:
    if df_15m is None or len(df_15m) == 0:
        return False
    start = df_15m.index[0]
    return float((test_df.index >= start).mean()) >= threshold


def _wf_avg_pf(rows: list[dict], coverage_only: bool = False) -> float:
    subset = [r for r in rows
              if r["n_trades"] > 0
              and not math.isnan(r.get("profit_factor", float("nan")))
              and (not coverage_only or r.get("has_15m", False))]
    return float(np.mean([r["profit_factor"] for r in subset])) if subset else float("nan")


def _wf_avg_exp(rows: list[dict], coverage_only: bool = False) -> float:
    subset = [r for r in rows
              if r["n_trades"] > 0
              and not math.isnan(r.get("expectancy", float("nan")))
              and (not coverage_only or r.get("has_15m", False))]
    return float(np.mean([r["expectancy"] for r in subset])) if subset else float("nan")


def _wf_trades(rows: list[dict], coverage_only: bool = False) -> int:
    return sum(r["n_trades"] for r in rows
               if not coverage_only or r.get("has_15m", False))


def _classify(wf_pf: float, has_15m: bool) -> str:
    """Return STRONG / PROMISING / WEAK / NO_15M."""
    if not has_15m:
        return "NO_15M"
    if math.isnan(wf_pf):
        return "INSUFFICIENT"
    if wf_pf >= STRONG_THRESHOLD:
        return "STRONG"
    if wf_pf >= PROMISING_THRESHOLD:
        return "PROMISING"
    return "WEAK"


# ── WF runner ─────────────────────────────────────────────────────────────────

def _run_wf(
    df:           pd.DataFrame,
    feature_cols: list[str],
    variant:      str,             # "Baseline" or "15m-AB"
    df_15m:       pd.DataFrame | None,
) -> list[dict]:
    n    = len(df)
    rows = []
    for train_frac in config.WF_TRAIN_FRACTIONS:
        test_frac  = train_frac + config.WF_TEST_FRACTION
        train_end  = int(n * train_frac)
        test_end   = min(int(n * test_frac), n)
        purge_end  = train_end - config.FORWARD_RETURN_WINDOW

        train_slice = df.iloc[0:purge_end]
        test_slice  = df.iloc[train_end:test_end]

        empty = {k: (0 if k == "n_trades" else float("nan")) for k in _WF_KEYS}
        empty.update({"train_frac": train_frac, "test_frac": min(test_frac, 1.0),
                       "has_15m": False, "n_skipped": 0, "n_pullback": 0})

        if len(train_slice) < 50 or len(test_slice) < 5:
            rows.append(empty)
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = fit_model(train_slice, feature_cols)

        signals_wf = apply_signals(model, feature_cols, test_slice)
        has_cov    = _has_15m_coverage(test_slice, df_15m)

        if variant == "15m-AB" and df_15m is not None:
            ann = annotate_signals_AB(signals_wf, df_15m)
            n_skipped  = int(((signals_wf["signal"] != 0) &
                              (ann["signal_15m_A"] == 0)).sum())
            n_pullback = int(ann["entry_price_15m"].notna().sum())
            ann["signal"] = ann["signal_15m_A"]
            df_bt = ann
        else:
            df_bt      = signals_wf
            n_skipped  = 0
            n_pullback = 0

        trades, equity = run_backtest(df_bt)
        m = compute_metrics(trades, equity)

        if "error" in m:
            rows.append(empty)
        else:
            row = {k: m.get(k, float("nan")) for k in _WF_KEYS}
            row["n_trades"]   = int(row["n_trades"] or 0)
            row["train_frac"] = train_frac
            row["test_frac"]  = min(test_frac, 1.0)
            row["has_15m"]    = has_cov
            row["n_skipped"]  = n_skipped
            row["n_pullback"] = n_pullback
            rows.append(row)
    return rows


# ── Per-asset runner ───────────────────────────────────────────────────────────

def run_asset(symbol: str, data_1h: str, data_5m: str | None) -> dict:
    """
    Run full pipeline (Baseline + 15m-AB) for one asset.
    Returns a results dict used for selection ranking.
    """
    print(f"\n  {'='*68}")
    mode = "1H + 5m->15m" if data_5m else "1H baseline only"
    print(f"  {symbol}  ({mode})")
    print(f"  {'='*68}")

    t0 = time.time()

    # ── Load + feature pipeline ───────────────────────────────────────────────
    df = load_ohlcv(data_1h)
    df = generate_features(df)
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)
    n = len(df)

    # ── Load 15m data ─────────────────────────────────────────────────────────
    df_15m = None
    if data_5m is not None and os.path.exists(data_5m):
        df_15m = load_5m_as_15m(data_5m)
        print(f"  1H bars : {n:,}  ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"  15m bars: {len(df_15m):,}  ({df_15m.index[0].date()} to {df_15m.index[-1].date()})")
    else:
        print(f"  1H bars : {n:,}  ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"  15m data: not available")

    # ── Train (full 80/20 split for quick BT) ─────────────────────────────────
    feature_cols = get_feature_columns(df)
    split        = int(n * (1 - config.TEST_SIZE))
    train_df     = df.iloc[:split]
    test_df      = df.iloc[split:]
    n_test_days  = max((test_df.index[-1] - test_df.index[0]).days, 1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = fit_model(train_df, feature_cols)
    print(f"  Trained {time.time()-t0:.1f}s  |  test: {len(test_df):,}b "
          f"({n_test_days}d, {test_df.index[0].date()} to {test_df.index[-1].date()})")

    signals_df = apply_signals(model, feature_cols, test_df)
    n_sig = int((signals_df["signal"] != 0).sum())
    print(f"  Signals in test: {n_sig}  "
          f"(L={int((signals_df['signal']==1).sum())}  "
          f"S={int((signals_df['signal']==-1).sum())})")

    # ── Backtest: Baseline ────────────────────────────────────────────────────
    trades_base, eq_base = run_backtest(signals_df)
    m_base = compute_metrics(trades_base, eq_base)

    # ── Backtest: 15m-AB ──────────────────────────────────────────────────────
    m_ab      = None
    trades_ab = None
    eq_ab     = None
    ann       = None
    if df_15m is not None:
        ann = annotate_signals_AB(signals_df, df_15m)
        ann["signal"] = ann["signal_15m_A"]
        trades_ab, eq_ab = run_backtest(ann)
        m_ab = compute_metrics(trades_ab, eq_ab)

    # Print quick summary
    def _r(m):
        if m and "error" not in m:
            return (f"n={m.get('n_trades',0):>3}  "
                    f"PF={_fmt(m.get('profit_factor'), '.3f')}  "
                    f"exp={_fmt(m.get('expectancy'), '+.2f')}")
        return "no trades"

    print(f"  BT Baseline: {_r(m_base)}")
    if m_ab:
        print(f"  BT 15m-AB:   {_r(m_ab)}")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    wf_base = _run_wf(df, feature_cols, "Baseline", df_15m)
    wf_ab   = _run_wf(df, feature_cols, "15m-AB",   df_15m) if df_15m is not None else None

    # Print WF summary per window
    wf_sep = "─" * 68
    print(f"\n  Walk-Forward ({symbol})")
    print(f"  {wf_sep}")
    for wi, rb in enumerate(wf_base):
        cov  = "(15m)" if rb.get("has_15m") else "(----)"
        lbl  = f"  Win{wi+1} [{rb['train_frac']*100:.0f}%-{rb['test_frac']*100:.0f}%] {cov}"
        line = f"{lbl:<32}  Baseline: {rb['n_trades']:>3}tr PF={_fmt(rb.get('profit_factor'),'.3f')}"
        if wf_ab:
            ra   = wf_ab[wi]
            sk   = ra.get("n_skipped", 0)
            skst = f" sk={sk}" if ra.get("has_15m") and sk else ""
            line += f"  |  15m-AB: {ra['n_trades']:>3}tr PF={_fmt(ra.get('profit_factor'),'.3f')}{skst}"
        print(line)
    print(f"  {wf_sep}")
    base_pf_co = _wf_avg_pf(wf_base, coverage_only=True)
    ab_pf_co   = _wf_avg_pf(wf_ab,   coverage_only=True) if wf_ab else float("nan")
    print(f"  WF avg PF (base/15m-AB, covered): {_fmt(base_pf_co,'.3f')} / {_fmt(ab_pf_co,'.3f')}")
    print(f"  {wf_sep}")

    # Primary WF PF used for classification: 15m-AB if available, else Baseline (all)
    if wf_ab is not None:
        primary_pf  = ab_pf_co
        primary_exp = _wf_avg_exp(wf_ab, coverage_only=True)
    else:
        primary_pf  = _wf_avg_pf(wf_base, coverage_only=False)
        primary_exp = _wf_avg_exp(wf_base, coverage_only=False)

    cls = _classify(primary_pf, df_15m is not None)
    print(f"  Classification: {cls}  (WF PF={_fmt(primary_pf,'.3f')})")

    print(f"\n  [{symbol}] done in {time.time()-t0:.0f}s")

    return {
        "symbol":        symbol,
        "has_15m":       df_15m is not None,
        "test_start":    test_df.index[0].date(),
        "test_end":      test_df.index[-1].date(),
        # Backtest
        "bt_base_n":     m_base.get("n_trades", 0),
        "bt_base_pf":    m_base.get("profit_factor", float("nan")),
        "bt_base_exp":   m_base.get("expectancy", float("nan")),
        "bt_ab_n":       m_ab.get("n_trades", 0) if m_ab else None,
        "bt_ab_pf":      m_ab.get("profit_factor", float("nan")) if m_ab else None,
        "bt_ab_exp":     m_ab.get("expectancy", float("nan")) if m_ab else None,
        # Walk-forward
        "wf_base_pf_all": _wf_avg_pf(wf_base, coverage_only=False),
        "wf_base_pf_co":  _wf_avg_pf(wf_base, coverage_only=True),
        "wf_base_exp_co": _wf_avg_exp(wf_base, coverage_only=True),
        "wf_base_tr_co":  _wf_trades(wf_base, coverage_only=True),
        "wf_ab_pf_co":    _wf_avg_pf(wf_ab,   coverage_only=True) if wf_ab else None,
        "wf_ab_exp_co":   _wf_avg_exp(wf_ab,  coverage_only=True) if wf_ab else None,
        "wf_ab_tr_co":    _wf_trades(wf_ab,   coverage_only=True) if wf_ab else None,
        # Classification
        "primary_pf":    primary_pf,
        "primary_exp":   primary_exp,
        "classification": cls,
        # Raw rows for per-window breakdown
        "wf_rows_base":  wf_base,
        "wf_rows_ab":    wf_ab,
        # Trade lists for portfolio simulation
        "trades_ab":     trades_ab,   # list[dict] or None
        "equity_ab":     eq_ab,       # pd.Series or None
        "trades_base":   trades_base,
        "equity_base":   eq_base,
    }


# ── Portfolio simulation ───────────────────────────────────────────────────────

def _portfolio_sim(results: list[dict], top_symbols: list[str]) -> dict | None:
    """
    Equal-capital portfolio across top_symbols using 15m-AB trades.

    Each symbol's trades are scaled to capital = INITIAL_CAPITAL / n_assets.
    All trades are merged chronologically and a combined equity curve built.
    """
    selected = [r for r in results if r["symbol"] in top_symbols
                and r.get("trades_ab") is not None]
    if not selected:
        return None

    n = len(selected)
    cap_each = config.INITIAL_CAPITAL / n

    # Scale each trade's net_pnl to the per-asset capital
    all_trades = []
    for r in selected:
        scale = cap_each / config.INITIAL_CAPITAL
        for t in r["trades_ab"]:
            scaled = dict(t)
            scaled["net_pnl"]   = t["net_pnl"]   * scale
            scaled["gross_pnl"] = t["gross_pnl"]  * scale
            scaled["symbol"]    = r["symbol"]
            all_trades.append(scaled)

    if not all_trades:
        return None

    all_trades.sort(key=lambda t: t["exit_date"])

    # Build combined equity
    equity = config.INITIAL_CAPITAL
    equity_curve = []
    for t in all_trades:
        equity += t["net_pnl"]
        equity_curve.append(equity)

    gross_profits = sum(t["net_pnl"] for t in all_trades if t["net_pnl"] > 0)
    gross_losses  = abs(sum(t["net_pnl"] for t in all_trades if t["net_pnl"] < 0))
    pf     = gross_profits / gross_losses if gross_losses > 0 else float("inf")
    wins   = sum(1 for t in all_trades if t["net_pnl"] > 0)
    wr     = wins / len(all_trades) if all_trades else 0
    exp    = float(np.mean([t["net_pnl"] for t in all_trades]))
    final  = equity_curve[-1] if equity_curve else config.INITIAL_CAPITAL
    ret    = (final / config.INITIAL_CAPITAL - 1) * 100
    series = pd.Series(equity_curve)
    dd_arr = series / series.cummax() - 1
    max_dd = float(dd_arr.min()) * 100

    return {
        "symbols":      [r["symbol"] for r in selected],
        "n_trades":     len(all_trades),
        "profit_factor": pf,
        "win_rate":      wr,
        "expectancy":   exp,
        "total_return_pct": ret,
        "max_drawdown_pct": max_dd,
        "final_equity": final,
    }


# ── Output: ranked table + recommendations ────────────────────────────────────

def _print_selection(results: list[dict], run_portfolio: bool) -> None:
    sep = "=" * 92

    print(f"\n\n  {sep}")
    print(f"  MULTI-ASSET SELECTION — ARKAD MRK  |  15m-AB Execution Layer")
    print(f"  {sep}")

    # Sort by primary_pf descending, then primary_exp descending
    ranked = sorted(
        results,
        key=lambda r: (
            0 if math.isnan(r["primary_pf"]) else r["primary_pf"],
            0 if math.isnan(r.get("primary_exp", float("nan"))) else r["primary_exp"],
        ),
        reverse=True,
    )

    # ── Ranked table ─────────────────────────────────────────────────────────
    col  = 10
    hdr  = (f"  {'Symbol':<10} {'Trades':>{col}} {'BT PF':>{col}} "
            f"{'BT exp$':>{col}} {'WF PF':>{col}} {'WF exp$':>{col}} "
            f"{'WF Tr':>{col}} {'Class':<12}")
    print(f"\n  Ranked by WF PF (15m-covered windows):\n")
    print(hdr)
    print(f"  {'─'*82}")

    for rank, r in enumerate(ranked, 1):
        has15 = r["has_15m"]
        if has15:
            n_tr  = r["bt_ab_n"]   or 0
            bt_pf = r["bt_ab_pf"]
            bt_exp= r["bt_ab_exp"]
            wf_pf = r["wf_ab_pf_co"]
            wf_exp= r["wf_ab_exp_co"]
            wf_tr = r["wf_ab_tr_co"] or 0
        else:
            n_tr  = r["bt_base_n"]   or 0
            bt_pf = r["bt_base_pf"]
            bt_exp= r["bt_base_exp"]
            wf_pf = r["wf_base_pf_all"]
            wf_exp= r["wf_base_exp_co"]
            wf_tr = r["wf_base_tr_co"] or 0

        cls   = r["classification"]
        mark  = ""
        if cls == "STRONG":
            mark = " <<<"
        elif cls == "PROMISING":
            mark = " <"

        print(f"  {rank}. {r['symbol']:<8} "
              f"{n_tr:>{col}} "
              f"{_fmt(bt_pf, '.3f'):>{col}} "
              f"{_fmt(bt_exp, '+.2f'):>{col}} "
              f"{_fmt(wf_pf, '.3f'):>{col}} "
              f"{_fmt(wf_exp, '+.2f'):>{col}} "
              f"{wf_tr:>{col}} "
              f"{cls:<12}{mark}")

    print(f"  {'─'*82}")
    print(f"  WF PF = walk-forward avg profit factor (15m-covered windows)")
    print(f"  <<< = STRONG (WF PF >= {STRONG_THRESHOLD})  "
          f"< = PROMISING (>= {PROMISING_THRESHOLD})")

    # ── Per-window breakdown ──────────────────────────────────────────────────
    print(f"\n  Per-window WF breakdown:")
    win_sep = "─" * 70
    for r in ranked:
        sym    = r["symbol"]
        wf_b   = r["wf_rows_base"]
        wf_ab  = r["wf_rows_ab"]
        has15  = r["has_15m"]
        print(f"\n  {sym}  [{r['test_start']} to {r['test_end']}]")
        for wi, rb in enumerate(wf_b):
            cov  = "(15m)" if rb.get("has_15m") else "(----)"
            lbl  = f"    Win{wi+1} [{rb['train_frac']*100:.0f}%-{rb['test_frac']*100:.0f}%] {cov}"
            line = f"{lbl:<32}  Base: {rb['n_trades']:>3}tr PF={_fmt(rb.get('profit_factor'),'.3f')}"
            if wf_ab:
                ra   = wf_ab[wi]
                sk   = ra.get("n_skipped", 0)
                skst = f" sk={sk}" if ra.get("has_15m") and sk else ""
                line += f"  |  15m-AB: {ra['n_trades']:>3}tr PF={_fmt(ra.get('profit_factor'),'.3f')}{skst}"
            print(line)

    # ── Recommendations ───────────────────────────────────────────────────────
    strong    = [r for r in ranked if r["classification"] == "STRONG"]
    promising = [r for r in ranked if r["classification"] == "PROMISING"]
    weak      = [r for r in ranked if r["classification"] == "WEAK"]
    no_15m    = [r for r in ranked if r["classification"] == "NO_15M"]

    print(f"\n  {sep}")
    print(f"  RECOMMENDATIONS")
    print(f"  {sep}")

    top = strong if strong else promising[:2]
    top_syms = [r["symbol"] for r in top]

    if strong:
        print(f"\n  TOP PICKS (STRONG — WF PF >= {STRONG_THRESHOLD}):")
        for i, r in enumerate(strong, 1):
            wf_pf_str = _fmt(r["wf_ab_pf_co"], ".3f") if r["has_15m"] else _fmt(r["wf_base_pf_all"], ".3f")
            exp_str   = _fmt(r.get("wf_ab_exp_co") if r["has_15m"] else r.get("wf_base_exp_co"), "+.2f")
            print(f"    {i}. {r['symbol']:<12}  WF PF = {wf_pf_str}  WF exp = {exp_str}  ({r['bt_ab_n'] or r['bt_base_n']} trades/test)")
    else:
        print(f"\n  No STRONG assets found. Best PROMISING picks:")
        for i, r in enumerate(promising[:3], 1):
            wf_pf_str = _fmt(r["wf_ab_pf_co"], ".3f") if r["has_15m"] else _fmt(r["wf_base_pf_all"], ".3f")
            print(f"    {i}. {r['symbol']:<12}  WF PF = {wf_pf_str}")

    if weak:
        print(f"\n  REJECTED (WEAK — WF PF < {PROMISING_THRESHOLD}):")
        for r in weak:
            pf_str = _fmt(r["wf_ab_pf_co"] if r["has_15m"] else r["wf_base_pf_all"], ".3f")
            print(f"    - {r['symbol']:<12}  WF PF = {pf_str}")

    if no_15m:
        print(f"\n  AWAITING 15m DATA (baseline only, cannot classify):")
        for r in no_15m:
            print(f"    - {r['symbol']:<12}  Baseline WF PF = {_fmt(r['wf_base_pf_all'],'.3f')}")

    # ── Why top / why not ─────────────────────────────────────────────────────
    print(f"\n  ANALYSIS:")
    if strong:
        print(f"\n    Why top picks work:")
        for r in strong:
            wf_pf = r["wf_ab_pf_co"] if r["has_15m"] else r["wf_base_pf_all"]
            base_pf = r["wf_base_pf_all"]
            lift = wf_pf - base_pf if not (math.isnan(wf_pf) or math.isnan(base_pf)) else float("nan")
            print(f"    {r['symbol']}: WF PF={_fmt(wf_pf,'.3f')}  "
                  f"(baseline={_fmt(base_pf,'.3f')}, 15m-AB lift={_fmt(lift,'+.3f')})")
            print(f"      → 15m momentum filter consistently removes low-quality signals OOS.")
    if weak or no_15m:
        print(f"\n    Why others fail / need data:")
        for r in weak:
            print(f"    {r['symbol']}: WF PF={_fmt(r['wf_ab_pf_co'],'.3f')} — "
                  f"15m filter insufficient to overcome baseline noise (base={_fmt(r['wf_base_pf_all'],'.3f')})")
        for r in no_15m:
            print(f"    {r['symbol']}: no 5m data — 15m-AB cannot be applied. "
                  f"Baseline WF={_fmt(r['wf_base_pf_all'],'.3f')} (not actionable alone).")

    # ── Portfolio simulation ───────────────────────────────────────────────────
    if run_portfolio and top_syms:
        print(f"\n  {sep}")
        print(f"  PORTFOLIO SIMULATION  ({', '.join(top_syms)})")
        print(f"  {sep}")
        pf_result = _portfolio_sim(results, top_syms)
        if pf_result:
            n  = len(pf_result["symbols"])
            print(f"\n  Assets in portfolio : {', '.join(pf_result['symbols'])}")
            print(f"  Capital each       : ${config.INITIAL_CAPITAL / n:,.0f}")
            print(f"  Total trades       : {pf_result['n_trades']}")
            print(f"  Combined PF        : {_fmt(pf_result['profit_factor'], '.3f')}")
            print(f"  Win rate           : {pf_result['win_rate']*100:.1f}%")
            print(f"  Avg trade exp $    : {_fmt(pf_result['expectancy'], '+.2f')}")
            print(f"  Total return       : {_fmt(pf_result['total_return_pct'], '+.1f')}%")
            print(f"  Max drawdown       : {_fmt(pf_result['max_drawdown_pct'], '.1f')}%")
            print(f"  Final equity       : ${pf_result['final_equity']:,.2f} "
                  f"(started at ${config.INITIAL_CAPITAL:,.0f})")
        else:
            print(f"  No qualifying trades for portfolio simulation.")

    print(f"\n  {sep}\n")

    return ranked


# ── CSV output ────────────────────────────────────────────────────────────────

def _save_csv(results: list[dict], out_path: str) -> None:
    rows = []
    for r in results:
        has15 = r["has_15m"]
        rows.append({
            "symbol":            r["symbol"],
            "has_15m":           has15,
            "test_start":        r["test_start"],
            "test_end":          r["test_end"],
            "bt_n":              r["bt_ab_n"] if has15 else r["bt_base_n"],
            "bt_pf":             r["bt_ab_pf"] if has15 else r["bt_base_pf"],
            "bt_exp":            r["bt_ab_exp"] if has15 else r["bt_base_exp"],
            "wf_pf":             r["wf_ab_pf_co"] if has15 else r["wf_base_pf_all"],
            "wf_exp":            r["wf_ab_exp_co"] if has15 else r["wf_base_exp_co"],
            "wf_trades":         r["wf_ab_tr_co"] if has15 else r["wf_base_tr_co"],
            "wf_base_pf_all":    r["wf_base_pf_all"],
            "wf_base_pf_co":     r["wf_base_pf_co"],
            "wf_ab_pf_co":       r["wf_ab_pf_co"],
            "classification":    r["classification"],
        })
    # Sort by WF PF descending
    rows.sort(key=lambda x: (x["wf_pf"] or 0), reverse=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Results saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-asset selection with 15m-AB execution layer.")
    parser.add_argument("--no-portfolio", action="store_true",
                        help="Skip portfolio simulation.")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  ARKAD MRK — Multi-Asset Selection")
    print(f"  Pipeline: 1H CalibratedRF  +  15m-AB execution (A+B)")
    print(f"  Config  : threshold={config.BUY_PROB_THRESHOLD}  "
          f"TP={config.TAKE_PROFIT_ATR_MULT}x  SL={config.STOP_LOSS_ATR_MULT}x  "
          f"HOLD={config.HOLD_BARS}H")
    print(f"{'='*80}")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    assets   = _discover_assets(data_dir)

    if not assets:
        print("  No assets found. Run: python data/download_all.py")
        return

    print(f"\n  Assets to evaluate: {len(assets)}")
    for a in assets:
        tag = "1H+5m" if a["data_5m"] else "1H only"
        print(f"    {a['symbol']:<14}  {tag}")

    results = []
    for a in assets:
        t0 = time.time()
        r  = run_asset(a["symbol"], a["data_1h"], a["data_5m"])
        results.append(r)

    ranked = _print_selection(results, run_portfolio=not args.no_portfolio)

    out_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "multi_asset_selection.csv"
    )
    _save_csv(results, out_csv)


if __name__ == "__main__":
    main()

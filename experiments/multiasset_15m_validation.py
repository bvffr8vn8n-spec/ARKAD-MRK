"""
experiments/multiasset_15m_validation.py
— Cross-asset robustness check for the 15m-AB execution layer.

Purpose
-------
Determine whether the 15m-AB result (BTC WF PF=1.029, Win3 PF=1.27) transfers
to other assets, or is a BTC-2024-specific effect.

Assets evaluated
----------------
  Full 15m coverage (5m resampled to 15m, covers the full 4-year dataset):
    BTCUSDT  — 2021-01 to 2024-12  (1H + 5m_4y)
    AVAXUSDT — 2022-03 to 2026-03  (1H + 5m_4y)  ← test set in 2025, truly OOS

  Baseline-only reference (no 5m data available):
    ETHUSDT  — 2022-03 to 2026-03  (1H only)
    ADAUSDT  — 2022-03 to 2026-03  (1H only)

  Excluded:
    MATICUSDT — only 18 months of 1H data (too short for 3 WF windows)
    XRPUSDT   — no 1H data

Verdict criteria
----------------
  Strong : both BTC + AVAX WF avg PF (covered) >= 1.0 with positive expectancy
  Moderate: one of two assets clears 1.0 OOS
  Weak  : neither asset clears 1.0 OOS
  Inconclusive: one or both assets have too few WF trades (<30 per window)

Usage
-----
  python experiments/multiasset_15m_validation.py
"""

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
    load_15m_data,
    load_5m_as_15m,
    annotate_signals_AB,
    coverage_stats,
)
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


# ── Asset registry ─────────────────────────────────────────────────────────────

ASSETS = [
    # Full 15m validation
    {
        "symbol":  "BTCUSDT",
        "data_1h": "data/BTCUSDT_1h_4y.csv",
        "data_5m": "data/BTCUSDT_5m_4y.csv",
        "note":    "2021-01 to 2024-12 | 5m resampled to 15m",
    },
    {
        "symbol":  "AVAXUSDT",
        "data_1h": "data/AVAXUSDT_1h_4y.csv",
        "data_5m": "data/AVAXUSDT_5m_4y.csv",
        "note":    "2022-03 to 2026-03 | test set in 2025 (truly OOS)",
    },
    # Baseline reference only
    {
        "symbol":  "ETHUSDT",
        "data_1h": "data/ETHUSDT_1h_4y.csv",
        "data_5m": None,
        "note":    "2022-03 to 2026-03 | no 5m data — baseline only",
    },
    {
        "symbol":  "ADAUSDT",
        "data_1h": "data/ADAUSDT_1h_4y.csv",
        "data_5m": None,
        "note":    "2022-03 to 2026-03 | no 5m data — baseline only",
    },
]

_WF_KEYS = ["n_trades", "profit_factor", "win_rate", "expectancy", "max_drawdown"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_15m_coverage(test_df: pd.DataFrame, df_15m: pd.DataFrame,
                      threshold: float = 0.5) -> bool:
    if df_15m is None or len(df_15m) == 0:
        return False
    start = df_15m.index[0]
    return float((test_df.index >= start).mean()) >= threshold


def _fmt(v, fmt="", fallback="N/A"):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return fallback
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


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


# ── Per-asset runner ──────────────────────────────────────────────────────────

def run_asset(
    symbol:   str,
    data_1h:  str,
    data_5m:  str | None,
    note:     str,
) -> dict:
    """
    Run Baseline vs 15m-AB for one asset.  Returns a results dict.
    When data_5m is None, only Baseline is run (no 15m-AB).
    """
    sep = "─" * 68
    print(f"\n  {'='*68}")
    print(f"  {symbol}  |  {note}")
    print(f"  {'='*68}")

    # ── Load 1H data ─────────────────────────────────────────────────────────
    t0 = time.time()
    df  = load_ohlcv(data_1h)
    df  = generate_features(df)
    df  = add_regime_columns(df)
    df  = add_session_column(df)
    df  = add_labels(df)
    df.dropna(inplace=True)
    n   = len(df)

    # ── Load 15m data (if available) ─────────────────────────────────────────
    df_15m = None
    if data_5m is not None:
        df_15m = load_5m_as_15m(data_5m)
        print(f"  1H bars: {n:,}  ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"  15m bars: {len(df_15m):,}  ({df_15m.index[0].date()} to {df_15m.index[-1].date()})")
    else:
        print(f"  1H bars: {n:,}  ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"  15m data: NONE — baseline-only asset")

    # ── Train on 80% ─────────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    split        = int(n * (1 - config.TEST_SIZE))
    train_df     = df.iloc[:split]
    test_df      = df.iloc[split:]
    n_test_days  = max((test_df.index[-1] - test_df.index[0]).days, 1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = fit_model(train_df, feature_cols)
    print(f"  Model trained in {time.time()-t0:.1f}s  |  "
          f"test: {len(test_df):,} bars ({n_test_days}d, "
          f"{test_df.index[0].date()} to {test_df.index[-1].date()})")

    signals_df = apply_signals(model, feature_cols, test_df)
    n_raw      = int((signals_df["signal"] != 0).sum())
    print(f"  Raw signals in test set: {n_raw}  "
          f"(long={int((signals_df['signal']==1).sum())}  "
          f"short={int((signals_df['signal']==-1).sum())})")

    # ── Backtest: Baseline ────────────────────────────────────────────────────
    trades_base, eq_base = run_backtest(signals_df)
    m_base = compute_metrics(trades_base, eq_base)

    # ── Backtest: 15m-AB (if 15m data available) ──────────────────────────────
    m_ab = None
    stats_ab = None
    if df_15m is not None:
        ann = annotate_signals_AB(signals_df, df_15m)
        ann["signal"] = ann["signal_15m_A"]
        stats_ab      = coverage_stats(signals_df, ann, df_15m, window_label=symbol)
        trades_ab, eq_ab = run_backtest(ann)
        m_ab = compute_metrics(trades_ab, eq_ab)
        print(f"  15m coverage: {stats_ab['signals_with_15m']}/{stats_ab['total_signals']} signals  "
              f"|  pullback={stats_ab['pullback_entries']}  "
              f"filtered_by_A={stats_ab['total_signals'] - stats_ab['pullback_entries'] - stats_ab['baseline_fallback']}")

    # Print quick BT comparison
    def _r(m):
        if m and "error" not in m:
            return (f"n={m.get('n_trades',0)}  "
                    f"PF={_fmt(m.get('profit_factor'), '.3f')}  "
                    f"exp={_fmt(m.get('expectancy'), '+.2f')}")
        return "no trades"

    print(f"  {sep}")
    print(f"  BT Baseline: {_r(m_base)}")
    if m_ab is not None:
        print(f"  BT 15m-AB:   {_r(m_ab)}")
    print(f"  {sep}")

    # ── Walk-forward ─────────────────────────────────────────────────────────
    wf_base = _run_wf(df, feature_cols, "Baseline", df_15m)
    wf_ab   = _run_wf(df, feature_cols, "15m-AB",   df_15m) if df_15m is not None else None

    # Print WF table
    _print_wf_rows(symbol, wf_base, wf_ab, df_15m)

    return {
        "symbol":       symbol,
        "has_15m":      df_15m is not None,
        "n_test_days":  n_test_days,
        "test_start":   test_df.index[0].date(),
        "test_end":     test_df.index[-1].date(),
        # Baseline
        "bt_base_n":    m_base.get("n_trades",     0),
        "bt_base_pf":   m_base.get("profit_factor", float("nan")),
        "bt_base_exp":  m_base.get("expectancy",    float("nan")),
        "wf_base_pf_all": _wf_avg_pf(wf_base, coverage_only=False),
        "wf_base_pf_co":  _wf_avg_pf(wf_base, coverage_only=True),
        "wf_base_exp_co": _wf_avg_exp(wf_base, coverage_only=True),
        "wf_base_tr_co":  _wf_trades(wf_base, coverage_only=True),
        # 15m-AB
        "bt_ab_n":      m_ab.get("n_trades",     0) if m_ab else None,
        "bt_ab_pf":     m_ab.get("profit_factor", float("nan")) if m_ab else None,
        "bt_ab_exp":    m_ab.get("expectancy",    float("nan")) if m_ab else None,
        "wf_ab_pf_all": _wf_avg_pf(wf_ab,  coverage_only=False) if wf_ab else None,
        "wf_ab_pf_co":  _wf_avg_pf(wf_ab,  coverage_only=True)  if wf_ab else None,
        "wf_ab_exp_co": _wf_avg_exp(wf_ab,  coverage_only=True)  if wf_ab else None,
        "wf_ab_tr_co":  _wf_trades(wf_ab,  coverage_only=True)  if wf_ab else None,
        "wf_rows_base": wf_base,
        "wf_rows_ab":   wf_ab,
    }


def _run_wf(
    df:           pd.DataFrame,
    feature_cols: list[str],
    variant:      str,
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


def _print_wf_rows(
    symbol:   str,
    wf_base:  list[dict],
    wf_ab:    list[dict] | None,
    df_15m:   pd.DataFrame | None,
) -> None:
    col_w = 22
    lbl_w = 28
    variants = ["Baseline"] + (["15m-AB"] if wf_ab else [])
    sep = "─" * (lbl_w + col_w * len(variants) + 2)

    print(f"\n  Walk-Forward  ({symbol})")
    print(f"  {sep}")
    print(f"  {'Window':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in variants))
    print(f"  {sep}")

    for wi, r_base in enumerate(wf_base):
        cov  = "(15m)" if r_base.get("has_15m") else "(no 15m)"
        lbl  = (f"Win{wi+1} [{r_base['train_frac']*100:.0f}%–"
                f"{r_base['test_frac']*100:.0f}%] {cov}")
        row  = f"  {lbl:<{lbl_w}}"

        n_b  = r_base["n_trades"]
        pf_b = r_base.get("profit_factor", float("nan"))
        row += f"{'%dtr PF=%s' % (n_b, _fmt(pf_b, '.2f')):>{col_w}}"

        if wf_ab:
            r_ab  = wf_ab[wi]
            n_ab  = r_ab["n_trades"]
            pf_ab = r_ab.get("profit_factor", float("nan"))
            sk    = r_ab.get("n_skipped", 0)
            cell  = f"{n_ab}tr PF={_fmt(pf_ab, '.2f')}"
            if r_ab.get("has_15m") and sk:
                cell += f" sk={sk}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {sep}")
    for label, fn, cov_only, fmt_str in [
        ("WF avg PF  (all)",    _wf_avg_pf,  False, ".3f"),
        ("WF avg PF  (15m)",    _wf_avg_pf,  True,  ".3f"),
        ("WF avg exp (15m) $",  _wf_avg_exp, True,  "+.2f"),
        ("WF trades  (15m)",    _wf_trades,  True,  "d"),
    ]:
        row = f"  {label:<{lbl_w}}"
        row += f"{_fmt(fn(wf_base, coverage_only=cov_only), fmt_str):>{col_w}}"
        if wf_ab:
            row += f"{_fmt(fn(wf_ab, coverage_only=cov_only), fmt_str):>{col_w}}"
        print(row)
    print(f"  {sep}")


# ── Summary + verdict ─────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    sep = "=" * 80

    print(f"\n\n  {sep}")
    print(f"  CROSS-ASSET VALIDATION SUMMARY — 15m-AB Execution Layer")
    print(f"  {sep}")

    # Table header
    col_w = 12
    lbl_w = 12
    cols  = ["BT PF", "BT exp$", "WF PF*", "WF exp$*", "WF tr*", "OOS>1?"]
    n_cols = len(cols)
    print(f"\n  {'Symbol':<{lbl_w}} {'Mode':<10}" +
          "".join(f"{c:>{col_w}}" for c in cols))
    print(f"  {'─'*(lbl_w + 10 + col_w * n_cols)}")

    full_15m = []   # assets where 15m-AB was run
    baseline_ref = []

    for r in results:
        sym = r["symbol"]

        # Baseline row
        print(f"  {sym:<{lbl_w}} {'Baseline':<10}"
              f"{_fmt(r['bt_base_pf'],   '.3f'):>{col_w}}"
              f"{_fmt(r['bt_base_exp'],  '+.2f'):>{col_w}}"
              f"{_fmt(r['wf_base_pf_co'], '.3f'):>{col_w}}"
              f"{_fmt(r['wf_base_exp_co'],'+.2f'):>{col_w}}"
              f"{_fmt(r['wf_base_tr_co'], 'd'):>{col_w}}"
              f"{'—':>{col_w}}")

        if r["has_15m"]:
            ab_wf  = r.get("wf_ab_pf_co", float("nan"))
            oos_ok = "YES" if (isinstance(ab_wf, float) and ab_wf > 1.0) else "NO"
            print(f"  {'':<{lbl_w}} {'15m-AB':<10}"
                  f"{_fmt(r['bt_ab_pf'],   '.3f'):>{col_w}}"
                  f"{_fmt(r['bt_ab_exp'],  '+.2f'):>{col_w}}"
                  f"{_fmt(r['wf_ab_pf_co'], '.3f'):>{col_w}}"
                  f"{_fmt(r['wf_ab_exp_co'],'+.2f'):>{col_w}}"
                  f"{_fmt(r['wf_ab_tr_co'], 'd'):>{col_w}}"
                  f"{oos_ok:>{col_w}}")
            full_15m.append(r)
        else:
            print(f"  {'':<{lbl_w}} {'15m-AB':<10}"
                  f"{'— no 15m data —':>{col_w*3}}")
            baseline_ref.append(r)

        print(f"  {'─'*(lbl_w + 10 + col_w * n_cols)}")

    print(f"  (* = 15m-covered WF windows only)")

    # Per-window breakdown for 15m assets
    print(f"\n  Per-window WF breakdown  (15m-covered assets):")
    win_sep = "─" * 70
    for r in full_15m:
        sym = r["symbol"]
        print(f"\n  {sym}")
        for wi, (rb, ra) in enumerate(zip(r["wf_rows_base"], r["wf_rows_ab"])):
            cov   = "(15m)" if rb.get("has_15m") else "(no 15m)"
            pf_b  = _fmt(rb.get("profit_factor"), ".3f")
            pf_ab = _fmt(ra.get("profit_factor"), ".3f")
            n_b   = rb["n_trades"]
            n_ab  = ra["n_trades"]
            sk    = ra.get("n_skipped", 0)
            print(f"    Win{wi+1} [{rb['train_frac']*100:.0f}–{rb['test_frac']*100:.0f}%] {cov:<10}"
                  f"  Baseline: {n_b:>3}tr PF={pf_b}"
                  f"  |  15m-AB: {n_ab:>3}tr PF={pf_ab}"
                  + (f"  sk={sk}" if ra.get("has_15m") and sk else ""))

    # Verdict
    print(f"\n  {sep}")
    print(f"  VERDICT")
    print(f"  {sep}")

    ab_pfs = [r["wf_ab_pf_co"] for r in full_15m
              if isinstance(r.get("wf_ab_pf_co"), float) and not math.isnan(r["wf_ab_pf_co"])]
    ab_n   = [r["wf_ab_tr_co"] for r in full_15m
              if isinstance(r.get("wf_ab_tr_co"), (int, float))]

    n_above_1 = sum(1 for pf in ab_pfs if pf > 1.0)
    n_total   = len(ab_pfs)
    avg_pf    = float(np.mean(ab_pfs)) if ab_pfs else float("nan")
    min_trades = min(ab_n) if ab_n else 0

    print(f"\n  15m-AB assets evaluated : {n_total}")
    print(f"  Assets with WF PF > 1.0  : {n_above_1} / {n_total}")
    print(f"  Cross-asset avg WF PF    : {_fmt(avg_pf, '.3f')}")
    print(f"  Min WF trades (covered)  : {min_trades}")

    if n_total == 0:
        verdict = "INCONCLUSIVE — no assets had 15m data"
        action  = "Acquire 15m data for ETH or ADA to extend validation."
    elif n_above_1 == n_total and avg_pf >= 1.0 and min_trades >= 60:
        verdict = "STRONG — 15m-AB clears WF PF > 1.0 on ALL tested assets"
        action  = ("Treat 15m-AB as the MVP execution architecture.\n"
                   "  Recommended next step: integrate into main pipeline with 4H context gate.")
    elif n_above_1 >= 1 and avg_pf >= 0.90:
        verdict = f"MODERATE — WF PF > 1.0 on {n_above_1}/{n_total} assets, avg={avg_pf:.3f}"
        action  = ("15m-AB shows directional improvement but not universally above 1.0.\n"
                   "  Treat as the best current architecture with moderate OOS confidence.\n"
                   "  Recommended: acquire more asset data before full deployment.")
    elif avg_pf >= 0.80:
        verdict = f"WEAK — avg WF PF = {avg_pf:.3f}, does not clear 1.0 on most assets"
        action  = ("15m-AB improves over baseline but OOS edge is not robust.\n"
                   "  Do not deploy. Investigate Approach C or alternative architectures.")
    else:
        verdict = f"REJECTED — avg WF PF = {avg_pf:.3f} (below baseline range)"
        action  = "15m-AB does not transfer. The BTC result was likely data-specific."

    print(f"\n  Result : {verdict}")
    print(f"  Action : {action}")
    print(f"  {sep}\n")

    # Save CSV
    _save_summary(results)


def _save_summary(results: list[dict]) -> None:
    rows = []
    for r in results:
        rows.append({
            "symbol":         r["symbol"],
            "has_15m":        r["has_15m"],
            "test_start":     r["test_start"],
            "test_end":       r["test_end"],
            "bt_base_pf":     r["bt_base_pf"],
            "bt_base_exp":    r["bt_base_exp"],
            "bt_ab_pf":       r["bt_ab_pf"],
            "bt_ab_exp":      r["bt_ab_exp"],
            "wf_base_pf_all": r["wf_base_pf_all"],
            "wf_base_pf_co":  r["wf_base_pf_co"],
            "wf_ab_pf_all":   r["wf_ab_pf_all"],
            "wf_ab_pf_co":    r["wf_ab_pf_co"],
            "wf_ab_exp_co":   r["wf_ab_exp_co"],
            "wf_ab_tr_co":    r["wf_ab_tr_co"],
        })
    out = os.path.join(config.EXPERIMENTS_DIR, "multiasset_15m_validation.csv")
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"  Results saved: {out}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*80}")
    print(f"  ARKAD MRK — Cross-Asset 15m-AB Robustness Validation")
    print(f"  Hypothesis: 15m-AB (momentum filter + pullback entry) is a")
    print(f"  transferable edge, not a BTC-2024 artefact.")
    print(f"{'='*80}")

    results = []
    for cfg in ASSETS:
        t0 = time.time()
        r  = run_asset(
            symbol  = cfg["symbol"],
            data_1h = cfg["data_1h"],
            data_5m = cfg["data_5m"],
            note    = cfg["note"],
        )
        print(f"  [{cfg['symbol']}] completed in {time.time()-t0:.0f}s")
        results.append(r)

    _print_summary(results)


if __name__ == "__main__":
    main()

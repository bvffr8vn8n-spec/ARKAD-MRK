"""
experiments/execution_15m_test.py — 15m Execution Layer Validation (A / B / A+B).

Compares four variants:
  Baseline : 1H signal, enter at 1H bar close (current behaviour)
  15m-B    : pullback entry (fallback to baseline if no pullback — no trades lost)
  15m-A    : momentum filter (skip if <2 of first 4 15m bars align — hard filter)
  15m-AB   : A first (hard filter), then B on surviving signals (pullback price)

Primary metric  : WF avg PF (15m-covered windows only)
Secondary metric: WF avg expectancy, trade reduction vs baseline
Coverage note   : 15m data starts 2024-01-01.
  Win1 (60-70%): 2023-05 to 2023-10 — NO 15m data → all variants = baseline.
  Win2 (70-80%): partial coverage (~2023-10 to 2024-03).
  Win3 (80-90%) + main test: full coverage. Primary evaluation window.

Usage
-----
  python experiments/execution_15m_test.py --data data/BTCUSDT_1h_4y.csv \\
                                            --data15m data/BTCUSDT_15m.csv
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
    load_15m_data,
    annotate_signals_A,
    annotate_signals_B,
    annotate_signals_AB,
    coverage_stats,
    print_coverage_stats,
)
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


_VARIANTS = ["Baseline", "15m-B", "15m-A", "15m-AB"]
_WF_KEYS  = ["n_trades", "profit_factor", "win_rate", "expectancy", "max_drawdown"]

_15M_START = pd.Timestamp("2024-01-01")


# ── Variant preparation ───────────────────────────────────────────────────────

def _prepare(
    signals_df: pd.DataFrame,
    variant:    str,
    df_15m:     pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply execution logic for `variant` and return (df_for_backtest, stats).

    stats keys (variant-specific, may be 0/NaN):
      n_pullback, n_fallback   (15m-B)
      n_skipped, n_passed_15m  (15m-A)
    """
    if variant == "Baseline":
        return signals_df.copy(), {}

    if variant == "15m-B":
        annotated  = annotate_signals_B(signals_df, df_15m)
        stats      = coverage_stats(signals_df, annotated, df_15m)
        return annotated, {
            "n_pullback": stats["pullback_entries"],
            "n_fallback": stats["baseline_fallback"],
            "n_skipped":  0,
        }

    # 15m-A
    if variant == "15m-A":
        annotated   = annotate_signals_A(signals_df, df_15m)
        n_skipped   = int(((signals_df["signal"] != 0) &
                           (annotated["signal_15m_A"] == 0)).sum())
        df_bt       = annotated.copy()
        df_bt["signal"] = df_bt["signal_15m_A"]
        return df_bt, {
            "n_pullback": 0,
            "n_fallback": 0,
            "n_skipped":  n_skipped,
        }

    # 15m-AB: A filters first, B seeks pullback on surviving signals
    annotated   = annotate_signals_AB(signals_df, df_15m)
    n_skipped   = int(((signals_df["signal"] != 0) &
                       (annotated["signal_15m_A"] == 0)).sum())
    n_pullback  = int(annotated["entry_price_15m"].notna().sum())
    df_bt       = annotated.copy()
    df_bt["signal"] = df_bt["signal_15m_A"]
    return df_bt, {
        "n_pullback": n_pullback,
        "n_fallback": 0,
        "n_skipped":  n_skipped,
    }


# ── Backtest helpers ──────────────────────────────────────────────────────────

def _has_15m_coverage(test_df: pd.DataFrame, threshold: float = 0.5) -> bool:
    return float((test_df.index >= _15M_START).mean()) >= threshold


def _backtest_variant(
    signals_df: pd.DataFrame,
    variant:    str,
    df_15m:     pd.DataFrame,
) -> dict:
    df_bt, _   = _prepare(signals_df, variant, df_15m)
    trades, eq = run_backtest(df_bt)
    m = compute_metrics(trades, eq)
    if "error" in m:
        return {k: (0 if k == "n_trades" else float("nan")) for k in _WF_KEYS}
    return m


def _run_wf(
    df:           pd.DataFrame,
    feature_cols: list[str],
    variant:      str,
    df_15m:       pd.DataFrame,
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
        empty.update({
            "train_frac": train_frac, "test_frac": min(test_frac, 1.0),
            "has_15m": False, "n_pullback": 0, "n_fallback": 0, "n_skipped": 0,
        })

        if len(train_slice) < 50 or len(test_slice) < 5:
            rows.append(empty)
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = fit_model(train_slice, feature_cols)

        signals_wf = apply_signals(model, feature_cols, test_slice)
        has_cov    = _has_15m_coverage(test_slice)
        df_bt, xstats = _prepare(signals_wf, variant, df_15m)

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
            row["n_pullback"] = xstats.get("n_pullback", 0)
            row["n_fallback"] = xstats.get("n_fallback", 0)
            row["n_skipped"]  = xstats.get("n_skipped",  0)
            rows.append(row)
    return rows


# ── Aggregation helpers ───────────────────────────────────────────────────────

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


def _wf_total_trades(rows: list[dict], coverage_only: bool = False) -> int:
    return sum(r["n_trades"] for r in rows
               if not coverage_only or r.get("has_15m", False))


def _fmt(v, fmt="", fallback="N/A"):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return fallback
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_backtest_table(bt: dict) -> None:
    col_w = 11
    lbl_w = 22
    sep   = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)

    print(f"\n  Backtest Metrics  (20% test set — full 15m coverage)")
    print(f"  {sep}")
    print(f"  {'Metric':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS))
    print(f"  {sep}")
    for label, key, fmt_str, suffix in [
        ("Trades",        "n_trades",      "d",    ""),
        ("Win rate",      "win_rate",      ".1%",  ""),
        ("Profit factor", "profit_factor", ".3f",  ""),
        ("Expectancy $",  "expectancy",    "+.2f", ""),
        ("Max drawdown",  "max_drawdown",  ".2f",  "%"),
        ("Total return",  "total_return",  ".2f",  "%"),
    ]:
        row = f"  {label:<{lbl_w}}"
        for v in _VARIANTS:
            val = bt[v].get(key)
            row += f"{_fmt(val, fmt_str) + suffix:>{col_w}}"
        print(row)
    print(f"  {sep}")


def _print_signal_funnel(bt: dict, n_raw: int, n_test_days: float) -> None:
    col_w = 11
    lbl_w = 30
    sep   = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)

    print(f"\n  Signal Funnel  ({n_test_days:.0f} test days | raw signals = {n_raw})")
    print(f"  {sep}")
    print(f"  {'Stage':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS))
    print(f"  {sep}")
    print(f"  {'Raw signals':<{lbl_w}}" +
          "".join(f"{n_raw:>{col_w},}" for _ in _VARIANTS))
    row = f"  {'After execution filter':<{lbl_w}}"
    for v in _VARIANTS:
        n   = int(bt[v].get("n_trades", 0))
        pct = n / n_raw * 100 if n_raw > 0 else 0.0
        row += f"{f'{n} ({pct:.0f}%)':>{col_w}}"
    print(row)
    print(f"  {sep}")


def _print_wf_table(wf: dict) -> None:
    col_w     = 18
    lbl_w     = 28
    n_windows = len(config.WF_TRAIN_FRACTIONS)
    sep       = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)

    print(f"\n  Walk-Forward Results  ({n_windows} windows)")
    print(f"  {sep}")
    print(f"  {'Window':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS))
    print(f"  {sep}")

    for wi in range(n_windows):
        rows = {v: wf[v][wi] for v in _VARIANTS}
        r0   = rows["Baseline"]
        cov  = "(15m)" if r0.get("has_15m") else "(no 15m)"
        lbl  = (f"Win{wi+1} [{r0['train_frac']*100:.0f}%–"
                f"{r0['test_frac']*100:.0f}%] {cov}")
        row  = f"  {lbl:<{lbl_w}}"
        for v in _VARIANTS:
            r  = rows[v]
            n  = int(r["n_trades"])
            pf = r.get("profit_factor", float("nan"))
            pb = r.get("n_pullback", 0)
            sk = r.get("n_skipped",  0)
            cell = f"{n}tr PF={_fmt(pf, '.2f')}"
            if v == "15m-B" and r.get("has_15m") and pb > 0:
                cell += f" pb={pb}"
            if v == "15m-A" and r.get("has_15m") and sk > 0:
                cell += f" skip={sk}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {sep}")

    # Aggregate block
    agg_rows = [
        ("WF avg PF  (all windows)",  "pf",  False),
        ("WF avg PF  (15m only)   ",  "pf",  True),
        ("WF avg exp (all windows)",  "exp", False),
        ("WF avg exp (15m only)   ",  "exp", True),
        ("WF trades  (all windows)",  "tr",  False),
        ("WF trades  (15m only)   ",  "tr",  True),
    ]
    for label, kind, cov_only in agg_rows:
        row = f"  {label:<{lbl_w}}"
        for v in _VARIANTS:
            if kind == "pf":
                val = _wf_avg_pf(wf[v], coverage_only=cov_only)
                row += f"{_fmt(val, '.3f'):>{col_w}}"
            elif kind == "exp":
                val = _wf_avg_exp(wf[v], coverage_only=cov_only)
                row += f"{_fmt(val, '+.2f'):>{col_w}}"
            else:
                val = _wf_total_trades(wf[v], coverage_only=cov_only)
                row += f"{_fmt(val, 'd'):>{col_w}}"
        print(row)
    print(f"  {sep}")


def _print_trade_reduction(bt: dict, n_raw: int) -> None:
    """Show trade reduction and its breakdown by cause."""
    col_w = 11
    lbl_w = 30
    sep   = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)

    print(f"\n  Trade Reduction Summary  (test set, raw signals = {n_raw})")
    print(f"  {sep}")
    print(f"  {'Metric':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS))
    print(f"  {sep}")

    # Use the raw test-set data cached in the main function
    for label, key in [
        ("Trades executed",     "n_trades"),
        ("WR",                  "win_rate"),
        ("PF",                  "profit_factor"),
        ("Exp $",               "expectancy"),
    ]:
        row = f"  {label:<{lbl_w}}"
        for v in _VARIANTS:
            val = bt[v].get(key)
            fmt = ".1%" if key == "win_rate" else (".3f" if "factor" in key else ("+.2f" if "exp" in key.lower() else "d"))
            row += f"{_fmt(val, fmt):>{col_w}}"
        print(row)
    print(f"  {sep}")


def _print_verdict(bt: dict, wf: dict, test_a_stats: dict, test_b_stats: dict) -> None:
    sep = "=" * 72

    base_wf = _wf_avg_pf(wf["Baseline"], coverage_only=True)
    base_pf = bt["Baseline"].get("profit_factor", float("nan"))
    base_exp= bt["Baseline"].get("expectancy",    float("nan"))
    base_n  = bt["Baseline"].get("n_trades",      0)
    a_wf    = _wf_avg_pf(wf["15m-A"],  coverage_only=True)
    ab_wf   = _wf_avg_pf(wf["15m-AB"], coverage_only=True)

    print(f"\n  {sep}")
    print(f"  VERDICT — 15m Execution Layer Comparison (A / B / A+B)")
    print(f"  {sep}")

    # Per-variant summary rows
    variant_meta = [
        ("15m-B",  "Approach B  (pullback entry)   "),
        ("15m-A",  "Approach A  (momentum filter)  "),
        ("15m-AB", "Approach A+B (A filter, B price)"),
    ]
    for v, label in variant_meta:
        v_wf  = _wf_avg_pf(wf[v], coverage_only=True)
        v_pf  = bt[v].get("profit_factor", float("nan"))
        v_exp = bt[v].get("expectancy",    float("nan"))
        v_n   = bt[v].get("n_trades",      0)
        d_wf  = v_wf  - base_wf  if not (math.isnan(v_wf)  or math.isnan(base_wf))  else float("nan")
        d_a   = v_wf  - a_wf     if not (math.isnan(v_wf)  or math.isnan(a_wf))     else float("nan")
        pct_kept = v_n / base_n * 100 if base_n > 0 else float("nan")

        print(f"\n  {label}")
        print(f"    WF avg PF (15m)  : {_fmt(v_wf, '.3f')}  "
              f"delta_vs_baseline={_fmt(d_wf, '+.3f')}"
              + (f"  delta_vs_A={_fmt(d_a, '+.3f')}" if v == "15m-AB" else ""))
        print(f"    BT PF / exp $    : {_fmt(v_pf, '.3f')} / {_fmt(v_exp, '+.2f')}"
              f"  vs baseline={_fmt(base_pf, '.3f')} / {_fmt(base_exp, '+.2f')}")
        print(f"    Trades (test)    : {v_n}/{base_n}  ({_fmt(pct_kept, '.0f')}% of baseline)")

        if math.isnan(d_wf):
            verdict = "INSUFFICIENT DATA"
        elif d_wf > 0.10:
            verdict = f"STRONG IMPROVEMENT ({d_wf:+.3f})"
        elif d_wf > 0.05:
            verdict = f"POSITIVE ({d_wf:+.3f})"
        elif d_wf > 0.01:
            verdict = f"MARGINAL ({d_wf:+.3f})"
        elif d_wf >= -0.01:
            verdict = f"NEUTRAL ({d_wf:+.3f})"
        else:
            verdict = f"NEGATIVE ({d_wf:+.3f})"
        print(f"    Result           : {verdict}")

    # Key question: does A+B beat A alone?
    print(f"\n  {'─'*68}")
    print(f"  Key question: does B add value on top of A?")
    if not (math.isnan(ab_wf) or math.isnan(a_wf)):
        d = ab_wf - a_wf
        if d > 0.03:
            print(f"  YES  — A+B WF PF ({ab_wf:.3f}) > A alone ({a_wf:.3f})  by {d:+.3f}")
            print(f"  Action: use 15m-AB as the primary execution layer.")
        elif d >= -0.03:
            print(f"  NEUTRAL — A+B WF PF ({ab_wf:.3f}) ≈ A alone ({a_wf:.3f})  (delta={d:+.3f})")
            print(f"  B adds price improvement but no meaningful OOS PF gain.")
            print(f"  Action: prefer 15m-A (simpler, fewer parameters).")
        else:
            print(f"  NO   — A+B WF PF ({ab_wf:.3f}) < A alone ({a_wf:.3f})  by {d:+.3f}")
            print(f"  B degrades quality when layered on top of A.")
            print(f"  Action: use 15m-A alone. Do not stack.")
    else:
        print(f"  Cannot compare — insufficient WF data.")

    # Win3 spotlight (most reliable OOS window)
    n_wf = len(config.WF_TRAIN_FRACTIONS)
    if n_wf >= 3:
        print(f"\n  Win3 spotlight (most reliable OOS window — full 15m coverage):")
        for v in _VARIANTS:
            r   = wf[v][2]  # Win3 = index 2
            pf  = r.get("profit_factor", float("nan"))
            n   = r["n_trades"]
            sk  = r.get("n_skipped",  0)
            pb  = r.get("n_pullback", 0)
            tag = ""
            if v == "15m-A"  and sk: tag = f"  skip={sk}"
            if v == "15m-AB" and sk: tag = f"  skip={sk} pb={pb}"
            print(f"    {v:<12} trades={n:>3}  PF={_fmt(pf, '.3f')}{tag}")

    print(f"  {sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str, data15m_path: str) -> None:
    print(f"\n{'='*72}")
    print(f"  ARKAD MRK — 15m Execution Layer Validation (Approach A + B)")
    print(f"  1H data  : {os.path.basename(data_path)}")
    print(f"  15m data : {os.path.basename(data15m_path)}")
    print(f"  Variants : {_VARIANTS}")
    print(f"  15m coverage from : {_15M_START.date()}")
    print(f"{'='*72}\n")

    # ── [1/4] Load + features ────────────────────────────────────────────────
    print("[1/4] Loading data and building features...")
    df     = load_ohlcv(data_path)
    df     = generate_features(df)
    df     = add_regime_columns(df)
    df     = add_session_column(df)
    df     = add_labels(df)
    df.dropna(inplace=True)
    df_15m = load_15m_data(data15m_path)

    print(f"      {len(df):,} bars after dropna")
    print(f"      15m: {len(df_15m):,} bars "
          f"({df_15m.index[0].date()} to {df_15m.index[-1].date()})\n")

    # ── [2/4] Train on 80% ───────────────────────────────────────────────────
    print("[2/4] Training model on 80% train set...")
    feature_cols = get_feature_columns(df)
    split        = int(len(df) * (1 - config.TEST_SIZE))
    train_df     = df.iloc[:split]
    test_df      = df.iloc[split:]
    n_test_days  = max((test_df.index[-1] - test_df.index[0]).days, 1)

    t0    = time.time()
    model = fit_model(train_df, feature_cols)
    print(f"      Trained in {time.time()-t0:.1f}s  |  "
          f"test: {len(test_df):,} bars (~{n_test_days}d)")

    signals_df = apply_signals(model, feature_cols, test_df)
    n_raw = int((signals_df["signal"] != 0).sum())

    # Pre-annotate all variants for coverage/stats reporting
    ann_b  = annotate_signals_B(signals_df, df_15m)
    ann_a  = annotate_signals_A(signals_df, df_15m)
    ann_ab = annotate_signals_AB(signals_df, df_15m)

    stats_b = coverage_stats(signals_df, ann_b, df_15m, window_label="main test (B)")
    print_coverage_stats(stats_b)

    n_15m_signals = int(((signals_df["signal"] != 0) &
                         (signals_df.index + pd.Timedelta(hours=1) >= _15M_START)).sum())

    n_skipped_a  = int(((signals_df["signal"] != 0) & (ann_a["signal_15m_A"]  == 0)).sum())
    n_skipped_ab = int(((signals_df["signal"] != 0) & (ann_ab["signal_15m_A"] == 0)).sum())
    n_pullback_ab= int(ann_ab["entry_price_15m"].notna().sum())

    print(f"\n  15m-A / 15m-AB Filter — main test  ({n_15m_signals} signals with 15m data)")
    print(f"  {'─'*56}")
    print(f"    15m-A  : filtered out {n_skipped_a:>3} / {n_15m_signals}"
          f"  ({_fmt(n_skipped_a/n_15m_signals*100 if n_15m_signals else float('nan'), '.0f')}%)"
          f"  kept {n_15m_signals - n_skipped_a}")
    print(f"    15m-AB : filtered out {n_skipped_ab:>3} / {n_15m_signals}"
          f"  ({_fmt(n_skipped_ab/n_15m_signals*100 if n_15m_signals else float('nan'), '.0f')}%)"
          f"  pullback overrides on survivors: {n_pullback_ab}")
    print(f"  {'─'*56}\n")

    test_a_stats  = {"n_skipped": n_skipped_a,  "total_raw": n_15m_signals}
    test_b_stats  = stats_b

    # ── [3/4] Backtest ───────────────────────────────────────────────────────
    print("[3/4] Running backtest for each variant...")
    bt: dict[str, dict] = {}
    for v in _VARIANTS:
        bt[v] = _backtest_variant(signals_df, v, df_15m)
        n   = bt[v].get("n_trades", 0)
        pf  = bt[v].get("profit_factor", float("nan"))
        exp = bt[v].get("expectancy",    float("nan"))
        print(f"      {v:<12} trades={n:>4}  PF={_fmt(pf, '.3f'):>6}  exp={_fmt(exp, '+.2f'):>7}")

    _print_signal_funnel(bt, n_raw, n_test_days)
    _print_backtest_table(bt)
    _print_trade_reduction(bt, n_raw)

    # ── [4/4] Walk-forward ───────────────────────────────────────────────────
    n_wf = len(config.WF_TRAIN_FRACTIONS)
    print(f"\n[4/4] Running walk-forward ({n_wf} windows × {len(_VARIANTS)} variants = "
          f"{n_wf * len(_VARIANTS)} fits)...")

    wf: dict[str, list[dict]] = {}
    for v in _VARIANTS:
        print(f"      {v} ...", end="", flush=True)
        t_wf = time.time()
        wf[v] = _run_wf(df, feature_cols, v, df_15m)

        # Per-window detail line
        for r in wf[v]:
            cov = "(15m)" if r.get("has_15m") else "(no 15m)"
            pb  = r.get("n_pullback", 0)
            sk  = r.get("n_skipped",  0)
            detail = f"pb={pb}" if (v == "15m-B" and pb > 0) else \
                     f"skip={sk}" if (v == "15m-A" and sk > 0) else ""
            print(f"\n        Win [{r['train_frac']*100:.0f}%-{r['test_frac']*100:.0f}%] "
                  f"{cov}  {detail}  trades={r['n_trades']}"
                  f"  PF={_fmt(r.get('profit_factor'), '.2f')}",
                  end="")

        total = sum(r["n_trades"] for r in wf[v])
        avg   = _wf_avg_pf(wf[v])
        print(f"\n      done in {time.time()-t_wf:.0f}s  |  total={total}  avg_PF={_fmt(avg, '.3f')}")

    _print_wf_table(wf)
    _print_verdict(bt, wf, test_a_stats, test_b_stats)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    rows_out = []
    for v in _VARIANTS:
        bt_m = bt[v]
        row  = {
            "variant":        v,
            "bt_trades":      bt_m.get("n_trades",     0),
            "bt_win_rate":    bt_m.get("win_rate",      float("nan")),
            "bt_pf":          bt_m.get("profit_factor", float("nan")),
            "bt_exp":         bt_m.get("expectancy",    float("nan")),
            "bt_max_dd":      bt_m.get("max_drawdown",  float("nan")),
            "wf_avg_pf_all":  _wf_avg_pf(wf[v], coverage_only=False),
            "wf_avg_pf_15m":  _wf_avg_pf(wf[v], coverage_only=True),
            "wf_avg_exp_all": _wf_avg_exp(wf[v], coverage_only=False),
            "wf_avg_exp_15m": _wf_avg_exp(wf[v], coverage_only=True),
            "wf_trades_all":  _wf_total_trades(wf[v], coverage_only=False),
            "wf_trades_15m":  _wf_total_trades(wf[v], coverage_only=True),
        }
        for wi, r in enumerate(wf[v]):
            row[f"wf_win{wi+1}_trades"]  = r["n_trades"]
            row[f"wf_win{wi+1}_pf"]      = r.get("profit_factor", float("nan"))
            row[f"wf_win{wi+1}_has15m"]  = r.get("has_15m", False)
            row[f"wf_win{wi+1}_pb"]      = r.get("n_pullback", 0)
            row[f"wf_win{wi+1}_skipped"] = r.get("n_skipped",  0)
        rows_out.append(row)

    out_path = os.path.join(config.EXPERIMENTS_DIR, "execution_15m_results.csv")
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARKAD MRK — 15m Execution Layer Validation (A + B)"
    )
    parser.add_argument("--data",    required=True, help="Path to 1H OHLCV CSV")
    parser.add_argument("--data15m", required=True, help="Path to 15m OHLCV CSV")
    args = parser.parse_args()
    main(args.data, args.data15m)

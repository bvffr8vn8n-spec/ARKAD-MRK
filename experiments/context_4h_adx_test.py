"""
experiments/context_4h_adx_test.py — Design A: ADX-gated 4H context validation.

Compares four signal variants side-by-side:
  Baseline       : no 4H filter
  CrossoverRef   : proven SMA20/50 crossover strict (WF PF = 0.988 benchmark)
  ADX-Strict     : price-vs-SMA-slow + ADX > 20, neutral blocked
  ADX-Range      : price-vs-SMA-slow + ADX > 20, neutral (range) allowed for both

Success criteria
----------------
  Primary   : ADX variant WF avg PF > 0.988 (beats proven crossover)
  Secondary : ADX variant WF avg PF >= 0.988 WITH more total trades
              (same quality, higher frequency — directly addresses 0.23 t/day)

Design A hypothesis
-------------------
  ADX identifies whether a trend is statistically present, independent of SMA
  crossover lag.  In strict mode this may achieve the same quality filter earlier
  in a trend move.  In adx_range mode, confirmed-range bars (ADX < 20) are
  allowed for both directions — recovering the mean-reversion setups that the
  crossover neutral zone blocks entirely.

Usage
-----
  python experiments/context_4h_adx_test.py --data data/BTCUSDT_1h_4y.csv
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
from features.context_4h import add_4h_context, apply_4h_context_filter
from features.context_4h_adx import (
    add_4h_context_adx,
    apply_4h_context_adx_filter,
    print_4h_adx_bias_stats,
)
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


# ── Variant definitions ───────────────────────────────────────────────────────

# Each entry: (label, filter_fn)
# filter_fn(signals_df) → filtered signals_df with 'signal' overwritten
def _filter_none(df):
    return df

def _filter_crossover_strict(df):
    df = apply_4h_context_filter(df, mode="strict")
    df["signal"] = df["signal_4h_filtered"]
    return df

def _filter_adx_strict(df):
    df = apply_4h_context_adx_filter(df, mode="strict")
    df["signal"] = df["signal_4h_adx_filtered"]
    return df

def _filter_adx_range(df):
    df = apply_4h_context_adx_filter(df, mode="adx_range")
    df["signal"] = df["signal_4h_adx_filtered"]
    return df

_VARIANTS = {
    "Baseline":      _filter_none,
    "CrossoverRef":  _filter_crossover_strict,
    "ADX-Strict":    _filter_adx_strict,
    "ADX-Range":     _filter_adx_range,
}

_WF_METRIC_KEYS = ["n_trades", "profit_factor", "win_rate", "expectancy", "max_drawdown"]

# Known benchmark from proven crossover strict
_CROSSOVER_WF_PF = 0.988


# ── Helpers ───────────────────────────────────────────────────────────────────

def _backtest_variant(signals_df: pd.DataFrame, filter_fn) -> dict:
    df     = filter_fn(signals_df.copy())
    trades, equity = run_backtest(df)
    m = compute_metrics(trades, equity)
    if "error" in m:
        return {k: (0 if k == "n_trades" else float("nan"))
                for k in ["n_trades", "win_rate", "profit_factor",
                           "expectancy", "max_drawdown", "total_return"]}
    return m


def _run_wf(df: pd.DataFrame, feature_cols: list[str], filter_fn) -> list[dict]:
    n    = len(df)
    rows = []
    for train_frac in config.WF_TRAIN_FRACTIONS:
        test_frac  = train_frac + config.WF_TEST_FRACTION
        train_end  = int(n * train_frac)
        test_end   = min(int(n * test_frac), n)
        purge_end  = train_end - config.FORWARD_RETURN_WINDOW

        train_slice = df.iloc[0:purge_end]
        test_slice  = df.iloc[train_end:test_end]

        empty = {k: (0 if k == "n_trades" else float("nan"))
                 for k in _WF_METRIC_KEYS}
        empty.update({"train_frac": train_frac,
                      "test_frac":  min(test_frac, 1.0)})

        if len(train_slice) < 50 or len(test_slice) < 5:
            rows.append(empty)
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = fit_model(train_slice, feature_cols)

        signals_wf = apply_signals(model, feature_cols, test_slice)
        signals_wf = filter_fn(signals_wf.copy())

        trades, equity = run_backtest(signals_wf)
        m = compute_metrics(trades, equity)

        if "error" in m:
            rows.append(empty)
        else:
            row = {k: m.get(k, float("nan")) for k in _WF_METRIC_KEYS}
            row["n_trades"] = int(row["n_trades"] or 0)
            row.update({"train_frac": train_frac,
                        "test_frac":  min(test_frac, 1.0)})
            rows.append(row)
    return rows


def _wf_avg_pf(rows):
    vals = [r["profit_factor"] for r in rows
            if r["n_trades"] > 0 and not math.isnan(r.get("profit_factor", float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def _wf_avg_exp(rows):
    vals = [r["expectancy"] for r in rows
            if r["n_trades"] > 0 and not math.isnan(r.get("expectancy", float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def _wf_worst_dd(rows):
    vals = [r["max_drawdown"] for r in rows
            if not math.isnan(r.get("max_drawdown", float("nan")))]
    return float(max(vals)) if vals else float("nan")


def _fmt(v, fmt="", fallback="N/A"):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return fallback
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_bias_comparison(signals_df: pd.DataFrame, n_days: float) -> None:
    n_bars = len(signals_df)
    col_w  = 14
    lbl_w  = 14
    sep    = "-" * (lbl_w + col_w * 2 + 4)

    print(f"\n  Bias Distribution Comparison  ({n_bars:,} test bars | ~{n_days:.0f} days)")
    print(f"  {sep}")
    print(f"  {'State':<{lbl_w}}{'Crossover':>{col_w}}{'ADX':>{col_w}}")
    print(f"  {sep}")
    for label in ["bull", "neutral", "bear"]:
        c_cnt = int((signals_df.get("bias_4h",     pd.Series()) == label).sum()) \
                if "bias_4h"     in signals_df.columns else 0
        a_cnt = int((signals_df.get("bias_4h_adx", pd.Series()) == label).sum()) \
                if "bias_4h_adx" in signals_df.columns else 0
        c_pct = c_cnt / n_bars * 100
        a_pct = a_cnt / n_bars * 100
        print(f"  {label:<{lbl_w}}"
              f"{f'{c_cnt:,} ({c_pct:.1f}%)':>{col_w}}"
              f"{f'{a_cnt:,} ({a_pct:.1f}%)':>{col_w}}")
    print(f"  {sep}")


def _print_signal_funnel(signals_df: pd.DataFrame, n_days: float) -> None:
    n_bars = len(signals_df)
    raw_n  = int((signals_df["signal"] != 0).sum())
    col_w  = 14
    lbl_w  = 28
    sep    = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)

    print(f"\n  Signal Funnel  ({n_bars:,} test bars | ~{n_days:.0f} days)")
    print(f"  {sep}")
    hdr = f"  {'Stage':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS)
    print(hdr)
    print(f"  {sep}")
    print(f"  {'Raw model signals':<{lbl_w}}" +
          "".join(f"{raw_n:>{col_w},}" for _ in _VARIANTS))

    row = f"  {'After context filter':<{lbl_w}}"
    for name, filter_fn in _VARIANTS.items():
        if name == "Baseline":
            row += f"{'—':>{col_w}}"
            continue
        tmp = filter_fn(signals_df.copy())
        # count the final signal (already written back to 'signal' by filter_fn)
        n_fil = int((tmp["signal"] != 0).sum())
        pct   = n_fil / raw_n * 100 if raw_n > 0 else 0.0
        row  += f"{f'{n_fil:,} ({pct:.0f}%)':>{col_w}}"
    print(row)
    print(f"  {sep}")


def _print_backtest_table(bt_results: dict) -> None:
    col_w = 14
    lbl_w = 22
    sep   = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)

    print(f"\n  Backtest Metrics  (20% test set)")
    print(f"  {sep}")
    hdr = f"  {'Metric':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS)
    print(hdr)
    print(f"  {sep}")

    rows_def = [
        ("Trades",        "n_trades",      "d",    ""),
        ("Win rate",      "win_rate",      ".1%",  ""),
        ("Profit factor", "profit_factor", ".3f",  ""),
        ("Expectancy $",  "expectancy",    "+.2f", ""),
        ("Max drawdown",  "max_drawdown",  ".2f",  "%"),
        ("Total return",  "total_return",  ".2f",  "%"),
    ]
    for label, key, fmt_str, suffix in rows_def:
        row = f"  {label:<{lbl_w}}"
        for name in _VARIANTS:
            val = bt_results[name].get(key)
            row += f"{_fmt(val, fmt_str) + suffix:>{col_w}}"
        print(row)
    print(f"  {sep}")


def _print_wf_table(wf_results: dict) -> None:
    col_w     = 18
    lbl_w     = 22
    sep       = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)
    n_windows = len(config.WF_TRAIN_FRACTIONS)

    print(f"\n  Walk-Forward Results  ({n_windows} expanding windows)")
    print(f"  {sep}")
    hdr = f"  {'Window':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS)
    print(hdr)
    print(f"  {sep}")

    for wi in range(n_windows):
        rows_for_window = {name: wf_results[name][wi] for name in _VARIANTS}
        r0  = next(iter(rows_for_window.values()))
        lbl = f"Win {wi+1} [{r0['train_frac']*100:.0f}%–{r0['test_frac']*100:.0f}%]"
        row = f"  {lbl:<{lbl_w}}"
        for name in _VARIANTS:
            r  = rows_for_window[name]
            n  = int(r["n_trades"])
            pf = r.get("profit_factor", float("nan"))
            cell = f"{n} tr  PF={pf:.2f}" if n > 0 and not math.isnan(pf) \
                   else "0 tr  PF=N/A"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {sep}")

    agg_rows = [
        ("WF avg PF",    _wf_avg_pf,  ".3f"),
        ("WF avg exp $", _wf_avg_exp, "+.2f"),
        ("WF worst DD%", _wf_worst_dd, ".1f"),
    ]
    for label, fn, fmt_str in agg_rows:
        row = f"  {label:<{lbl_w}}"
        for name in _VARIANTS:
            row += f"{_fmt(fn(wf_results[name]), fmt_str):>{col_w}}"
        print(row)
    print(f"  {sep}")


def _print_verdict(wf_results: dict) -> None:
    sep = "=" * 72

    crossover_pf = _wf_avg_pf(wf_results["CrossoverRef"])
    if math.isnan(crossover_pf):
        crossover_pf = _CROSSOVER_WF_PF

    crossover_trades = sum(r["n_trades"] for r in wf_results["CrossoverRef"])

    print(f"\n  {sep}")
    print(f"  HYPOTHESIS VERDICT — Design A (ADX gate)")
    print(f"  {sep}")
    print(f"  Primary   : ADX variant WF avg PF > {crossover_pf:.3f} (beats crossover)")
    print(f"  Secondary : ADX variant WF avg PF >= {crossover_pf:.3f}"
          f" with more WF trades ({crossover_trades} crossover trades)")
    print(f"  CrossoverRef WF avg PF : {_fmt(crossover_pf, '.3f')}"
          f"  (WF total trades = {crossover_trades})")

    adx_names = [n for n in _VARIANTS if n.startswith("ADX")]
    best_name  = None
    best_pf    = float("-inf")
    for name in adx_names:
        pf = _wf_avg_pf(wf_results[name])
        if not math.isnan(pf) and pf > best_pf:
            best_pf   = pf
            best_name = name

    if best_name is None or math.isnan(best_pf):
        print(f"\n  Result : FAILED — no ADX variant produced a valid WF PF.")
        print(f"  Action : Design A does not improve on crossover. Investigate Design C (1D layer).")
    else:
        best_trades = sum(r["n_trades"] for r in wf_results[best_name])
        delta_pf    = best_pf - crossover_pf
        delta_trades = best_trades - crossover_trades

        print(f"  Best ADX variant      : {best_name}"
              f"  WF avg PF = {best_pf:.3f}"
              f"  (delta PF = {delta_pf:+.3f},"
              f"  delta trades = {delta_trades:+d})")

        if best_pf > crossover_pf:
            print(f"\n  Result : CONFIRMED — ADX-gate beats crossover PF.")
            print(f"  Action : Set CONTEXT_4H_ADX variant as primary context gate.")
            print(f"           Update USE_4H_CONTEXT path in main.py or create")
            print(f"           a dedicated USE_4H_ADX_CONTEXT flag in config.py.")
        elif best_pf >= crossover_pf - 0.01 and delta_trades > 0:
            print(f"\n  Result : EQUIVALENT QUALITY, MORE FREQUENCY — secondary criterion met.")
            print(f"  Action : Prefer ADX variant if trade frequency is the bottleneck.")
            print(f"           Crossover remains valid if quality is the primary concern.")
        else:
            pf_gap = crossover_pf - best_pf
            print(f"\n  Result : INCONCLUSIVE — ADX variant is {pf_gap:.3f} PF below crossover.")
            print(f"  Action : Crossover remains the recommended context gate (WF PF={crossover_pf:.3f}).")
            print(f"           Consider Design C (1D macro layer) as the next experiment.")

    print(f"  {sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str) -> None:
    print(f"\n{'='*72}")
    print(f"  ARKAD MRK — 4H ADX Context Validation (Design A)")
    print(f"  Data        : {os.path.basename(data_path)}")
    print(f"  SMA-slow    : {config.CONTEXT_4H_SMA_SLOW}x4H"
          f" ({config.CONTEXT_4H_SMA_SLOW * 4} 1H bars)")
    print(f"  ADX window  : {config.CONTEXT_4H_ADX_WINDOW} 4H bars"
          f"  threshold={config.CONTEXT_4H_ADX_THRESHOLD}")
    print(f"  Crossover   : SMA{config.CONTEXT_4H_SMA_FAST}/SMA{config.CONTEXT_4H_SMA_SLOW}"
          f"  (benchmark WF PF = {_CROSSOVER_WF_PF})")
    print(f"  Variants    : {list(_VARIANTS.keys())}")
    print(f"{'='*72}\n")

    # ── [1/4] Load + features + both context layers ───────────────────────────
    print("[1/4] Loading data and building features...")
    df = load_ohlcv(data_path)
    df = generate_features(df)
    df = add_4h_context(df)           # adds bias_4h     (crossover reference)
    df = add_4h_context_adx(df)       # adds bias_4h_adx (Design A)
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    n_buy  = (df["label"] ==  1).sum()
    n_sell = (df["label"] == -1).sum()
    n_neut = (df["label"] ==  0).sum()
    print(f"      {len(df):,} bars after dropna  |  "
          f"buy={n_buy:,}  sell={n_sell:,}  neutral={n_neut:,}")

    # Print both bias distributions for comparison
    n_bars = len(df)
    print(f"\n  Bias Distribution (full dataset, {n_bars:,} bars)")
    print(f"  {'─'*52}")
    print(f"  {'State':<12}{'Crossover':>18}{'ADX':>18}")
    print(f"  {'─'*52}")
    for label in ["bull", "neutral", "bear"]:
        c = int((df["bias_4h"]     == label).sum()); cp = c / n_bars * 100
        a = int((df["bias_4h_adx"] == label).sum()); ap = a / n_bars * 100
        print(f"  {label:<12}{f'{c:,} ({cp:.1f}%)':>18}{f'{a:,} ({ap:.1f}%)':>18}")
    print(f"  {'─'*52}\n")

    # ── [2/4] Train on 80% ────────────────────────────────────────────────────
    print("[2/4] Training model on 80% train set...")
    feature_cols = get_feature_columns(df)
    split        = int(len(df) * (1 - config.TEST_SIZE))
    train_df     = df.iloc[:split]
    test_df      = df.iloc[split:]
    n_test_days  = max((test_df.index[-1] - test_df.index[0]).days, 1)

    t0 = time.time()
    model = fit_model(train_df, feature_cols)
    print(f"      Trained in {time.time()-t0:.1f}s  |  "
          f"test set: {len(test_df):,} bars  (~{n_test_days} days)\n")

    signals_df = apply_signals(model, feature_cols, test_df)
    raw_long   = int((signals_df["signal"] ==  1).sum())
    raw_short  = int((signals_df["signal"] == -1).sum())
    print(f"      Raw model signals: long={raw_long}  short={raw_short}  "
          f"total={raw_long + raw_short}")

    _print_bias_comparison(signals_df, n_test_days)

    # ── [3/4] Backtest ────────────────────────────────────────────────────────
    print("[3/4] Running backtest for each variant...")
    bt_results: dict[str, dict] = {}
    for name, filter_fn in _VARIANTS.items():
        bt_results[name] = _backtest_variant(signals_df, filter_fn)
        n  = bt_results[name].get("n_trades", 0)
        pf = bt_results[name].get("profit_factor", float("nan"))
        exp = bt_results[name].get("expectancy", float("nan"))
        print(f"      {name:<16} trades={n:>4}  "
              f"PF={_fmt(pf, '.3f'):>6}  exp={_fmt(exp, '+.2f'):>7}")

    _print_signal_funnel(signals_df, n_test_days)
    _print_backtest_table(bt_results)

    # ── [4/4] Walk-forward ────────────────────────────────────────────────────
    print(f"[4/4] Running walk-forward ({len(config.WF_TRAIN_FRACTIONS)} windows"
          f" x {len(_VARIANTS)} variants = "
          f"{len(config.WF_TRAIN_FRACTIONS) * len(_VARIANTS)} model fits)...")

    wf_results: dict[str, list[dict]] = {}
    for name, filter_fn in _VARIANTS.items():
        print(f"      {name} ...", end="", flush=True)
        t_wf = time.time()
        wf_results[name] = _run_wf(df, feature_cols, filter_fn)
        avg_pf = _wf_avg_pf(wf_results[name])
        total  = sum(r["n_trades"] for r in wf_results[name])
        print(f"  done in {time.time()-t_wf:.0f}s  |  "
              f"total_trades={total}  avg_PF={_fmt(avg_pf, '.3f')}")

    _print_wf_table(wf_results)
    _print_verdict(wf_results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows_out = []
    for name, filter_fn in _VARIANTS.items():
        bt  = bt_results[name]
        rows_out.append({
            "variant":         name,
            "bt_trades":       bt.get("n_trades",     0),
            "bt_win_rate":     bt.get("win_rate",      float("nan")),
            "bt_pf":           bt.get("profit_factor", float("nan")),
            "bt_exp":          bt.get("expectancy",    float("nan")),
            "bt_max_dd":       bt.get("max_drawdown",  float("nan")),
            "wf_avg_pf":       _wf_avg_pf(wf_results[name]),
            "wf_avg_exp":      _wf_avg_exp(wf_results[name]),
            "wf_worst_dd":     _wf_worst_dd(wf_results[name]),
            "wf_total_trades": sum(r["n_trades"] for r in wf_results[name]),
        })
        for wi, r in enumerate(wf_results[name]):
            rows_out[-1][f"wf_win{wi+1}_trades"] = r["n_trades"]
            rows_out[-1][f"wf_win{wi+1}_pf"]     = r.get("profit_factor", float("nan"))

    out_path = os.path.join(config.EXPERIMENTS_DIR, "context_4h_adx_results.csv")
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARKAD MRK — 4H ADX Context Validation (Design A)"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to 1H OHLCV CSV (e.g. data/BTCUSDT_1h_4y.csv)",
    )
    args = parser.parse_args()
    main(args.data)

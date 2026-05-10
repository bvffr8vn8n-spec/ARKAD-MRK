"""
experiments/context_4h_test.py — 4H context layer validation experiment.

Tests whether gating the existing 1H/24H model signals by a 4H directional
bias improves walk-forward profit factor above the baseline of 0.64.

Three signal variants are compared:
  Baseline   : existing 1H model, no 4H filter
  4H-Strict  : long only when bias_4h == 'bull'; short only when bias_4h == 'bear'
  4H-Relaxed : long when bias_4h in ('bull','neutral'); short when in ('bear','neutral')

For each variant:
  - Single backtest on the 20% held-out test set
  - Walk-forward across 3 expanding windows (retrain from scratch each window)
  - Side-by-side comparison tables for both backtest and walk-forward

Success criterion (from research report):
  WF avg PF improves from baseline (~0.64) to > 0.80 for at least one variant.
  Any improvement >= +0.16 vs baseline confirms the 4H hypothesis.

Usage
-----
  python experiments/context_4h_test.py --data data/BTCUSDT_1h_4y.csv
  python experiments/context_4h_test.py --data data/BTCUSDT_1h.csv
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
from features.context_4h import add_4h_context, apply_4h_context_filter, print_4h_bias_stats
from features.market_regime import add_regime_columns, add_session_column
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


# ── Constants ─────────────────────────────────────────────────────────────────

_VARIANTS = {
    "Baseline":    "none",
    "4H-Strict":   "strict",
    "4H-Relaxed":  "relaxed",
}

_WF_METRIC_KEYS = ["n_trades", "profit_factor", "win_rate", "expectancy", "max_drawdown"]

_BASELINE_WF_PF = 0.64   # known baseline from prior research; used in verdict
_SUCCESS_DELTA  = 0.16   # WF avg PF must improve by at least this much


# ── Filter application ────────────────────────────────────────────────────────

def _apply_variant(signals_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Apply filter variant and set 'signal' to the filtered result."""
    df = signals_df.copy()
    if mode == "none":
        return df
    df = apply_4h_context_filter(df, mode=mode)
    df["signal"] = df["signal_4h_filtered"]
    return df


# ── Backtest helpers ──────────────────────────────────────────────────────────

def _backtest_variant(signals_df: pd.DataFrame, mode: str) -> dict:
    """Run backtest for a single variant. Returns safe metrics dict."""
    df = _apply_variant(signals_df, mode)
    trades, equity = run_backtest(df)
    m = compute_metrics(trades, equity)
    if "error" in m:
        return {k: (0 if k == "n_trades" else float("nan"))
                for k in ["n_trades", "win_rate", "profit_factor",
                           "expectancy", "max_drawdown", "total_return"]}
    return m


# ── Walk-forward helpers ──────────────────────────────────────────────────────

def _run_wf(df: pd.DataFrame, feature_cols: list[str], mode: str) -> list[dict]:
    """
    Run walk-forward for one variant mode.  Silent (no printing).

    Returns
    -------
    list of per-window result dicts (length == len(WF_TRAIN_FRACTIONS)).
    """
    n    = len(df)
    rows = []

    for train_frac in config.WF_TRAIN_FRACTIONS:
        test_frac = train_frac + config.WF_TEST_FRACTION
        train_end = int(n * train_frac)
        test_end  = min(int(n * test_frac), n)
        purge_end = train_end - config.FORWARD_RETURN_WINDOW

        train_slice = df.iloc[0:purge_end]
        test_slice  = df.iloc[train_end:test_end]

        empty_row = {k: (0 if k == "n_trades" else float("nan"))
                     for k in _WF_METRIC_KEYS}
        empty_row.update({"train_frac": train_frac,
                           "test_frac":  min(test_frac, 1.0)})

        if len(train_slice) < 50 or len(test_slice) < 5:
            rows.append(empty_row)
            continue

        # Silence sklearn/RF output during fit
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = fit_model(train_slice, feature_cols)

        signals_wf = apply_signals(model, feature_cols, test_slice)
        signals_wf = _apply_variant(signals_wf, mode)

        trades, equity = run_backtest(signals_wf)
        m = compute_metrics(trades, equity)

        if "error" in m:
            rows.append(empty_row)
        else:
            row = {k: m.get(k, float("nan")) for k in _WF_METRIC_KEYS}
            row["n_trades"] = int(row["n_trades"] or 0)
            row.update({"train_frac": train_frac,
                         "test_frac":  min(test_frac, 1.0)})
            rows.append(row)

    return rows


def _wf_avg_pf(rows: list[dict]) -> float:
    """Mean PF across windows that had at least one trade."""
    pf_vals = [r["profit_factor"] for r in rows
               if r["n_trades"] > 0 and not math.isnan(r.get("profit_factor", float("nan")))]
    return float(np.mean(pf_vals)) if pf_vals else float("nan")


def _wf_avg_exp(rows: list[dict]) -> float:
    """Mean expectancy across windows that had trades."""
    exp_vals = [r["expectancy"] for r in rows
                if r["n_trades"] > 0 and not math.isnan(r.get("expectancy", float("nan")))]
    return float(np.mean(exp_vals)) if exp_vals else float("nan")


def _wf_worst_dd(rows: list[dict]) -> float:
    """Worst max drawdown across all windows."""
    dd_vals = [r["max_drawdown"] for r in rows
               if not math.isnan(r.get("max_drawdown", float("nan")))]
    return float(max(dd_vals)) if dd_vals else float("nan")


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(v, fmt: str = "", fallback: str = "N/A") -> str:
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return fallback
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_signal_funnel(signals_df: pd.DataFrame, n_days: float) -> None:
    """Print how many signals each variant produces and the resulting trade rate."""
    n_bars = len(signals_df)
    sep    = "-" * 68

    col_w = 13
    lbl_w = 26

    raw_n = int((signals_df["signal"] != 0).sum())

    print(f"\n  Signal Funnel  ({n_bars:,} test bars  |  ~{n_days:.0f} days)")
    print(f"  {sep}")
    hdr = f"  {'Stage':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS)
    print(hdr)
    print(f"  {sep}")
    print(f"  {'Raw signals (model)':<{lbl_w}}" +
          "".join(f"{raw_n:>{col_w},}" for _ in _VARIANTS))

    # One row showing how many signals survive each 4H filter variant
    row = f"  {'After 4H context filter':<{lbl_w}}"
    for name, mode in _VARIANTS.items():
        if mode == "none":
            row += f"{'—':>{col_w}}"
        else:
            filtered = apply_4h_context_filter(signals_df, mode=mode)
            n_fil = int((filtered["signal_4h_filtered"] != 0).sum())
            pct   = n_fil / raw_n * 100 if raw_n > 0 else 0.0
            row  += f"{n_fil:>{col_w - 5},} ({pct:.0f}%)"
    print(row)
    print(f"  {sep}")


def _print_backtest_table(bt_results: dict[str, dict]) -> None:
    """Print side-by-side backtest metrics for all variants."""
    col_w = 13
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
            m   = bt_results[name]
            val = m.get(key)
            row += f"{_fmt(val, fmt_str) + suffix:>{col_w}}"
        print(row)

    print(f"  {sep}")


def _print_wf_table(wf_results: dict[str, list[dict]]) -> None:
    """Print per-window and aggregate walk-forward results for all variants."""
    col_w = 18
    lbl_w = 22
    sep   = "-" * (lbl_w + col_w * len(_VARIANTS) + 2)
    n_windows = len(config.WF_TRAIN_FRACTIONS)

    print(f"\n  Walk-Forward Results  ({n_windows} expanding windows)")
    print(f"  {sep}")
    hdr = f"  {'Window':<{lbl_w}}" + "".join(f"{v:>{col_w}}" for v in _VARIANTS)
    print(hdr)
    print(f"  {sep}")

    # Per-window rows
    for wi in range(n_windows):
        rows_for_window = {name: wf_results[name][wi] for name in _VARIANTS}
        # Build window label from any variant (same fracs for all)
        r0   = next(iter(rows_for_window.values()))
        t0   = f"{r0['train_frac']*100:.0f}"
        t1   = f"{r0['test_frac']*100:.0f}"
        lbl  = f"Win {wi+1} [{t0}%–{t1}%]"

        row = f"  {lbl:<{lbl_w}}"
        for name in _VARIANTS:
            r  = rows_for_window[name]
            n  = int(r["n_trades"])
            pf = r.get("profit_factor", float("nan"))
            if n == 0 or math.isnan(pf):
                cell = "0 tr  PF=N/A"
            else:
                cell = f"{n} tr  PF={pf:.2f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"  {sep}")

    # Aggregate rows
    agg_rows = [
        ("WF avg PF",      _wf_avg_pf,  ".3f"),
        ("WF avg exp $",   _wf_avg_exp, "+.2f"),
        ("WF worst DD%",   _wf_worst_dd, ".1f"),
    ]

    for label, fn, fmt_str in agg_rows:
        row = f"  {label:<{lbl_w}}"
        for name in _VARIANTS:
            val = fn(wf_results[name])
            row += f"{_fmt(val, fmt_str):>{col_w}}"
        print(row)

    print(f"  {sep}")


def _print_verdict(wf_results: dict[str, list[dict]]) -> None:
    """Print hypothesis verdict based on WF avg PF comparison."""
    sep = "=" * 68

    baseline_wf_pf = _wf_avg_pf(wf_results["Baseline"])
    if math.isnan(baseline_wf_pf):
        baseline_wf_pf = _BASELINE_WF_PF  # fall back to prior known value

    best_name = None
    best_pf   = float("-inf")
    for name, mode in _VARIANTS.items():
        if mode == "none":
            continue
        pf = _wf_avg_pf(wf_results[name])
        if not math.isnan(pf) and pf > best_pf:
            best_pf   = pf
            best_name = name

    print(f"\n  {sep}")
    print(f"  HYPOTHESIS VERDICT")
    print(f"  {sep}")
    print(f"  Success criterion : WF avg PF > 0.80"
          f"  (i.e., delta vs baseline >= +{_SUCCESS_DELTA:.2f})")
    print(f"  Baseline WF avg PF: {_fmt(baseline_wf_pf, '.3f')}")

    if best_name is None or math.isnan(best_pf):
        print(f"\n  Result : FAILED — no 4H variant produced a valid WF PF.")
        print(f"  Action : 4H context adds no signal; do not proceed to v2 MTF.")
    else:
        delta = best_pf - baseline_wf_pf
        print(f"  Best variant      : {best_name}  WF avg PF = {best_pf:.3f}"
              f"  (delta = {delta:+.3f})")

        if best_pf > 0.80:
            print(f"\n  Result : CONFIRMED — 4H context improves WF PF above 0.80.")
            print(f"  Action : Proceed to Priority 2 (15m execution layer).")
            print(f"           Set USE_4H_CONTEXT=True in config.py to activate")
            print(f"           and CONTEXT_4H_MODE='{_VARIANTS[best_name]}' for main pipeline.")
        elif delta >= _SUCCESS_DELTA:
            print(f"\n  Result : PARTIAL — WF PF did not reach 0.80 but improved"
                  f" by >= {_SUCCESS_DELTA:.2f}.")
            print(f"  Action : 4H context is directionally helpful.  Combine with")
            print(f"           multi-asset expansion before building the 15m layer.")
        else:
            print(f"\n  Result : INCONCLUSIVE — WF PF improved but below success threshold.")
            print(f"  Action : Test with more data or alternative 4H SMA windows before")
            print(f"           concluding the hypothesis is invalid.")

    print(f"  {sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str) -> None:
    symbol = os.path.splitext(os.path.basename(data_path))[0]

    print(f"\n{'='*68}")
    print(f"  ARKAD MRK — 4H Context Layer Validation")
    print(f"  Data: {os.path.basename(data_path)}")
    print(f"  4H SMA: fast={config.CONTEXT_4H_SMA_FAST}x4H "
          f"({config.CONTEXT_4H_SMA_FAST * 4} 1H bars)  "
          f"slow={config.CONTEXT_4H_SMA_SLOW}x4H "
          f"({config.CONTEXT_4H_SMA_SLOW * 4} 1H bars)")
    print(f"  Variants: {list(_VARIANTS.keys())}")
    print(f"{'='*68}\n")

    # ── [1/4] Load + features + 4H context ───────────────────────────────────
    print("[1/4] Loading data and building features...")
    df = load_ohlcv(data_path)
    df = generate_features(df)
    df = add_4h_context(df)
    df = add_regime_columns(df)
    df = add_session_column(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    n_buy  = (df["label"] ==  1).sum()
    n_sell = (df["label"] == -1).sum()
    n_neut = (df["label"] ==  0).sum()
    print(f"      {len(df):,} bars after dropna  |  "
          f"buy={n_buy:,}  sell={n_sell:,}  neutral={n_neut:,}")
    print_4h_bias_stats(df)

    # ── [2/4] Train single model on 80% train set ─────────────────────────────
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

    raw_long  = int((signals_df["signal"] ==  1).sum())
    raw_short = int((signals_df["signal"] == -1).sum())
    print(f"      Raw model signals: long={raw_long}  short={raw_short}  "
          f"total={raw_long + raw_short}")

    # Print 4H bias distribution in the test set
    if "bias_4h" in signals_df.columns:
        print(f"      4H bias in test set:")
        for lbl in ["bull", "neutral", "bear"]:
            cnt = int((signals_df["bias_4h"] == lbl).sum())
            pct = cnt / len(signals_df) * 100
            print(f"        {lbl:<10} {cnt:>5,} bars  ({pct:.1f}%)")
    print()

    # ── [3/4] Backtest comparison ─────────────────────────────────────────────
    print("[3/4] Running backtest for each variant...")
    bt_results: dict[str, dict] = {}
    for name, mode in _VARIANTS.items():
        bt_results[name] = _backtest_variant(signals_df, mode)
        n = bt_results[name].get("n_trades", 0)
        pf = bt_results[name].get("profit_factor", float("nan"))
        exp = bt_results[name].get("expectancy", float("nan"))
        print(f"      {name:<14} trades={n:>4}  "
              f"PF={_fmt(pf, '.3f'):>6}  exp={_fmt(exp, '+.2f'):>7}")

    _print_signal_funnel(signals_df, n_test_days)
    _print_backtest_table(bt_results)

    # ── [4/4] Walk-forward comparison ─────────────────────────────────────────
    print(f"[4/4] Running walk-forward ({len(config.WF_TRAIN_FRACTIONS)} windows"
          f" x {len(_VARIANTS)} variants = "
          f"{len(config.WF_TRAIN_FRACTIONS) * len(_VARIANTS)} model fits)...")

    wf_results: dict[str, list[dict]] = {}
    for name, mode in _VARIANTS.items():
        print(f"      {name} ...", end="", flush=True)
        t_wf = time.time()
        wf_results[name] = _run_wf(df, feature_cols, mode)
        avg_pf = _wf_avg_pf(wf_results[name])
        total  = sum(r["n_trades"] for r in wf_results[name])
        print(f"  done in {time.time()-t_wf:.0f}s  |  "
              f"total_trades={total}  avg_PF={_fmt(avg_pf, '.3f')}")

    _print_wf_table(wf_results)
    _print_verdict(wf_results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows_out = []
    for name, mode in _VARIANTS.items():
        bt = bt_results[name]
        wf_avg_pf_val  = _wf_avg_pf(wf_results[name])
        wf_avg_exp_val = _wf_avg_exp(wf_results[name])
        rows_out.append({
            "variant":        name,
            "mode":           mode,
            "bt_trades":      bt.get("n_trades",      0),
            "bt_win_rate":    bt.get("win_rate",       float("nan")),
            "bt_pf":          bt.get("profit_factor",  float("nan")),
            "bt_exp":         bt.get("expectancy",     float("nan")),
            "bt_max_dd":      bt.get("max_drawdown",   float("nan")),
            "wf_avg_pf":      wf_avg_pf_val,
            "wf_avg_exp":     wf_avg_exp_val,
            "wf_worst_dd":    _wf_worst_dd(wf_results[name]),
            "wf_total_trades":sum(r["n_trades"] for r in wf_results[name]),
        })
        # Per-window details
        for wi, r in enumerate(wf_results[name]):
            rows_out[-1][f"wf_win{wi+1}_trades"] = r["n_trades"]
            rows_out[-1][f"wf_win{wi+1}_pf"]     = r.get("profit_factor", float("nan"))

    out_df   = pd.DataFrame(rows_out)
    out_path = os.path.join(config.EXPERIMENTS_DIR, "context_4h_results.csv")
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARKAD MRK — 4H Context Layer Validation"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to 1H OHLCV CSV (e.g. data/BTCUSDT_1h_4y.csv)",
    )
    args = parser.parse_args()
    main(args.data)

"""
experiments/walk_forward.py — Walk-forward (anchored expanding-window) validation.

Each window trains on all data from bar 0 up to a cutoff, then tests on the
next WF_TEST_FRACTION of the dataset.  The model is re-trained from scratch for
every window, so each out-of-sample test period is genuinely unseen.

Window layout (on a 1,500-bar dataset with WF_TEST_FRACTION=0.10):

  Window 1:  train [0, 900)    test [900, 1050)
  Window 2:  train [0, 1050)   test [1050, 1200)
  Window 3:  train [0, 1200)   test [1200, 1350)

Purge gap
---------
The label at bar t uses close[t + FORWARD_RETURN_WINDOW].  The last
FORWARD_RETURN_WINDOW training bars therefore have labels computed from
prices inside the test window — a boundary look-ahead leak.  We purge these
rows from the training slice before fitting.

Aggregated metrics reported
---------------------------
  avg_win_rate      : mean win rate across windows that produced trades
  avg_profit_factor : mean profit factor across those windows
  avg_expectancy    : mean per-trade expectancy across those windows
  worst_drawdown    : max of per-window max drawdowns
  total_trades      : sum of trades across all windows
"""

import os
import pandas as pd

import config
from models.classifier import get_feature_columns, fit_model, apply_signals
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


_METRIC_KEYS = ["n_trades", "win_rate", "risk_reward", "profit_factor",
                "expectancy", "max_drawdown"]

_NO_TRADES_METRICS = {k: (0 if k == "n_trades" else float("nan"))
                      for k in _METRIC_KEYS}


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run walk-forward validation on a fully-prepared DataFrame (features + labels).

    Parameters
    ----------
    df : DataFrame with all feature columns, 'label', and OHLCV columns present.
         Produced by generate_features() followed by add_labels() + dropna().

    Returns
    -------
    pd.DataFrame — one row per window with per-window performance metrics.
    """
    feature_cols = get_feature_columns(df)
    windows      = _build_windows(df)

    rows = []
    for idx, w in enumerate(windows, start=1):
        # Purge: drop the last FORWARD_RETURN_WINDOW rows from training to
        # prevent label leakage across the train/test boundary.
        purge_end = w["train_end"] - config.FORWARD_RETURN_WINDOW
        train_df  = df.iloc[w["train_start"] : purge_end]
        test_df   = df.iloc[w["train_end"]   : w["test_end"]]

        train_pct_str = f"{w['train_frac']*100:.0f}%"
        test_range    = f"{w['train_frac']*100:.0f}%-{w['test_frac']*100:.0f}%"

        print(f"      Window {idx}: train [0-{train_pct_str}]  "
              f"test [{test_range}]  "
              f"({len(train_df):,} / {len(test_df):,} bars)", end="")

        if len(train_df) < 50 or len(test_df) < 5:
            print("  -- skipped (insufficient data)")
            continue

        model      = fit_model(train_df, feature_cols)
        signals_df = apply_signals(model, feature_cols, test_df)
        trades, equity_curve = run_backtest(signals_df)
        metrics    = compute_metrics(trades, equity_curve)

        n = metrics.get("n_trades", 0) if "error" not in metrics else 0
        print(f"  -> {n} trades")

        if "error" in metrics:
            window_metrics = _NO_TRADES_METRICS.copy()
        else:
            window_metrics = {k: metrics[k] for k in _METRIC_KEYS}

        rows.append({
            "window":     idx,
            "train_bars": len(train_df),
            "test_bars":  len(test_df),
            "train_end_pct":  w["train_frac"],
            "test_end_pct":   w["test_frac"],
            **window_metrics,
        })

    if not rows:
        print("      No windows produced results.")
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    _print_summary(results)
    _save_csv(results)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_windows(df: pd.DataFrame) -> list[dict]:
    """Build the list of (train_start, train_end, test_end) index positions."""
    n = len(df)
    windows = []
    for train_frac in config.WF_TRAIN_FRACTIONS:
        test_frac = train_frac + config.WF_TEST_FRACTION
        windows.append({
            "train_start": 0,
            "train_end":   int(n * train_frac),
            "test_end":    min(int(n * test_frac), n),
            "train_frac":  train_frac,
            "test_frac":   min(test_frac, 1.0),
        })
    return windows


def _print_summary(results: pd.DataFrame) -> None:
    sep = "-" * 72

    # Per-window table
    print(f"\n  Walk-Forward Results — Per Window")
    print(f"  {sep}")
    print(
        f"  {'Win':>3}  {'Train':>7}  {'Test':>11}  "
        f"{'Trades':>6}  {'Win%':>6}  {'R:R':>5}  {'PF':>5}  "
        f"{'Expect':>8}  {'MaxDD':>6}"
    )
    print(f"  {sep}")

    for _, r in results.iterrows():
        train_range = f"0-{r['train_end_pct']*100:.0f}%"
        test_range  = f"{r['train_end_pct']*100:.0f}-{r['test_end_pct']*100:.0f}%"
        n    = int(r["n_trades"])
        win  = f"{r['win_rate']*100:.1f}%"    if not pd.isna(r["win_rate"])      else "  N/A"
        rr   = f"{r['risk_reward']:.2f}"      if not pd.isna(r["risk_reward"])   else " N/A"
        pf   = f"{r['profit_factor']:.2f}"    if not pd.isna(r["profit_factor"]) else " N/A"
        exp  = f"${r['expectancy']:>6.2f}"    if not pd.isna(r["expectancy"])    else "    N/A"
        dd   = f"{r['max_drawdown']:.1f}%"    if not pd.isna(r["max_drawdown"])  else "  N/A"
        print(
            f"  {int(r['window']):>3}  {train_range:>7}  {test_range:>11}  "
            f"{n:>6}  {win:>6}  {rr:>5}  {pf:>5}  {exp:>8}  {dd:>6}"
        )

    print(f"  {sep}")

    # Aggregate — only over windows that had trades
    traded = results[results["n_trades"] > 0]
    total_trades    = int(results["n_trades"].sum())
    avg_win_rate    = traded["win_rate"].mean()
    avg_pf          = traded["profit_factor"].mean()
    avg_expectancy  = traded["expectancy"].mean()
    worst_dd        = results["max_drawdown"].max()

    print(f"\n  Walk-Forward Aggregate ({len(results)} windows, "
          f"{len(traded)} with trades)")
    print(f"  {sep}")
    print(f"  {'Total trades':<28} {total_trades:>10}")
    print(f"  {'Avg win rate':<28} {avg_win_rate*100:>9.1f}%"
          if not pd.isna(avg_win_rate) else f"  {'Avg win rate':<28} {'N/A':>10}")
    print(f"  {'Avg profit factor':<28} {avg_pf:>10.2f}"
          if not pd.isna(avg_pf) else f"  {'Avg profit factor':<28} {'N/A':>10}")
    print(f"  {'Avg expectancy':<28} ${avg_expectancy:>9.2f}"
          if not pd.isna(avg_expectancy) else f"  {'Avg expectancy':<28} {'N/A':>10}")
    print(f"  {'Worst drawdown':<28} {worst_dd:>9.2f}%"
          if not pd.isna(worst_dd) else f"  {'Worst drawdown':<28} {'N/A':>10}")
    print(f"  {sep}\n")


def _save_csv(results: pd.DataFrame) -> None:
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EXPERIMENTS_DIR, "walk_forward_results.csv")

    save_df = results.copy()
    for col in ["win_rate", "risk_reward", "profit_factor", "expectancy", "max_drawdown"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].round(4)

    save_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}")

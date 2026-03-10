"""
experiments/threshold_sweep.py — Confidence-filter sweep over BUY thresholds.

For each threshold in SWEEP_THRESHOLDS the function:
  1. Applies the threshold to the pre-computed buy_prob column (no re-training).
  2. Runs the full backtest.
  3. Collects key performance metrics.
  4. Prints a comparison table.
  5. Identifies the threshold with the highest profit factor.
  6. Saves all results to experiments/threshold_comparison.csv.

The sweep is cheap: buy_prob is computed once during signal generation,
so iterating over thresholds is just filtering + backtest simulation.
"""

import math
import os
import pandas as pd

import config
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


def _sweep_score(n_trades: int, pf: float, exp: float) -> float:
    """score = PF × log(1 + trades) × expectancy; -inf if non-positive or too few trades."""
    if n_trades < config.MIN_TRADES or pf <= 0 or exp <= 0:
        return float("-inf")
    return pf * math.log1p(n_trades) * exp


# Metrics shown in the comparison table (subset of compute_metrics output)
_TABLE_COLS = ["threshold", "n_trades", "win_rate", "risk_reward",
               "profit_factor", "expectancy", "max_drawdown"]

_NO_TRADES_ROW = {
    "n_trades":     0,
    "win_rate":     float("nan"),
    "risk_reward":  float("nan"),
    "profit_factor": float("nan"),
    "expectancy":   float("nan"),
    "max_drawdown": float("nan"),
}


def run_threshold_sweep(
    signals_df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Sweep BUY probability thresholds and compare backtest performance.

    Parameters
    ----------
    signals_df : DataFrame with columns buy_prob, close, high, low, signal.
                 Produced by models.classifier.generate_signals().
    thresholds : List of float thresholds to test.
                 Defaults to config.SWEEP_THRESHOLDS.

    Returns
    -------
    pd.DataFrame — one row per threshold with all performance metrics.
                   Best row by profit_factor is identified in printed output.
    """
    if thresholds is None:
        thresholds = config.SWEEP_THRESHOLDS

    if "buy_prob" not in signals_df.columns:
        raise ValueError(
            "signals_df must contain a 'buy_prob' column. "
            "Call generate_signals() before run_threshold_sweep()."
        )

    rows = []
    for t in thresholds:
        sweep_df = signals_df.copy()
        sweep_df["signal"] = (sweep_df["buy_prob"] >= t).astype(int)

        trades, equity_curve = run_backtest(sweep_df)
        metrics = compute_metrics(trades, equity_curve)

        if "error" in metrics:
            row = {"threshold": t, **_NO_TRADES_ROW, "score": float("-inf")}
        else:
            n   = int(metrics["n_trades"] or 0)
            pf  = float(metrics["profit_factor"] or 0)
            exp = float(metrics["expectancy"] or -9_999)
            row = {
                "threshold":    t,
                "n_trades":     n,
                "win_rate":     metrics["win_rate"],
                "risk_reward":  metrics["risk_reward"],
                "profit_factor": pf,
                "expectancy":   exp,
                "max_drawdown": metrics["max_drawdown"],
                "score":        _sweep_score(n, pf, exp),
            }

        rows.append(row)

    results = pd.DataFrame(rows)
    _print_table(results)
    _save_csv(results)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_table(results: pd.DataFrame) -> None:
    sep = "-" * 84

    print(f"\n  Threshold Sweep — Confidence Filter Comparison")
    print(f"  {sep}")
    print(
        f"  {'Threshold':>9}  {'Trades':>6}  {'Win%':>6}  "
        f"{'R:R':>5}  {'PF':>6}  {'Expect':>8}  {'MaxDD':>6}  {'Score':>9}"
    )
    print(f"  {sep}")

    best_score = float("-inf")
    best_thr   = None

    for _, r in results.iterrows():
        t        = r["threshold"]
        n        = int(r["n_trades"])
        win_pct  = f"{r['win_rate']*100:.1f}%" if not pd.isna(r["win_rate"]) else "  N/A"
        rr       = f"{r['risk_reward']:.2f}"   if not pd.isna(r["risk_reward"]) else " N/A"
        pf       = f"{r['profit_factor']:.2f}" if not pd.isna(r["profit_factor"]) else " N/A"
        exp      = f"${r['expectancy']:>7.2f}" if not pd.isna(r["expectancy"]) else "     N/A"
        dd       = f"{r['max_drawdown']:.1f}%" if not pd.isna(r["max_drawdown"]) else "  N/A"
        sc_val   = r.get("score", float("-inf"))
        sc       = f"{sc_val:>9.3f}" if sc_val != float("-inf") else "      N/A"

        print(f"  {t:>9.2f}  {n:>6}  {win_pct:>6}  {rr:>5}  {pf:>6}  {exp:>8}  {dd:>6}  {sc:>9}")

        if sc_val != float("-inf") and sc_val > best_score:
            best_score = sc_val
            best_thr   = t

    print(f"  {sep}")

    if best_thr is not None:
        print(f"  Best by score: threshold = {best_thr:.2f}  "
              f"(score = {best_score:.3f},  min_trades = {config.MIN_TRADES})")
    else:
        print(f"  No threshold met criteria (PF>0, exp>0, trades>={config.MIN_TRADES}).")
    print()


def _save_csv(results: pd.DataFrame) -> None:
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EXPERIMENTS_DIR, "threshold_comparison.csv")

    # Format floats to a readable precision before saving
    save_df = results.copy()
    for col in ["win_rate", "risk_reward", "profit_factor", "expectancy", "max_drawdown"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].round(4)

    save_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}")

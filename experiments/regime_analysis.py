"""
experiments/regime_analysis.py — Strategy performance broken down by market regime.

For each regime slice (trend, volatility, and their combinations), the analysis:
  1. Zeros out model signals on every bar outside the slice.
  2. Runs the full backtest on the filtered signals — gives a correct,
     sequential equity curve within that slice.
  3. Computes all standard metrics on the result.

This produces 15 independent performance snapshots:
  - 3 trend-only slices     : trend_up / range / trend_down
  - 3 volatility-only slices: low_vol / normal_vol / high_vol
  - 9 combinations          : every trend × vol pairing

The goal is to identify which regime combinations produce positive expectancy
so that future work can optionally activate a targeted gate.

Output
------
  Console : formatted table grouped by (trend-only | vol-only | combinations)
  CSV     : experiments/regime_analysis.csv  — one row per slice
"""

import math
import os
import pandas as pd

import config
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics


def _slice_score(n_trades: int, pf: float, exp: float) -> float:
    """score = PF × log(1 + trades) × expectancy; -inf when criteria not met."""
    if n_trades < config.MIN_TRADES or pf <= 0 or exp <= 0:
        return float("-inf")
    return pf * math.log1p(n_trades) * exp


_TREND_LABELS   = ["trend_up", "range", "trend_down"]
_VOL_LABELS     = ["low_vol", "normal_vol", "high_vol"]
_SESSION_LABELS = ["asia", "london_open", "eu_mid", "us_open", "us_afternoon", "late"]

_METRIC_KEYS  = ["n_signals", "n_trades", "win_rate", "risk_reward",
                 "profit_factor", "expectancy", "max_drawdown"]

_NO_TRADES = {
    "n_trades":     0,
    "win_rate":     float("nan"),
    "risk_reward":  float("nan"),
    "profit_factor": float("nan"),
    "expectancy":   float("nan"),
    "max_drawdown": float("nan"),
}


def run_regime_analysis(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse strategy performance across all regime slices.

    Parameters
    ----------
    signals_df : DataFrame from generate_signals(), containing columns:
                 signal, trend, vol_regime, close, high, low, atr_pct.

    Returns
    -------
    pd.DataFrame — one row per regime slice with full metrics.
    """
    for col in ("signal", "trend", "vol_regime"):
        if col not in signals_df.columns:
            raise ValueError(
                f"run_regime_analysis() requires '{col}' column. "
                f"Call generate_signals() and add_regime_columns() first."
            )

    rows: list[dict] = []

    # ── Group 1: trend-only slices ────────────────────────────────────────────
    for trend in _TREND_LABELS:
        rows.append(_analyze_slice(signals_df, trend_filter=trend, vol_filter="all"))

    # ── Group 2: volatility-only slices ──────────────────────────────────────
    for vol in _VOL_LABELS:
        rows.append(_analyze_slice(signals_df, trend_filter="all", vol_filter=vol))

    # ── Group 3: trend × vol combinations ────────────────────────────────────
    for trend in _TREND_LABELS:
        for vol in _VOL_LABELS:
            rows.append(_analyze_slice(signals_df, trend_filter=trend, vol_filter=vol))

    results = pd.DataFrame(rows)
    _print_table(results)
    _save_csv(results)

    return results


# ── Core slice analysis ───────────────────────────────────────────────────────

def _analyze_slice(
    signals_df:     pd.DataFrame,
    trend_filter:   str,
    vol_filter:     str,
    session_filter: str = "all",
) -> dict:
    """
    Run a backtest restricted to bars matching the given regime/session filters.

    'all' in any filter means no restriction on that dimension.
    """
    # Build a boolean mask for bars in this slice
    mask = pd.Series(True, index=signals_df.index)
    if trend_filter != "all":
        mask &= signals_df["trend"] == trend_filter
    if vol_filter != "all":
        mask &= signals_df["vol_regime"] == vol_filter
    if session_filter != "all":
        mask &= signals_df["session"] == session_filter

    # Count model signals that fall inside this slice (before backtest blocking)
    n_signals = int((signals_df["signal"] * mask.astype(int)).sum())

    # Zero out signals outside the slice — engine sees only in-slice entries
    filtered = signals_df.copy()
    filtered["signal"] = signals_df["signal"] * mask.astype(int)

    trades, equity_curve = run_backtest(filtered)
    metrics = compute_metrics(trades, equity_curve)

    if "error" in metrics:
        return {
            "trend":          trend_filter,
            "vol_regime":     vol_filter,
            "session":        session_filter,
            "n_signals":      n_signals,
            **_NO_TRADES,
        }

    return {
        "trend":         trend_filter,
        "vol_regime":    vol_filter,
        "session":       session_filter,
        "n_signals":     n_signals,
        "n_trades":      metrics["n_trades"],
        "win_rate":      metrics["win_rate"],
        "risk_reward":   metrics["risk_reward"],
        "profit_factor": metrics["profit_factor"],
        "expectancy":    metrics["expectancy"],
        "max_drawdown":  metrics["max_drawdown"],
    }


# ── Presentation ──────────────────────────────────────────────────────────────

def _print_table(results: pd.DataFrame) -> None:
    sep  = "-" * 78
    hdr  = (f"  {'Trend':<12} {'Vol':<12} {'Sigs':>5} {'Trades':>6}  "
            f"{'Win%':>6}  {'R:R':>5}  {'PF':>5}  {'Expect':>8}  {'MaxDD':>6}")

    print(f"\n  Regime Analysis — Performance by Market Condition")
    print(f"  {sep}")

    groups = [
        ("Trend regimes",       results[results["vol_regime"] == "all"]),
        ("Volatility regimes",  results[(results["trend"] == "all") & (results["vol_regime"] != "all")]),
        ("Combinations",        results[(results["trend"] != "all") & (results["vol_regime"] != "all")]),
    ]

    for group_name, group_df in groups:
        if group_df.empty:
            continue
        print(f"  {group_name}:")
        print(hdr)
        print(f"  {sep}")
        for _, r in group_df.iterrows():
            _print_row(r)
        print(f"  {sep}")
        print()

    # Highlight the best and worst slices by frequency-weighted score
    valid = results[results["n_trades"] >= 5].dropna(subset=["profit_factor", "expectancy"])
    if not valid.empty:
        valid = valid.copy()
        valid["_score"] = valid.apply(
            lambda r: _slice_score(
                int(r["n_trades"]),
                float(r["profit_factor"] or 0),
                float(r["expectancy"] or -9_999),
            ),
            axis=1,
        )
        scored = valid[valid["_score"] != float("-inf")]
        if not scored.empty:
            best  = scored.loc[scored["_score"].idxmax()]
            worst = scored.loc[scored["_score"].idxmin()]
            print(f"  Best  slice (score): "
                  f"{best['trend']} x {best['vol_regime']}  "
                  f"PF={best['profit_factor']:.2f}  Trades={int(best['n_trades'])}"
                  f"  Score={best['_score']:.3f}")
            print(f"  Worst slice (score): "
                  f"{worst['trend']} x {worst['vol_regime']}  "
                  f"PF={worst['profit_factor']:.2f}  Trades={int(worst['n_trades'])}"
                  f"  Score={worst['_score']:.3f}")
        else:
            # Fall back to PF ranking when no slice meets the score criteria
            best  = valid.loc[valid["profit_factor"].idxmax()]
            worst = valid.loc[valid["profit_factor"].idxmin()]
            print(f"  Best  slice (PF, no slice meets score criteria): "
                  f"{best['trend']} x {best['vol_regime']}  "
                  f"PF={best['profit_factor']:.2f}  Trades={int(best['n_trades'])}")
            print(f"  Worst slice (PF): "
                  f"{worst['trend']} x {worst['vol_regime']}  "
                  f"PF={worst['profit_factor']:.2f}  Trades={int(worst['n_trades'])}")
        print()


def _print_row(r: pd.Series) -> None:
    trend = r["trend"]
    vol   = r["vol_regime"]
    sigs  = int(r["n_signals"])
    n     = int(r["n_trades"])

    win  = f"{r['win_rate']*100:.1f}%"    if not pd.isna(r["win_rate"])       else "  N/A"
    rr   = f"{r['risk_reward']:.2f}"      if not pd.isna(r["risk_reward"])    else " N/A"
    pf   = f"{r['profit_factor']:.2f}"    if not pd.isna(r["profit_factor"])  else " N/A"
    exp  = f"${r['expectancy']:>6.2f}"    if not pd.isna(r["expectancy"])     else "    N/A"
    dd   = f"{r['max_drawdown']:.1f}%"    if not pd.isna(r["max_drawdown"])   else "  N/A"

    print(f"  {trend:<12} {vol:<12} {sigs:>5} {n:>6}  "
          f"{win:>6}  {rr:>5}  {pf:>5}  {exp:>8}  {dd:>6}")


def _save_csv(results: pd.DataFrame) -> None:
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EXPERIMENTS_DIR, "regime_analysis.csv")

    save_df = results.copy()
    for col in ["win_rate", "risk_reward", "profit_factor", "expectancy", "max_drawdown"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].round(4)

    save_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}\n")


# ── Session analysis ──────────────────────────────────────────────────────────

def run_session_analysis(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse strategy performance by trading session and session×regime combinations.

    Requires 'signal', 'trend', 'vol_regime', and 'session' columns in signals_df.

    Slices computed:
      - 4  session-only         : asia / eu / us / late
      - 36 trend × vol × session: every combination of 3 trends × 3 vol × 4 sessions

    Output
    ------
      Console : session-only table + best/worst combo highlights
      CSV     : experiments/session_analysis.csv  — all 40 rows
    """
    for col in ("signal", "trend", "vol_regime", "session"):
        if col not in signals_df.columns:
            raise ValueError(
                f"run_session_analysis() requires '{col}' column. "
                f"Ensure add_regime_columns(), add_session_column(), and "
                f"generate_signals() have been called."
            )

    rows: list[dict] = []

    # ── Group 1: session-only (4 slices) ─────────────────────────────────────
    for session in _SESSION_LABELS:
        rows.append(_analyze_slice(signals_df, "all", "all", session))

    # ── Group 2: trend × vol × session (36 slices) ───────────────────────────
    for trend in _TREND_LABELS:
        for vol in _VOL_LABELS:
            for session in _SESSION_LABELS:
                rows.append(_analyze_slice(signals_df, trend, vol, session))

    results = pd.DataFrame(rows)
    _print_session_table(results)
    _save_session_csv(results)

    return results


def _print_session_table(results: pd.DataFrame) -> None:
    sep = "-" * 72
    hdr = (f"  {'Session':<10} {'Sigs':>5} {'Trades':>6}  "
           f"{'Win%':>6}  {'R:R':>5}  {'PF':>5}  {'Expect':>8}  {'MaxDD':>6}")

    # ── Session-only rows ─────────────────────────────────────────────────────
    session_only = results[
        (results["trend"] == "all") & (results["vol_regime"] == "all")
    ]

    print(f"\n  Session Analysis — Performance by Trading Session")
    print(f"  {sep}")
    print(hdr)
    print(f"  {sep}")
    for _, r in session_only.iterrows():
        _print_session_row(r)
    print(f"  {sep}\n")

    # ── Best / worst across trend × vol × session combinations ───────────────
    combos = results[
        (results["trend"] != "all") & (results["vol_regime"] != "all")
    ]
    valid = combos[combos["n_trades"] >= 5].dropna(subset=["profit_factor", "expectancy"])
    if not valid.empty:
        valid = valid.copy()
        valid["_score"] = valid.apply(
            lambda r: _slice_score(
                int(r["n_trades"]),
                float(r["profit_factor"] or 0),
                float(r["expectancy"] or -9_999),
            ),
            axis=1,
        )
        scored = valid[valid["_score"] != float("-inf")]
        if not scored.empty:
            best  = scored.loc[scored["_score"].idxmax()]
            worst = scored.loc[scored["_score"].idxmin()]
            print(f"  Best  combo (score): "
                  f"{best['trend']} x {best['vol_regime']} x {best['session']}  "
                  f"PF={best['profit_factor']:.2f}  Trades={int(best['n_trades'])}"
                  f"  Score={best['_score']:.3f}")
            print(f"  Worst combo (score): "
                  f"{worst['trend']} x {worst['vol_regime']} x {worst['session']}  "
                  f"PF={worst['profit_factor']:.2f}  Trades={int(worst['n_trades'])}"
                  f"  Score={worst['_score']:.3f}")
        else:
            best  = valid.loc[valid["profit_factor"].idxmax()]
            worst = valid.loc[valid["profit_factor"].idxmin()]
            print(f"  Best  combo (PF, no combo meets score criteria): "
                  f"{best['trend']} x {best['vol_regime']} x {best['session']}  "
                  f"PF={best['profit_factor']:.2f}  Trades={int(best['n_trades'])}")
            print(f"  Worst combo (PF): "
                  f"{worst['trend']} x {worst['vol_regime']} x {worst['session']}  "
                  f"PF={worst['profit_factor']:.2f}  Trades={int(worst['n_trades'])}")
        print()


def _print_session_row(r: pd.Series) -> None:
    session = r["session"]
    sigs    = int(r["n_signals"])
    n       = int(r["n_trades"])

    win  = f"{r['win_rate']*100:.1f}%"    if not pd.isna(r["win_rate"])       else "  N/A"
    rr   = f"{r['risk_reward']:.2f}"      if not pd.isna(r["risk_reward"])    else " N/A"
    pf   = f"{r['profit_factor']:.2f}"    if not pd.isna(r["profit_factor"])  else " N/A"
    exp  = f"${r['expectancy']:>6.2f}"    if not pd.isna(r["expectancy"])     else "    N/A"
    dd   = f"{r['max_drawdown']:.1f}%"    if not pd.isna(r["max_drawdown"])   else "  N/A"

    print(f"  {session:<10} {sigs:>5} {n:>6}  "
          f"{win:>6}  {rr:>5}  {pf:>5}  {exp:>8}  {dd:>6}")


def _save_session_csv(results: pd.DataFrame) -> None:
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(config.EXPERIMENTS_DIR, "session_analysis.csv")

    save_df = results.copy()
    for col in ["win_rate", "risk_reward", "profit_factor", "expectancy", "max_drawdown"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].round(4)

    save_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}\n")

"""
run_multi_asset.py — Run the ARKAD MRK research pipeline across the
top 10 most liquid USDT perpetual pairs and produce a cross-asset
performance summary ranked by Profit Factor × Expectancy.

Prerequisites
-------------
    1. Download data first:
           python data/download_all.py
    2. Then run this script:
           python run_multi_asset.py

Usage
-----
    python run_multi_asset.py [options]

Options
-------
    --pairs    SYM,...  Comma-separated list of symbols (default: all 10).
    --verbose           Print full pipeline output for each pair to the
                        terminal instead of a per-pair log file.
    --skip-done         Skip pairs that already have a metrics JSON from
                        a previous run in the reports/ directory.

Output
------
    reports/multi_asset_summary.csv   — ranked CSV with all key metrics
    reports/<PAIR>_pipeline.log       — captured stdout/stderr per pair
                                        (only written in non-verbose mode)
    Terminal                          — ranked summary table + edge verdict

Edge classification
-------------------
A pair is labelled "YES" (edge detected) when the best-selected strategy
simultaneously satisfies ALL of the following:
    - Profit factor  > EDGE_MIN_PF        (default 1.10)
    - Expectancy     > 0                  (positive per-trade expectation)
    - Trades         >= EDGE_MIN_TRADES   (sufficient sample size)
    - Max drawdown   < EDGE_MAX_DD_PCT    (acceptable risk)

The cross-asset ranking score is PF × Expectancy.  It rewards both
edge quality (high PF) and absolute P&L per trade (high expectancy).
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

PAIRS: list[str] = [
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

DATA_DIR    = "data"
REPORTS_DIR = "reports"

# Per-pair pipeline timeout (seconds).  5m data + walk-forward can be slow.
PIPELINE_TIMEOUT_S = 900   # 15 minutes per pair

# Edge classification thresholds
EDGE_MIN_PF      = 1.10    # profit factor must exceed this
EDGE_MIN_TRADES  = 50      # minimum trades in the test window
EDGE_MAX_DD_PCT  = 30.0    # max drawdown must be below this (%)


# ── Metrics file helpers ───────────────────────────────────────────────────────

def _find_latest_metrics(symbol: str, after_ts: float) -> dict | None:
    """
    Return the parsed JSON from the most recently written metrics file for
    `symbol` whose mtime is >= `after_ts` (Unix seconds).

    save_report() writes:  reports/<symbol>_<timestamp>_metrics.json
    """
    pattern    = os.path.join(REPORTS_DIR, f"{symbol}_*_metrics.json")
    candidates = [
        p for p in glob.glob(pattern)
        if os.path.getmtime(p) >= after_ts
    ]
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    try:
        with open(latest, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _already_has_metrics(symbol: str) -> bool:
    """Return True if at least one metrics JSON exists for this symbol."""
    pattern = os.path.join(REPORTS_DIR, f"{symbol}_*_metrics.json")
    return bool(glob.glob(pattern))


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pair(symbol: str, verbose: bool) -> dict | None:
    """
    Invoke  python main.py --data ... --data5m ... --symbol <symbol>
    as a subprocess.

    Pipeline stdout/stderr go to:
        verbose=True  → terminal (pass-through)
        verbose=False → reports/<symbol>_pipeline.log

    Returns the parsed metrics dict on success, None on failure.
    """
    data_1h = os.path.join(DATA_DIR, f"{symbol}_1h_4y.csv")
    data_5m = os.path.join(DATA_DIR, f"{symbol}_5m_4y.csv")

    if not os.path.exists(data_1h):
        print(f"  [SKIP] 1H data file not found: {data_1h}")
        return None

    cmd = [sys.executable, "main.py", "--data", data_1h, "--symbol", symbol]
    if os.path.exists(data_5m):
        cmd += ["--data5m", data_5m]
    else:
        print(f"  [WARN] 5m data not found; running without intraday features")

    log_path = os.path.join(REPORTS_DIR, f"{symbol}_pipeline.log")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"  Running pipeline ...", flush=True)
    t0 = time.time()

    try:
        if verbose:
            proc = subprocess.run(cmd, timeout=PIPELINE_TIMEOUT_S)
        else:
            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=PIPELINE_TIMEOUT_S,
                )
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timed out after {PIPELINE_TIMEOUT_S}s")
        return None
    except Exception as exc:
        print(f"  [ERROR] Subprocess error: {exc}")
        return None

    elapsed = time.time() - t0

    if proc.returncode != 0:
        hint = f"see {log_path}" if not verbose else "check output above"
        print(f"  [ERROR] Pipeline exited with code {proc.returncode}  ({hint})")
        return None

    metrics = _find_latest_metrics(symbol, t0)
    if metrics is None:
        print(f"  [ERROR] No metrics JSON found after run  (see {log_path})")
        return None

    # Quick one-line status
    n   = metrics.get("n_trades",      "?")
    pf  = metrics.get("profit_factor", float("nan"))
    exp = metrics.get("expectancy",    float("nan"))
    pf_s  = f"{pf:.2f}"  if isinstance(pf,  float) and not math.isnan(pf)  else str(pf)
    exp_s = f"{exp:.2f}" if isinstance(exp, float) and not math.isnan(exp) else str(exp)
    print(f"  Done in {elapsed:.0f}s  |  trades={n}  PF={pf_s}  exp={exp_s}")
    if not verbose:
        print(f"  Full log: {log_path}")

    return metrics


# ── Metric helpers ────────────────────────────────────────────────────────────

def _safe_float(m: dict, key: str, default: float = 0.0) -> float:
    """Extract a numeric value; treat inf/nan/str as default."""
    v = m.get(key, default)
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return default
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return default
    return float(v)


def _pf_x_exp(m: dict) -> float:
    pf  = _safe_float(m, "profit_factor")
    exp = _safe_float(m, "expectancy")
    if pf <= 0 or exp <= 0:
        return 0.0
    return pf * exp


def _has_edge(m: dict) -> bool:
    pf  = _safe_float(m, "profit_factor")
    exp = _safe_float(m, "expectancy")
    n   = int(_safe_float(m, "n_trades", 0))
    dd  = _safe_float(m, "max_drawdown", 9999.0)
    return (
        pf  > EDGE_MIN_PF
        and exp > 0.0
        and n  >= EDGE_MIN_TRADES
        and dd < EDGE_MAX_DD_PCT
    )


def _build_row(symbol: str, m: dict) -> dict:
    return {
        "symbol":        symbol,
        "n_trades":      int(_safe_float(m, "n_trades", 0)),
        "win_rate_pct":  round(_safe_float(m, "win_rate") * 100, 2),
        "profit_factor": round(_safe_float(m, "profit_factor"), 4),
        "expectancy":    round(_safe_float(m, "expectancy"), 4),
        "max_drawdown":  round(_safe_float(m, "max_drawdown"), 2),
        "total_return":  round(_safe_float(m, "total_return"), 2),
        "pf_x_exp":      round(_pf_x_exp(m), 4),
        "has_edge":      _has_edge(m),
    }


def _failed_row(symbol: str) -> dict:
    return {
        "symbol":        symbol,
        "n_trades":      0,
        "win_rate_pct":  float("nan"),
        "profit_factor": float("nan"),
        "expectancy":    float("nan"),
        "max_drawdown":  float("nan"),
        "total_return":  float("nan"),
        "pf_x_exp":      0.0,
        "has_edge":      False,
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def _fmt(v, fmt: str = ".2f", fallback: str = "N/A") -> str:
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return fallback
        return format(v, fmt)
    except (TypeError, ValueError):
        return fallback


def _print_summary(rows: list[dict]) -> None:
    sep  = "=" * 100
    sep2 = "-" * 100

    print(f"\n{sep}")
    print(f"  ARKAD MRK — Multi-Asset Performance Summary  "
          f"(ranked by PF × Expectancy)")
    print(f"{sep}")
    print(
        f"  {'Rank':<5} {'Symbol':<12} {'Trades':>7} {'Win%':>6} "
        f"{'PF':>6} {'Exp$':>8} {'DD%':>6} {'Ret%':>7} {'PF×Exp':>8}  {'Edge':>5}"
    )
    print(f"  {sep2}")

    for rank, r in enumerate(rows, 1):
        edge_str = "YES" if r["has_edge"] else "no"
        mark     = " *" if r["has_edge"] else "  "
        print(
            f"  {rank:<5} {r['symbol']:<12} {r['n_trades']:>7} "
            f"{_fmt(r['win_rate_pct'], '.1f'):>5}% "
            f"{_fmt(r['profit_factor'], '.2f'):>6} "
            f"{_fmt(r['expectancy'], '.2f'):>8} "
            f"{_fmt(r['max_drawdown'], '.1f'):>5}% "
            f"{_fmt(r['total_return'], '.1f'):>6}% "
            f"{_fmt(r['pf_x_exp'], '.3f'):>8}"
            f"{mark} {edge_str:>5}"
        )

    print(f"  {sep2}")

    edge_pairs    = [r["symbol"] for r in rows if r["has_edge"]]
    no_edge_pairs = [r["symbol"] for r in rows if not r["has_edge"]]

    print(f"\n  Edge criteria applied:")
    print(f"    Profit factor  > {EDGE_MIN_PF}")
    print(f"    Expectancy     > 0")
    print(f"    Trades        >= {EDGE_MIN_TRADES}")
    print(f"    Max drawdown   < {EDGE_MAX_DD_PCT}%")

    print(f"\n  Assets WITH real edge  ({len(edge_pairs)}): "
          f"{', '.join(edge_pairs) if edge_pairs else 'none'}")
    print(f"  Assets WITHOUT edge  ({len(no_edge_pairs)}): "
          f"{', '.join(no_edge_pairs) if no_edge_pairs else 'none'}")
    print(f"\n  (*) marks assets with detected edge.")
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ARKAD pipeline across top-10 USDT perp pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_multi_asset.py\n"
            "  python run_multi_asset.py --verbose\n"
            "  python run_multi_asset.py --pairs BTCUSDT,ETHUSDT --verbose\n"
            "  python run_multi_asset.py --skip-done"
        ),
    )
    parser.add_argument(
        "--pairs", default=None,
        metavar="SYM1,SYM2,...",
        help="Comma-separated list of symbols (default: all 10).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full pipeline output to the terminal (very verbose).",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip pairs that already have a metrics JSON in reports/.",
    )
    args = parser.parse_args()

    pairs = (
        [p.strip().upper() for p in args.pairs.split(",")]
        if args.pairs else PAIRS
    )

    print(f"\n{'='*60}")
    print(f"  ARKAD MRK — Multi-Asset Research Run")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Pairs     : {', '.join(pairs)}")
    print(f"  Verbose   : {args.verbose}")
    print(f"  Skip done : {args.skip_done}\n")

    all_rows: list[dict] = []

    for i, symbol in enumerate(pairs, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(pairs)}]  {symbol}")
        print(f"{'─'*60}")

        if args.skip_done and _already_has_metrics(symbol):
            print(f"  [SKIP] Metrics JSON already exists; loading latest ...")
            # Load the most recent existing metrics for this symbol
            pattern    = os.path.join(REPORTS_DIR, f"{symbol}_*_metrics.json")
            candidates = glob.glob(pattern)
            latest = max(candidates, key=os.path.getmtime) if candidates else None
            if latest:
                with open(latest, encoding="utf-8") as f:
                    metrics = json.load(f)
                row = _build_row(symbol, metrics)
            else:
                row = _failed_row(symbol)
            all_rows.append(row)
            continue

        metrics = _run_pair(symbol, args.verbose)
        if metrics is None:
            all_rows.append(_failed_row(symbol))
        else:
            all_rows.append(_build_row(symbol, metrics))

    # ── Sort by pf_x_exp descending ──────────────────────────────────────────
    all_rows.sort(key=lambda r: r["pf_x_exp"], reverse=True)

    _print_summary(all_rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "multi_asset_summary.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()

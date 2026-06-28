"""
One-off analysis of paper_trades_tier1.csv split into two cohorts:
  - PRE-FIX: trades whose entry preceded the live buffer-seed correction
  - POST-FIX: trades whose entry came after it

Per-cohort: N, WR, PF, avg R, sum R, total $ PnL, avg MFE/MAE, by asset.

Run:
    python experiments/cohort_analysis.py [path_to_csv]
"""

import csv
import os
import sys
from collections import defaultdict
from statistics import mean, median

sys.stdout.reconfigure(encoding="utf-8")

# Default: server-uploaded csv copy in repo root; override via argv[1]
_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper_trades_tier1.csv",
)
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT

# Per user: the live model + buffer fix took effect at trade #37-38.
# Anything < SPLIT_ID is "old behaviour", >= SPLIT_ID is "new behaviour".
SPLIT_ID = 37


def _f(s, default=0.0):
    try:
        return float(s) if s not in ("", None) else default
    except ValueError:
        return default


def load() -> list[dict]:
    rows: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["trade_id"]    = int(row["trade_id"])
            row["net_pnl_usd"] = _f(row["net_pnl_usd"])
            row["r_multiple"]  = _f(row["r_multiple"])
            row["mfe_r"]       = _f(row["mfe_r"])
            row["mae_r"]       = _f(row["mae_r"])
            rows.append(row)
    return sorted(rows, key=lambda r: r["trade_id"])


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    pnl   = [r["net_pnl_usd"] for r in rows]
    rmul  = [r["r_multiple"] for r in rows]
    wins  = [p for p in pnl if p > 0]
    losses= [p for p in pnl if p <= 0]
    gw    = sum(wins)
    gl    = abs(sum(losses))
    pf    = (gw / gl) if gl > 0 else float("inf")
    return {
        "n":              len(rows),
        "wins":           len(wins),
        "wr":             len(wins) / len(rows) if rows else 0.0,
        "sum_pnl":        sum(pnl),
        "sum_r":          sum(rmul),
        "avg_r":          mean(rmul),
        "median_r":       median(rmul),
        "pf":             pf,
        "gross_win":      gw,
        "gross_loss":     gl,
        "avg_mfe":        mean(r["mfe_r"] for r in rows),
        "avg_mae":        mean(r["mae_r"] for r in rows),
        "best_pnl":       max(pnl),
        "worst_pnl":      min(pnl),
        "tp_exits":       sum(1 for r in rows if r["exit_reason"] == "tp"),
        "stop_exits":     sum(1 for r in rows if r["exit_reason"] == "stop"),
        "time_exits":     sum(1 for r in rows if r["exit_reason"] == "time"),
    }


def print_block(title: str, s: dict) -> None:
    print(f"\n  {title}")
    print(f"  {'-' * 60}")
    if s["n"] == 0:
        print("  (no trades)")
        return
    print(f"  {'N trades':<22} {s['n']:>10}")
    print(f"  {'Wins / Losses':<22} {s['wins']:>5} / {s['n'] - s['wins']:<5}")
    print(f"  {'Win rate':<22} {s['wr']*100:>9.1f}%")
    print(f"  {'Total PnL ($)':<22} {s['sum_pnl']:>+10.2f}")
    print(f"  {'Sum R':<22} {s['sum_r']:>+10.2f}")
    print(f"  {'Avg R':<22} {s['avg_r']:>+10.3f}")
    print(f"  {'Median R':<22} {s['median_r']:>+10.3f}")
    print(f"  {'Profit factor':<22} {s['pf']:>10.3f}")
    print(f"  {'Gross win / loss':<22} {s['gross_win']:>+5.2f} / {-s['gross_loss']:<+5.2f}")
    print(f"  {'Avg MFE (R)':<22} {s['avg_mfe']:>10.3f}")
    print(f"  {'Avg MAE (R)':<22} {s['avg_mae']:>10.3f}")
    print(f"  {'Best / worst PnL':<22} {s['best_pnl']:>+5.2f} / {s['worst_pnl']:<+5.2f}")
    print(f"  {'Exit reasons (TP/Stop/Time)':<32}"
          f"  {s['tp_exits']} / {s['stop_exits']} / {s['time_exits']}")


def per_asset(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["symbol"]].append(r)
    return {s: stats(rs) for s, rs in by.items()}


def print_per_asset(title: str, rows: list[dict]) -> None:
    print(f"\n  Per-asset breakdown — {title}")
    print(f"  {'-' * 60}")
    print(f"  {'asset':<10} {'n':>4} {'WR':>7} {'PF':>7} {'sum_R':>9} {'pnl_$':>10}")
    for asset, s in sorted(per_asset(rows).items()):
        pf = s["pf"]
        pf_str = f"{pf:>7.3f}" if pf < 1000 else "    inf"
        print(f"  {asset:<10} {s['n']:>4} {s['wr']*100:>6.1f}% {pf_str} "
              f"{s['sum_r']:>+9.2f} {s['sum_pnl']:>+10.2f}")


def main() -> None:
    rows = load()
    pre  = [r for r in rows if r["trade_id"] <  SPLIT_ID]
    post = [r for r in rows if r["trade_id"] >= SPLIT_ID]

    sep = "=" * 70
    print(sep)
    print(f"  ARKAD MRK — Pre/Post buffer-fix cohort analysis")
    print(f"  Split at trade_id = {SPLIT_ID}  ({CSV_PATH})")
    print(sep)
    print(f"  Total trades in log: {len(rows)}")
    print(f"  PRE-FIX  (id < {SPLIT_ID}): {len(pre)} trades")
    print(f"  POST-FIX (id >= {SPLIT_ID}): {len(post)} trades")

    print_block("PRE-FIX cohort  (broken / older buffer)", stats(pre))
    print_block("POST-FIX cohort (new buffer)",            stats(post))

    print_per_asset("PRE-FIX",  pre)
    print_per_asset("POST-FIX", post)

    s1, s2 = stats(pre), stats(post)
    print(f"\n  {sep}")
    print(f"  DELTA  (POST minus PRE)")
    print(f"  {sep}")
    print(f"  {'WR':<18} {s2['wr']*100 - s1['wr']*100:>+8.1f} pp")
    print(f"  {'Avg R':<18} {s2['avg_r'] - s1['avg_r']:>+8.3f}")
    print(f"  {'PF':<18} {(s2['pf'] if s2['pf']<1000 else 0) - (s1['pf'] if s1['pf']<1000 else 0):>+8.3f}")
    print(f"  {'Avg MFE (R)':<18} {s2['avg_mfe'] - s1['avg_mfe']:>+8.3f}")
    print(f"  {'Avg MAE (R)':<18} {s2['avg_mae'] - s1['avg_mae']:>+8.3f}")
    print()


if __name__ == "__main__":
    main()

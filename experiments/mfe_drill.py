"""
MFE drill-down on paper_trades_tier1.csv.

Answers:
  - distribution of MFE in R-buckets
  - per-bucket win/loss split
  - for LOSERS: how far did the trade go in our direction before reversing?
  - reversal rate at each potential TP1 level

Run:
    python experiments/mfe_drill.py [path_to_csv]
"""

import csv
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "paper_trades_tier1.csv",
)
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
SPLIT_ID = 37   # PRE-FIX vs POST-FIX, same as cohort_analysis

# Bucket edges in R-units of MFE.  Aligned with current scaled-exit levels:
#   TP1 = 0.65R, TP2 = 1.0R, TP3 = 1.67R.
BUCKETS = [
    (0.00, 0.20, "[0.0 – 0.2)   immediate reversal"),
    (0.20, 0.40, "[0.2 – 0.4)   tiny pop"),
    (0.40, 0.55, "[0.4 – 0.55)  decent move, no TP"),
    (0.55, 0.65, "[0.55 – 0.65) just shy of TP1"),
    (0.65, 1.00, "[0.65 – 1.0)  hit TP1"),
    (1.00, 1.67, "[1.0 – 1.67)  hit TP2"),
    (1.67, 999.0, "[1.67+ )       hit TP3"),
]


def _f(s, d=0.0):
    try:
        return float(s) if s not in ("", None) else d
    except ValueError:
        return d


def load() -> list[dict]:
    rows: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["trade_id"]    = int(r["trade_id"])
            r["net_pnl_usd"] = _f(r["net_pnl_usd"])
            r["r_multiple"]  = _f(r["r_multiple"])
            r["mfe_r"]       = _f(r["mfe_r"])
            r["mae_r"]       = _f(r["mae_r"])
            r["is_loser"]    = r["net_pnl_usd"] < 0
            rows.append(r)
    return rows


def bucket_for(mfe: float) -> str:
    for lo, hi, lbl in BUCKETS:
        if lo <= mfe < hi:
            return lbl
    return BUCKETS[-1][2]


def print_bucket_table(title: str, rows: list[dict]) -> None:
    print(f"\n  {title}  (n={len(rows)})")
    print(f"  {'-' * 76}")
    print(f"  {'MFE bucket':<35} {'n':>4} {'wins':>5} {'losers':>7} "
          f"{'wr':>6} {'avg_R':>7}")
    print(f"  {'-' * 76}")

    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[bucket_for(r["mfe_r"])].append(r)

    # Preserve order
    for _, _, lbl in BUCKETS:
        bucket = by_bucket.get(lbl, [])
        n = len(bucket)
        if n == 0:
            print(f"  {lbl:<35} {'0':>4} {'-':>5} {'-':>7} {'-':>6} {'-':>7}")
            continue
        wins = sum(1 for r in bucket if not r["is_loser"])
        losers = n - wins
        avg_r = sum(r["r_multiple"] for r in bucket) / n
        print(f"  {lbl:<35} {n:>4} {wins:>5} {losers:>7} "
              f"{wins/n*100:>5.0f}% {avg_r:>+7.2f}")


def print_reversal_table(title: str, rows: list[dict]) -> None:
    """For each potential TP1 threshold, count how many trades:
      reached_X = mfe_r >= X
      hit_then_reversed = mfe_r >= X AND net_pnl <= 0  (became loser despite touching X)
    """
    print(f"\n  {title}  — 'reached X then reversed to loss'")
    print(f"  {'-' * 76}")
    print(f"  {'X (R)':>6} {'n_reached_X':>13} {'reached_then_loss':>21} "
          f"{'reverse_rate':>14}")
    print(f"  {'-' * 76}")

    levels = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.80, 1.00]
    n_total = len(rows)
    for x in levels:
        reached = [r for r in rows if r["mfe_r"] >= x]
        rev = [r for r in reached if r["is_loser"]]
        rate = (len(rev) / len(reached) * 100) if reached else 0.0
        pct_universe = (len(reached) / n_total * 100) if n_total else 0.0
        print(f"  {x:>6.2f} {len(reached):>6}/{n_total:<6} ({pct_universe:>4.0f}%) "
              f"  {len(rev):>13} {rate:>13.0f}%")


def print_loser_mfe(title: str, rows: list[dict]) -> None:
    losers = [r for r in rows if r["is_loser"]]
    if not losers:
        print(f"\n  {title}: no losers in cohort")
        return

    print(f"\n  {title}  — distribution of MFE among LOSERS  (n={len(losers)})")
    print(f"  {'-' * 76}")

    buckets = defaultdict(int)
    for r in losers:
        buckets[bucket_for(r["mfe_r"])] += 1

    for _, _, lbl in BUCKETS:
        n = buckets.get(lbl, 0)
        pct = (n / len(losers) * 100) if losers else 0.0
        bar = "#" * int(round(pct / 4))
        print(f"  {lbl:<35} {n:>3}  {pct:>5.1f}%  {bar}")

    # Of all losers, average MFE before they died
    avg = sum(r["mfe_r"] for r in losers) / len(losers)
    print(f"\n  Avg MFE among losers: {avg:.3f} R")
    print(f"  Losers that touched ≥0.5 R favorable before reversing: "
          f"{sum(1 for r in losers if r['mfe_r'] >= 0.5)}/{len(losers)}")
    print(f"  Losers that touched ≥0.65R (TP1) before reversing:   "
          f"{sum(1 for r in losers if r['mfe_r'] >= 0.65)}/{len(losers)}")


def print_what_if_tp1(title: str, rows: list[dict]) -> None:
    """
    For each candidate TP1 level X, compute:
      n_would_hit = mfe_r >= X  (these trades would have closed 50% at X)
      net P/L if we replaced the current scaled exit with: 50% at X, no other TPs,
      stop at -1R for the remainder
    """
    print(f"\n  {title}  — counterfactual: TP1 only at X, then stop, no TP2/TP3")
    print(f"  {'-' * 76}")
    print(f"  {'X':>5}  {'n_hit':>6}  {'win_rate':>9}  "
          f"{'avg_R':>7}  {'sum_R':>8}")
    print(f"  {'-' * 76}")

    n = len(rows)
    for x in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]:
        sim_r = []
        for r in rows:
            if r["mfe_r"] >= x:
                # 50% closed at +X.  Remainder 50% rides with BE stop after TP1.
                # If MFE >= X and trade overall lost (i.e. came back), assume
                # second half closed at BE (0R).  If trade won overall, second
                # half rode further; use the original r_multiple as upper bound.
                if r["r_multiple"] > x:
                    # trade actually exceeded X; original r_multiple already
                    # reflects scaled-exit logic — close enough.
                    sim_r.append(r["r_multiple"])
                else:
                    # Reached X but turned into loser — half at +X, half at BE.
                    sim_r.append(0.5 * x)
            else:
                # Never reached X — full stop at -1R OR whatever original R was
                # (covers TP1=BE+stop cases that the engine already records).
                # Use min(r_multiple, 0) — losers stay losers at original R,
                # the rare wins below X get clipped to 0.
                sim_r.append(min(r["r_multiple"], 0.0))
        n_hit = sum(1 for r in rows if r["mfe_r"] >= x)
        wins = sum(1 for v in sim_r if v > 0)
        print(f"  {x:>5.2f}  {n_hit:>3}/{n:<2}  {wins/len(sim_r)*100:>8.1f}% "
              f"  {sum(sim_r)/len(sim_r):>+7.3f}  {sum(sim_r):>+8.2f}")


def main() -> None:
    rows = load()
    pre  = [r for r in rows if r["trade_id"] <  SPLIT_ID]
    post = [r for r in rows if r["trade_id"] >= SPLIT_ID]

    print("=" * 78)
    print(f"  ARKAD MRK — MFE drill-down  ({CSV_PATH})")
    print(f"  Split at trade_id = {SPLIT_ID}")
    print("=" * 78)

    print_bucket_table("MFE distribution — ALL trades",     rows)
    print_bucket_table("MFE distribution — PRE-FIX",        pre)
    print_bucket_table("MFE distribution — POST-FIX",       post)

    print_loser_mfe("ALL losers",       rows)
    print_loser_mfe("PRE-FIX losers",   pre)
    print_loser_mfe("POST-FIX losers",  post)

    print_reversal_table("ALL trades",      rows)
    print_reversal_table("POST-FIX trades", post)

    print_what_if_tp1("ALL trades",      rows)
    print_what_if_tp1("POST-FIX trades", post)


if __name__ == "__main__":
    main()

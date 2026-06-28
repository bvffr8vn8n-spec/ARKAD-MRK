"""
Counterfactual on time-exit trades:
  what if we forced an early close when MFE stayed below 0.3R after some
  intra-trade checkpoint?

CAVEAT: paper_trades_tier1.csv only stores final MFE/MAE, not the time-
series.  We cannot rebuild the actual price path between entry and exit.
This script does a sensitivity sweep over several "assumed early-close R"
values to bracket how big the upside could be.

Cohorts:
  - ALL trades vs POST-FIX
  - Filter: exit_reason == "time"  AND  mfe_r < THRESHOLD
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
SPLIT_ID = 37

MFE_THRESHOLDS = [0.20, 0.30, 0.40]                  # what counts as "going nowhere"
ASSUMED_EXIT_R = [0.0, -0.10, -0.20, -0.30, -0.50]   # what early close would yield


def _f(s, d=0.0):
    try:
        return float(s) if s not in ("", None) else d
    except ValueError:
        return d


def load() -> list[dict]:
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["trade_id"]    = int(r["trade_id"])
            r["net_pnl_usd"] = _f(r["net_pnl_usd"])
            r["r_multiple"]  = _f(r["r_multiple"])
            r["mfe_r"]       = _f(r["mfe_r"])
            r["mae_r"]       = _f(r["mae_r"])
            rows.append(r)
    return rows


def baseline(rows: list[dict]) -> dict:
    n = len(rows)
    rmul = [r["r_multiple"] for r in rows]
    wins = sum(1 for r in rmul if r > 0)
    gw   = sum(r for r in rmul if r > 0)
    gl   = abs(sum(r for r in rmul if r <= 0))
    return {
        "n":    n,
        "wr":   wins / n if n else 0,
        "sumR": sum(rmul),
        "avgR": sum(rmul) / n if n else 0,
        "pf":   gw / gl if gl > 0 else float("inf"),
    }


def simulate(rows: list[dict], mfe_thr: float, assumed_r: float) -> dict:
    """
    Replace r_multiple for trades where:
        exit_reason == "time"  AND  mfe_r < mfe_thr  AND  r_multiple < assumed_r
    Replacement = assumed_r (the early-close result).

    The third condition prevents 'improving' a trade that already closed
    LESS BADLY than assumed_r (e.g., a time-exit at -0.10R wouldn't benefit
    from an assumed-early-close at -0.20R).
    """
    sim_rmul = []
    n_replaced = 0
    saved_R    = 0.0
    for r in rows:
        original = r["r_multiple"]
        is_target = (
            r["exit_reason"] == "time"
            and r["mfe_r"] < mfe_thr
            and original < assumed_r
        )
        new = assumed_r if is_target else original
        sim_rmul.append(new)
        if is_target:
            n_replaced += 1
            saved_R += (new - original)

    n = len(sim_rmul)
    wins = sum(1 for r in sim_rmul if r > 0)
    gw   = sum(r for r in sim_rmul if r > 0)
    gl   = abs(sum(r for r in sim_rmul if r <= 0))
    return {
        "n":          n,
        "n_replaced": n_replaced,
        "saved_R":    saved_R,
        "wr":         wins / n if n else 0,
        "sumR":       sum(sim_rmul),
        "avgR":       sum(sim_rmul) / n if n else 0,
        "pf":         gw / gl if gl > 0 else float("inf"),
    }


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n  {title}  (n={len(rows)})")
    print(f"  {'-' * 88}")

    base = baseline(rows)
    print(f"  Baseline:                       "
          f"sumR={base['sumR']:>+7.2f}  WR={base['wr']*100:>4.1f}%  "
          f"avgR={base['avgR']:>+6.3f}  PF={base['pf']:>5.3f}")

    # Eligible trade count per threshold
    print()
    print(f"  Eligible "
          f"(exit=time AND mfe<thr AND original_R < assumed_R):")
    for thr in MFE_THRESHOLDS:
        elig = [r for r in rows if r["exit_reason"] == "time" and r["mfe_r"] < thr]
        worst_R = min((r["r_multiple"] for r in elig), default=0.0)
        sum_R_elig = sum(r["r_multiple"] for r in elig)
        print(f"    MFE<{thr:.1f}R : {len(elig):>2} trades "
              f"contributing total R={sum_R_elig:>+6.2f} "
              f"(worst single R={worst_R:+6.2f})")

    print()
    print(f"  Counterfactual sweep:")
    print(f"  {'thr':>5}  {'assume':>7}  {'replaced':>9}  {'saved_R':>8}  "
          f"{'new sumR':>9}  {'new WR':>7}  {'new avgR':>9}  {'new PF':>7}")
    print(f"  {'-' * 88}")
    for thr in MFE_THRESHOLDS:
        for ar in ASSUMED_EXIT_R:
            s = simulate(rows, thr, ar)
            delta_sumR = s["sumR"] - base["sumR"]
            print(f"  {thr:>5.2f}  {ar:>+7.2f}  {s['n_replaced']:>9}  "
                  f"{s['saved_R']:>+8.2f}  {s['sumR']:>+9.2f}  "
                  f"{s['wr']*100:>6.1f}%  {s['avgR']:>+9.3f}  {s['pf']:>7.3f}")


def main() -> None:
    rows = load()
    pre  = [r for r in rows if r["trade_id"] <  SPLIT_ID]
    post = [r for r in rows if r["trade_id"] >= SPLIT_ID]

    print("=" * 90)
    print(f"  ARKAD MRK — Time-exit early-close counterfactual")
    print(f"  CSV: {CSV_PATH}   |   Split at trade_id = {SPLIT_ID}")
    print("=" * 90)

    print_table("ALL trades",     rows)
    print_table("PRE-FIX  (id<37)", pre)
    print_table("POST-FIX (id>=37)", post)

    print()
    print(f"  Reading guide:")
    print(f"    'thr'      = MFE threshold below which a time-exit is treated")
    print(f"                 as 'going nowhere' and force-closed early")
    print(f"    'assume'   = the R-value the early close is assumed to realize")
    print(f"                 0.0 ≈ break-even, -0.5 ≈ half-stop")
    print(f"    'replaced' = how many trades had their R rewritten")
    print(f"    'saved_R'  = sum(new_R) - sum(original_R) over replaced trades")
    print()


if __name__ == "__main__":
    main()

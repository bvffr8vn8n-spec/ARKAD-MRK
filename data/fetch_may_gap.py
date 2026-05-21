"""
data/fetch_may_gap.py — One-shot: fetch April 25 → May 18, 2026 for Tier-1
                       (AVAX, ADA, SOL, XRP) at 1H and 5m, merge into the
                       existing data/<SYM>USDT_recent_{1h,5m}.csv files.

Idempotent: re-running just refreshes the tail.
"""

import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bybit_loader import fetch_klines, save_klines

TIER1 = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]

# Overlap by ~36 h to be safe; merge keeps last duplicate.
START = datetime(2026, 4, 25, tzinfo=timezone.utc)
END   = datetime(2026, 5, 18, 23, 59, 59, tzinfo=timezone.utc)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _merge_into(existing_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(existing_path):
        old = pd.read_csv(existing_path, parse_dates=["date"]).set_index("date")
    else:
        old = pd.DataFrame()
    merged = pd.concat([old, new_df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _save(df: pd.DataFrame, path: str) -> None:
    df = df.reset_index().rename(columns={"index": "date"})
    df.to_csv(path, index=False)


def main() -> None:
    for sym in TIER1:
        for interval, suffix in (("60", "1h"), ("5", "5m")):
            print(f"  {sym}  {suffix} ...", end=" ", flush=True)
            df_new = fetch_klines(symbol=sym, category="linear",
                                  interval=interval, start=START, end=END)
            out = os.path.join(DATA_DIR, f"{sym}_recent_{suffix}.csv")
            merged = _merge_into(out, df_new)
            _save(merged, out)
            print(f"new={len(df_new):,}  total={len(merged):,}  "
                  f"last={merged.index[-1]}")
    print("\nDone.")


if __name__ == "__main__":
    main()

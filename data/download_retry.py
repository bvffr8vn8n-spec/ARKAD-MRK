"""
data/download_retry.py — Retry downloader for missing 4y data files.

Checks which files from the ARKAD candidate pool are missing and downloads
them one by one, with automatic retry on transient connection errors.

Usage
-----
  python data/download_retry.py              # check and download missing
  python data/download_retry.py --force      # re-download all
  python data/download_retry.py --pairs SOLUSDT,ETHUSDT  # specific pairs
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bybit_loader import fetch_klines, save_klines

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

CANDIDATES = [
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

SYMBOL_FALLBACK = {"MATICUSDT": "POLUSDT"}

YEARS_BACK = 4
MAX_RETRIES = 3
RETRY_DELAY = 5   # seconds between retries


def _date_range():
    now   = datetime.now(tz=timezone.utc)
    end   = now.replace(hour=23, minute=59, second=59, microsecond=0)
    start = end.replace(year=end.year - YEARS_BACK, hour=0, minute=0, second=0, microsecond=0)
    return start, end


def _download_with_retry(
    symbol: str, interval: str, interval_tag: str,
    start: datetime, end: datetime, force: bool,
) -> bool:
    out_path = os.path.join(DATA_DIR, f"{symbol}_{interval_tag}_4y.csv")
    tmp_path = out_path + ".tmp"

    if not force and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"  [SKIP]  {os.path.basename(out_path)}  ({size_mb:.1f} MB)")
        return True

    candidates = [symbol]
    if symbol in SYMBOL_FALLBACK:
        candidates.append(SYMBOL_FALLBACK[symbol])

    for candidate in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  Downloading {candidate}  interval={interval_tag}  "
                  f"{start.date()} -> {end.date()}  (attempt {attempt}/{MAX_RETRIES})",
                  flush=True)
            t0 = time.time()
            try:
                df = fetch_klines(
                    symbol   = candidate,
                    category = "linear",
                    interval = interval,
                    start    = start,
                    end      = end,
                )
                save_klines(df, tmp_path)
                os.replace(tmp_path, out_path)
                elapsed = time.time() - t0
                size_mb = os.path.getsize(out_path) / 1_048_576
                print(f"  OK  {len(df):,} bars -> {os.path.basename(out_path)} "
                      f"({size_mb:.1f} MB, {elapsed:.0f}s)")
                return True
            except (ValueError, RuntimeError) as exc:
                print(f"  ERROR ({candidate}): {exc}")
                if candidate != candidates[-1]:
                    break   # try fallback ticker immediately
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s ...")
                    time.sleep(RETRY_DELAY)
            except Exception as exc:
                print(f"  UNEXPECTED ({candidate}): {exc}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s ...")
                    time.sleep(RETRY_DELAY)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"  FAILED: {symbol} {interval_tag} — all retries exhausted")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry downloader for ARKAD data.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pairs", default=None, metavar="SYM1,SYM2,...")
    args = parser.parse_args()

    pairs = (
        [p.strip().upper() for p in args.pairs.split(",")]
        if args.pairs else CANDIDATES
    )

    start, end = _date_range()

    print(f"\n{'='*60}")
    print(f"  ARKAD MRK — Retry Data Downloader")
    print(f"  Range  : {start.date()} -> {end.date()}  ({YEARS_BACK}y)")
    print(f"  Pairs  : {', '.join(pairs)}")
    print(f"  Force  : {args.force}")
    print(f"{'='*60}\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    results = []
    for pair in pairs:
        print(f"\n--- {pair} ---")
        ok_1h = _download_with_retry(pair, "60", "1h", start, end, args.force)
        ok_5m = _download_with_retry(pair, "5",  "5m", start, end, args.force)
        results.append((pair, ok_1h, ok_5m))

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"  {'Pair':<14}  {'1H':>6}  {'5m':>6}")
    print(f"  {'-'*30}")
    for pair, ok1, ok5 in results:
        print(f"  {pair:<14}  {'OK' if ok1 else 'FAIL':>6}  {'OK' if ok5 else 'FAIL':>6}")
    print(f"{'='*60}\n")

    failed = [(p, ok1, ok5) for p, ok1, ok5 in results if not ok1 or not ok5]
    if failed:
        print("Failed:")
        for p, ok1, ok5 in failed:
            parts = []
            if not ok1: parts.append("1H")
            if not ok5: parts.append("5m")
            print(f"  {p}: {', '.join(parts)}")
        sys.exit(1)
    print("All downloads completed successfully.")


if __name__ == "__main__":
    main()

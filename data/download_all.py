"""
data/download_all.py — Batch-download 1H and 5m historical klines for the
top 10 most liquid USDT perpetual pairs from the Bybit V5 public API.

Covers approximately 4 years of history ending today.

Output files (project-root-relative):
    data/<PAIR>_1h_4y.csv
    data/<PAIR>_5m_4y.csv

Columns: date, open, high, low, close, volume  (same format as load_ohlcv())

Usage
-----
    # Download everything (skip files that already exist)
    python data/download_all.py

    # Force re-download even if files exist
    python data/download_all.py --force

    # Download only specific pairs
    python data/download_all.py --pairs BTCUSDT,ETHUSDT

Notes
-----
* MATICUSDT (Polygon) was rebranded to POLUSDT on Bybit in late 2024.
  The script tries MATICUSDT first; if Bybit returns no data it
  automatically falls back to POLUSDT and saves the file as MATICUSDT_*.csv
  so the rest of the pipeline needs no special-casing.

* 5m data over 4 years = ~420 K candles per pair (~420 API pages).
  With the built-in 0.2 s page delay the download takes roughly 2-3 minutes
  per pair.  All 10 pairs complete in 20-30 minutes.

* Each file is written atomically via a .tmp file so a mid-download
  interruption leaves no corrupt CSV on disk.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

# Allow running as  python data/download_all.py  from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bybit_loader import fetch_klines, save_klines

# ── Constants ─────────────────────────────────────────────────────────────────

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
    "MATICUSDT",   # fallback → POLUSDT if unavailable under original ticker
]

# Primary → fallback ticker map (rebrands / renames on Bybit)
SYMBOL_FALLBACK: dict[str, str] = {
    "MATICUSDT": "POLUSDT",
}

CATEGORY    = "linear"   # USDT perpetual futures
INTERVAL_1H = "60"       # 60-minute bars
INTERVAL_5M = "5"        # 5-minute bars
YEARS_BACK  = 4

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_range() -> tuple[datetime, datetime]:
    """Return (start, end) covering the last YEARS_BACK years (UTC)."""
    now   = datetime.now(tz=timezone.utc)
    end   = now.replace(hour=23, minute=59, second=59, microsecond=0)
    start = end.replace(year=end.year - YEARS_BACK, hour=0, minute=0, second=0, microsecond=0)
    return start, end


def _out_path(pair: str, interval_tag: str) -> str:
    return os.path.join(DATA_DIR, f"{pair}_{interval_tag}_4y.csv")


def _download_one(
    pair:         str,
    interval:     str,
    interval_tag: str,
    start:        datetime,
    end:          datetime,
    force:        bool,
) -> bool:
    """
    Download klines for one (pair, interval) combination.

    * Tries the primary ticker first; falls back to SYMBOL_FALLBACK if needed.
    * Writes atomically: data goes to a .tmp file that is renamed on success.
    * Returns True on success (including skipped-existing), False on failure.
    """
    out_path = _out_path(pair, interval_tag)
    tmp_path = out_path + ".tmp"

    if not force and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"    [SKIP] {os.path.basename(out_path)} already exists ({size_mb:.1f} MB)")
        return True

    candidates = [pair]
    if pair in SYMBOL_FALLBACK:
        candidates.append(SYMBOL_FALLBACK[pair])

    for candidate in candidates:
        print(f"    Downloading {candidate}  interval={interval_tag}  "
              f"{start.date()} → {end.date()} ...", flush=True)
        t0 = time.time()
        try:
            df = fetch_klines(
                symbol   = candidate,
                category = CATEGORY,
                interval = interval,
                start    = start,
                end      = end,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"    WARNING: {candidate} failed — {exc}")
            if candidate != candidates[-1]:
                print(f"    Retrying with fallback ticker: {candidates[-1]} ...")
            continue
        except Exception as exc:
            print(f"    WARNING: unexpected error for {candidate} — {exc}")
            if candidate != candidates[-1]:
                print(f"    Retrying with fallback ticker: {candidates[-1]} ...")
            continue

        # Write to tmp then rename (atomic on same filesystem)
        save_klines(df, tmp_path)
        os.replace(tmp_path, out_path)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(out_path) / 1_048_576
        print(f"    Saved {len(df):,} bars → {os.path.basename(out_path)} "
              f"({size_mb:.1f} MB, {elapsed:.0f}s)")
        return True

    # All candidates failed
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"    FAILED: could not download {pair} {interval_tag}")
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-download 4Y OHLCV data for top-10 USDT perp pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/download_all.py\n"
            "  python data/download_all.py --force\n"
            "  python data/download_all.py --pairs BTCUSDT,ETHUSDT --force"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and overwrite files that already exist.",
    )
    parser.add_argument(
        "--pairs", default=None,
        metavar="SYM1,SYM2,...",
        help="Comma-separated list of symbols (default: all 10).",
    )
    args = parser.parse_args()

    pairs = (
        [p.strip().upper() for p in args.pairs.split(",")]
        if args.pairs else PAIRS
    )

    start, end = _date_range()

    print(f"\n{'='*60}")
    print(f"  ARKAD MRK — Batch Data Download")
    print(f"{'='*60}")
    print(f"  Pairs    : {', '.join(pairs)}")
    print(f"  Range    : {start.date()} → {end.date()}  ({YEARS_BACK} years)")
    print(f"  Category : {CATEGORY}")
    print(f"  Force    : {args.force}")
    print(f"  Data dir : {DATA_DIR}\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    results: list[tuple[str, bool, bool]] = []   # (pair, ok_1h, ok_5m)

    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}]  {pair}")
        print(f"  {'─'*56}")

        ok_1h = _download_one(pair, INTERVAL_1H, "1h", start, end, args.force)
        ok_5m = _download_one(pair, INTERVAL_5M, "5m", start, end, args.force)

        results.append((pair, ok_1h, ok_5m))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Download Summary")
    print(f"{'='*60}")
    print(f"  {'Pair':<14}  {'1H':>6}  {'5m':>6}")
    print(f"  {'-'*30}")
    for pair, ok_1h, ok_5m in results:
        s1 = "OK  " if ok_1h else "FAIL"
        s5 = "OK  " if ok_5m else "FAIL"
        print(f"  {pair:<14}  {s1:>6}  {s5:>6}")
    print(f"{'='*60}\n")

    failed = [p for p, ok1, ok5 in results if not ok1 or not ok5]
    if failed:
        print(f"  Failed pairs: {', '.join(failed)}")
        print(f"  Re-run with --force to retry failed downloads.\n")
        sys.exit(1)

    print("  All downloads completed successfully.\n")


if __name__ == "__main__":
    main()

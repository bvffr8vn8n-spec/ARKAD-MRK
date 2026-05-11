"""
data/download_bybit.py — CLI script to download Bybit historical klines to CSV.

The output CSV is directly compatible with the ARKAD pipeline's load_ohlcv().

Usage
-----
From the project root:

    python data/download_bybit.py --symbol BTCUSDT --category linear \\
        --interval 60 --start 2024-01-01 --end 2024-12-31 \\
        --out data/BTCUSDT_1h.csv

Then run the pipeline on the downloaded file:

    python main.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT

Interval reference
------------------
  Minutes : 1  3  5  15  30  60  120  240  360  720
  Daily   : D
  Weekly  : W
  Monthly : M
"""

import argparse
import os
import sys
from datetime import datetime, timezone

# Ensure the project root is on sys.path so "data.bybit_loader" resolves
# whether this script is run as:
#   python data/download_bybit.py          (script mode)
#   python -m data.download_bybit          (module mode from project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bybit_loader import fetch_klines, save_klines, VALID_INTERVALS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV klines from the Bybit V5 public API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/download_bybit.py --symbol BTCUSDT --interval 60 "
            "--start 2024-01-01 --end 2024-12-31 --out data/BTCUSDT_1h.csv\n"
            "  python data/download_bybit.py --symbol ETHUSDT --category linear "
            "--interval D --start 2023-01-01 --end 2024-01-01 --out data/ETH_daily.csv"
        ),
    )
    parser.add_argument(
        "--symbol", required=True,
        help="Trading pair symbol, e.g. BTCUSDT, ETHUSDT.",
    )
    parser.add_argument(
        "--category", default="linear",
        choices=["linear", "inverse", "spot"],
        help="Market category (default: linear).",
    )
    parser.add_argument(
        "--interval", required=True,
        metavar="{" + "|".join(sorted(VALID_INTERVALS, key=lambda x: (len(x), x))) + "}",
        help="Bar width in minutes (numeric) or D/W/M.",
    )
    parser.add_argument(
        "--start", required=True,
        metavar="YYYY-MM-DD",
        help="Inclusive start date in UTC.",
    )
    parser.add_argument(
        "--end", required=True,
        metavar="YYYY-MM-DD",
        help="Inclusive end date in UTC.  The end date's candle is included.",
    )
    parser.add_argument(
        "--out", required=True,
        metavar="PATH",
        help="Output CSV file path, e.g. data/BTCUSDT_1h.csv.",
    )
    return parser.parse_args()


def _parse_date(date_str: str, label: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Error: --{label} '{date_str}' is not a valid YYYY-MM-DD date.")
        sys.exit(1)


def main() -> None:
    args = _parse_args()

    start = _parse_date(args.start, "start")
    # Set end to 23:59:59 on the specified day so the final day's candles are included
    end   = _parse_date(args.end, "end").replace(hour=23, minute=59, second=59)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        print(f"Error: output directory does not exist: {out_dir}")
        sys.exit(1)

    print(
        f"\nBybit Historical Kline Download"
        f"\n  Symbol   : {args.symbol.upper()}"
        f"\n  Category : {args.category}"
        f"\n  Interval : {args.interval}"
        f"\n  Range    : {args.start}  ->  {args.end}"
        f"\n  Output   : {args.out}\n"
    )

    try:
        df = fetch_klines(
            symbol   = args.symbol,
            category = args.category,
            interval = args.interval,
            start    = start,
            end      = end,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        sys.exit(1)

    save_klines(df, args.out)

    print(
        f"\nDone."
        f"\n  Candles saved : {len(df):,}"
        f"\n  First bar     : {df.index[0]}"
        f"\n  Last bar      : {df.index[-1]}"
        f"\n  File          : {args.out}\n"
    )


if __name__ == "__main__":
    main()

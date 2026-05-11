"""
data/bybit_loader.py — Downloads historical OHLCV klines from the Bybit V5 API.

Public endpoint (no authentication required):
    GET https://api.bybit.com/v5/market/kline

Pagination model
----------------
Bybit returns candles in descending order (newest first), up to 1000 per
request.  To cover a range longer than 1000 bars, each subsequent page sets
    end = oldest_timestamp_in_previous_page - 1ms
stepping backwards until one of three stop conditions is met:
    1. Bybit returns fewer than `limit` candles  (no more data)
    2. The oldest candle in the page is at or before the requested start
    3. Bybit returns an empty list

All pages are accumulated, deduplicated, filtered to [start, end], sorted
ascending, and returned as a clean DataFrame.

Output format
-------------
DatetimeIndex  named "date", columns: open  high  low  close  volume (all float).
save_klines() writes this as a CSV that load_ohlcv() can consume directly.
"""

import hashlib
import hmac
import logging
import time
import requests
import pandas as pd
from datetime import datetime

from data.http_resilient import request_with_retry, DEFAULT_RETRY, BYBIT_BREAKER

log = logging.getLogger(__name__)

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"

VALID_INTERVALS = {
    "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
}

# Seconds to wait between pages — keeps usage well inside the public rate limit
_PAGE_DELAY_S = 0.2


def _auth_headers(params: dict) -> dict:
    """
    Build Bybit V5 HMAC-SHA256 authentication headers.
    Returns empty dict if secrets are not configured.
    """
    try:
        from paper_trading.secrets import BYBIT_API_KEY, BYBIT_API_SECRET
    except ImportError:
        return {}

    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        return {}

    ts          = str(int(time.time() * 1000))
    recv_window = "5000"
    query_str   = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sign_str    = ts + BYBIT_API_KEY + recv_window + query_str
    signature   = hmac.new(
        BYBIT_API_SECRET.encode("utf-8"),
        sign_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-BAPI-API-KEY":    BYBIT_API_KEY,
        "X-BAPI-SIGN":       signature,
        "X-BAPI-TIMESTAMP":  ts,
        "X-BAPI-RECV-WINDOW": recv_window,
    }


def fetch_klines(
    symbol:   str,
    category: str,
    interval: str,
    start:    datetime,
    end:      datetime,
    limit:    int = 1000,
) -> pd.DataFrame:
    """
    Download all historical klines for the given symbol and date range.

    Paginates automatically until the full range is covered.

    Parameters
    ----------
    symbol   : Trading pair, e.g. "BTCUSDT".
    category : Market type — "linear", "inverse", or "spot".
    interval : Bar width.  Must be one of VALID_INTERVALS.
               Numeric strings for minutes: "1","3","5","15","30","60",
               "120","240","360","720".  String literals: "D","W","M".
    start    : Inclusive start of the range (UTC, timezone-naive or UTC-aware).
    end      : Inclusive end of the range (UTC, timezone-naive or UTC-aware).
    limit    : Candles per page.  Maximum and default is 1000.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex named "date" (naive UTC), float columns:
        open, high, low, close, volume.
        Sorted chronologically ascending.

    Raises
    ------
    ValueError  : bad interval, start >= end, or no candles in range.
    RuntimeError: Bybit API returned a non-zero retCode.
    requests.HTTPError: HTTP-level error from the server.
    """
    if str(interval) not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Valid values: {sorted(VALID_INTERVALS, key=lambda x: (len(x), x))}"
        )

    limit = min(max(1, limit), 1000)

    # Convert datetimes to UTC millisecond timestamps
    start_ms = _to_ms(start)
    end_ms   = _to_ms(end)

    if start_ms >= end_ms:
        raise ValueError(f"start ({start}) must be before end ({end})")

    all_candles: list[list] = []
    page_end_ms = end_ms
    page_num    = 0

    while True:
        page_num += 1
        params = {
            "category": category,
            "symbol":   symbol.upper(),
            "interval": str(interval),
            "start":    start_ms,
            "end":      page_end_ms,
            "limit":    limit,
        }

        # Public market endpoint — do NOT send auth headers.
        # Bybit applies a stricter ACCOUNT-level rate limit on authenticated
        # requests; the IP-level limit on unauthenticated public market data
        # is far more generous and is what we want here.
        response = request_with_retry(
            BYBIT_KLINE_URL,
            params=params,
            headers={},
            timeout=20,
        )
        if response is None:
            # All retries exhausted, or circuit breaker is OPEN.
            # Treat as transient: return whatever we have so the caller can
            # try again on the next poll tick.  Logged inside request_with_retry.
            log.warning(
                "fetch_klines: giving up page %d for %s after retries/breaker",
                page_num, symbol,
            )
            break

        response.raise_for_status()

        body = response.json()
        rc = body.get("retCode")
        # 10006 = Bybit rate limit ("Too many visits").  Retrying immediately
        # makes it worse — trip the circuit breaker for a long cooldown so the
        # whole process backs off cleanly.
        if rc == 10006:
            BYBIT_BREAKER.trip(
                reason="retCode 10006 (rate limit)",
                cooldown_s=300.0,   # 5-minute lockout
            )
            log.error(
                "Bybit rate limit hit (10006) on %s — circuit breaker tripped "
                "for 5 min; aborting fetch.",
                symbol,
            )
            break
        if rc != 0:
            raise RuntimeError(
                f"Bybit API error {rc}: {body.get('retMsg')}"
            )

        candles = body["result"]["list"]  # list of lists, descending order

        # Stop condition 3: empty response
        if not candles:
            break

        oldest_ts = int(candles[-1][0])
        newest_ts = int(candles[0][0])
        print(f"  Page {page_num}: {len(candles):>4} candles  "
              f"{_ms_to_str(oldest_ts)} -> {_ms_to_str(newest_ts)}")

        all_candles.extend(candles)

        # Stop condition 1: partial page means Bybit has no earlier data
        if len(candles) < limit:
            break

        # Stop condition 2: oldest candle is at or before the requested start
        if oldest_ts <= start_ms:
            break

        # Advance the window backwards, just before the oldest candle this page
        page_end_ms = oldest_ts - 1

        time.sleep(_PAGE_DELAY_S)

    if not all_candles:
        raise ValueError(
            f"No candles returned for {symbol} interval={interval} "
            f"[{_ms_to_str(start_ms)} - {_ms_to_str(end_ms)}]. "
            f"Check that the symbol and category are correct."
        )

    return _build_dataframe(all_candles, start_ms, end_ms)


def save_klines(df: pd.DataFrame, out_path: str) -> None:
    """
    Write a klines DataFrame to CSV in the format expected by load_ohlcv().

    Output columns: date, open, high, low, close, volume
    The "date" column is written as an ISO datetime string (YYYY-MM-DD HH:MM:SS)
    which pandas parse_dates can read without any configuration.

    Parameters
    ----------
    df       : DataFrame returned by fetch_klines() (DatetimeIndex + OHLCV columns).
    out_path : Destination file path.  Parent directory must exist.
    """
    df_out = df.reset_index()  # DatetimeIndex "date" → regular column
    df_out.to_csv(out_path, index=False, encoding="utf-8")


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_dataframe(
    raw_candles: list[list],
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Convert the accumulated raw candle lists into a clean OHLCV DataFrame.

    Steps:
      1. Deduplicate by timestamp (pagination overlap guard).
      2. Filter to [start_ms, end_ms].
      3. Convert timestamps to naive UTC pd.Timestamp.
      4. Cast OHLCV columns to float.
      5. Sort ascending by date.
    """
    # Deduplicate: use the first occurrence of each timestamp
    seen:   set[int] = set()
    unique: list[list] = []
    for candle in raw_candles:
        ts = int(candle[0])
        if ts not in seen:
            seen.add(ts)
            unique.append(candle)

    # Bybit candle layout: [startTime, open, high, low, close, volume, turnover]
    records = []
    for candle in unique:
        ts = int(candle[0])
        if ts < start_ms or ts > end_ms:
            continue  # discard stray candles outside the requested window
        records.append({
            "date":   pd.Timestamp(ts, unit="ms"),  # naive UTC — matches pipeline
            "open":   float(candle[1]),
            "high":   float(candle[2]),
            "low":    float(candle[3]),
            "close":  float(candle[4]),
            "volume": float(candle[5]),
        })

    if not records:
        raise ValueError(
            "All downloaded candles were outside the requested date range after filtering."
        )

    df = (
        pd.DataFrame(records)
        .sort_values("date")
        .set_index("date")
    )
    df.index.name = "date"
    return df


def _to_ms(dt: datetime) -> int:
    """Convert a datetime to a UTC millisecond integer timestamp."""
    # datetime.timestamp() is UTC-correct for aware datetimes;
    # for naive datetimes it treats the value as local time on most systems.
    # We always treat naive inputs as UTC by using timegm on the UTC tuple.
    import calendar
    if dt.tzinfo is not None:
        # Aware datetime: convert to UTC epoch via timestamp()
        return int(dt.timestamp() * 1000)
    else:
        # Naive datetime: assume UTC
        return int(calendar.timegm(dt.timetuple()) * 1000)


def _ms_to_str(ts_ms: int) -> str:
    """Format a UTC millisecond timestamp as a human-readable string."""
    return datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M")

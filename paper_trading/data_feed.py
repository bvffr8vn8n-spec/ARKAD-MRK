"""
paper_trading/data_feed.py — Live market data feed backed by the Bybit V5 API.

Wraps data.bybit_loader.fetch_klines with:
  - Sensible default windows (WARMUP_1H_BARS for 1H, FETCH_15M_BARS for 15m)
  - Suppressed page-by-page print output from the downloader
  - Error handling: returns None on failure (caller decides what to do)

Usage
-----
    feed = LiveFeed()
    df_1h  = feed.get_1h_bars("AVAXUSDT")    # last WARMUP_1H_BARS + 1 bars
    df_15m = feed.get_15m_bars("AVAXUSDT")   # last FETCH_15M_BARS bars
"""

import io
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from paper_trading import config_live
from paper_trading.logger import get_logger

log = get_logger()


class LiveFeed:
    """Thin wrapper around fetch_klines for live-polling use."""

    def get_1h_bars(
        self,
        symbol: str,
        n_bars: int = config_live.FETCH_1H_BARS_LIVE,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch the most recent `n_bars` completed 1H bars for `symbol`.

        Default of FETCH_1H_BARS_LIVE (~5 bars) is enough to cover the new
        bar each tick plus a few-hour outage cushion.  The full WARMUP_1H_BARS
        history is loaded once at startup by the signal engine from the local
        CSV — we do NOT re-fetch 13 days of bars on every poll.

        Returns a DataFrame (DatetimeIndex "date", cols: open/high/low/close/volume)
        sorted ascending, or None on error.

        The end of the range is set to `now - 2 minutes` to avoid pulling a
        still-forming candle.
        """
        return self._fetch(symbol, config_live.INTERVAL_1H, n_bars, bar_minutes=60)

    def get_15m_bars(
        self,
        symbol: str,
        n_bars: int = config_live.FETCH_15M_BARS_LIVE,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch the most recent `n_bars` completed 15m bars for `symbol`.

        Default of FETCH_15M_BARS_LIVE (~10 bars = 2.5 h) covers normal tick
        cadence plus a few-hour gap.  Returns None on error.
        """
        return self._fetch(symbol, config_live.INTERVAL_15M, n_bars, bar_minutes=15)

    def get_ticker_price(self, symbol: str) -> Optional[dict]:
        """
        Fetch the current real-time market price from the Bybit ticker endpoint.

        This is the ONLY correct source for paper-trade entry prices.
        Candle close prices (from get_1h_bars / get_15m_bars) reflect prices
        that were valid at bar-close time, not at the moment of trade entry.

        Returns a dict with keys:
            last : float  — last traded price
            bid  : float  — best bid (you receive this when selling / going short)
            ask  : float  — best ask (you pay this when buying / going long)

        Returns None on any network or API error (caller must handle and skip entry).
        Uses the resilient HTTP layer: retries on transient failures, respects
        the global Bybit circuit breaker.

        Endpoint
        --------
        GET https://api.bybit.com/v5/market/tickers
            ?category=linear&symbol=<SYMBOL>
        """
        from data.http_resilient import request_with_retry, BYBIT_BREAKER
        try:
            params = {
                "category": config_live.BYBIT_CATEGORY,
                "symbol":   symbol.upper(),
            }
            # Public ticker endpoint — no auth headers (see bybit_loader.py
            # for the rate-limit rationale).
            resp = request_with_retry(
                "https://api.bybit.com/v5/market/tickers",
                params=params,
                headers={},
                timeout=10,
            )
            if resp is None:
                # Retries exhausted or circuit breaker is OPEN.
                log.warning("get_ticker_price: no response for %s "
                            "(retries exhausted or breaker open)", symbol)
                return None
            resp.raise_for_status()
            body = resp.json()
            rc = body.get("retCode")
            if rc == 10006:
                BYBIT_BREAKER.trip(
                    reason="retCode 10006 (ticker rate limit)",
                    cooldown_s=300.0,
                )
                log.error(
                    "Bybit rate limit (10006) on ticker %s — circuit breaker "
                    "tripped for 5 min.", symbol,
                )
                return None
            if rc != 0:
                log.error(
                    "Ticker API error for %s: retCode=%s  msg=%s",
                    symbol, rc, body.get("retMsg"),
                )
                return None
            items = body["result"]["list"]
            if not items:
                log.warning("Ticker API returned empty list for %s", symbol)
                return None
            t = items[0]
            return {
                "last": float(t["lastPrice"]),
                "bid":  float(t["bid1Price"]),
                "ask":  float(t["ask1Price"]),
            }
        except Exception as exc:
            log.error("get_ticker_price failed for %s: %s", symbol, exc)
            return None

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch(
        self,
        symbol: str,
        interval: str,
        n_bars: int,
        bar_minutes: int,
    ) -> Optional[pd.DataFrame]:
        """Generic fetcher: calculates a time range that covers n_bars and downloads."""
        # Import here to avoid circular imports at module level
        from data.bybit_loader import fetch_klines

        now = datetime.utcnow()
        # End: 2 minutes before now to avoid the currently-forming bar
        end   = now - timedelta(minutes=2)
        # Start: generous lookback so we always get >= n_bars
        start = end - timedelta(minutes=bar_minutes * (n_bars + 5))

        try:
            # fetch_klines prints one line per API page — suppress that noise
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                df = fetch_klines(
                    symbol   = symbol,
                    category = config_live.BYBIT_CATEGORY,
                    interval = interval,
                    start    = start,
                    end      = end,
                )
            finally:
                sys.stdout = old_stdout

            if df is None or len(df) == 0:
                log.warning("Empty response for %s %s-min", symbol, bar_minutes)
                return None

            # Return only the last n_bars
            return df.iloc[-n_bars:].copy()

        except Exception as exc:
            log.error("fetch_klines failed for %s %s: %s", symbol, interval, exc)
            return None

"""
paper_trading/poller.py — Main real-time polling loop for the paper trader.

Architecture
------------
Every POLL_INTERVAL_S seconds the loop wakes up and:

  1. Detects new completed 1H bars (by comparing bar timestamps to stored state).
     For each new 1H bar per asset:
       a. Push bar to SignalEngine buffer
       b. Score the bar → signal / buy_prob / atr
       c. If signal != 0 and no open trade / active monitor: start A_WATCHING
       d. Maybe retrain model

  2. Detects new completed 15m bars (by comparing bar timestamps to stored state).
     For each new 15m bar per asset:
       a. Advance the A/B monitor (state_machine.advance_monitor)
       b. If monitor just reached ENTERED: open a trade (trade_manager.open_trade)
       c. If monitor reached CANCELLED: clear it
       d. Update open trade (trade_manager.update_bar)
       e. If trade just closed: write CSV row + save state

  3. Saves state to disk after processing any bar events.

Timestamp conventions
---------------------
  - All datetimes are naive UTC pd.Timestamps or Python datetime objects.
  - 1H bar timestamps are the bar's open time (e.g., 14:00 for a bar covering
    14:00–14:59).  A bar labeled T is "closed" when T + 1H <= now.
  - 15m bars follow the same convention with a 15-minute window.

Restart recovery
----------------
  On startup the state file is loaded.  If the last processed 1H bar for an
  asset is several hours old, the poller fetches and replays all intervening
  bars in chronological order before entering the live loop.  This ensures
  monitors and trades are up-to-date after a restart.

Duplicate prevention
--------------------
  Three independent layers:
    1. processed_1h_bars / processed_15m_bars timestamps: a bar is only
       processed if its open timestamp > stored timestamp.
    2. SignalEngine.push_bar: skips if bar timestamp already in buffer.
    3. One monitor and one trade per asset at a time (dict keyed by asset).
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from paper_trading import config_live
from paper_trading.csv_writer import CsvWriter
from paper_trading.data_feed import LiveFeed
from paper_trading.logger import get_logger
from paper_trading.signal_engine import SignalEngine
from paper_trading.state_machine import MonitorState, advance_monitor
from paper_trading.state_store import StateStore
from paper_trading.trade_manager import TradeManager, TradeState

log = get_logger()


# ── Public entry point ────────────────────────────────────────────────────────

def run_poll_loop(
    engine: SignalEngine,
    store: StateStore,
    feed: LiveFeed,
    csv_writer: CsvWriter,
) -> None:
    """
    Blocking main loop.  Runs until interrupted (KeyboardInterrupt / SIGINT).

    The caller constructs all dependencies and passes them in; this makes the
    loop unit-testable without a live network connection.
    """
    log.info("Poll loop started.  Monitoring: %s", ", ".join(config_live.ASSETS))
    log.info("Poll interval: %ds | 1H warmup bars: %d | assets: %d",
             config_live.POLL_INTERVAL_S, config_live.WARMUP_1H_BARS, len(config_live.ASSETS))

    while True:
        try:
            _tick(engine, store, feed, csv_writer)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received — shutting down.")
            break
        except Exception as exc:
            log.error("Unexpected error in tick: %s", exc, exc_info=True)

        time.sleep(config_live.POLL_INTERVAL_S)


# ── Private ────────────────────────────────────────────────────────────────────

def _tick(
    engine: SignalEngine,
    store: StateStore,
    feed: LiveFeed,
    csv_writer: CsvWriter,
) -> None:
    """One iteration of the polling loop."""
    now    = datetime.utcnow()
    state  = store.load()
    dirty  = False   # will be set True whenever state needs saving

    # Reconstruct in-memory objects from persisted state
    monitors: dict[str, MonitorState] = {
        asset: MonitorState.from_dict(d)
        for asset, d in state.get("monitors", {}).items()
    }
    open_trade_dicts = state.get("open_trades", {})
    open_trade_objs: dict[str, TradeState] = {
        asset: TradeState.from_dict(d)
        for asset, d in open_trade_dicts.items()
    }
    trade_mgr = TradeManager(open_trade_objs)

    # ── Step 1: process new 1H bars ───────────────────────────────────────────
    new_1h_ts = _last_closed_1h_ts(now)

    for asset in config_live.ASSETS:
        stored_ts_str = state["processed_1h_bars"].get(asset, "")
        stored_ts     = _parse_ts(stored_ts_str)

        if new_1h_ts <= stored_ts:
            continue   # no new 1H bar for this asset

        # Fetch the warmup buffer + any bars since last stored
        df_1h = feed.get_1h_bars(asset)
        if df_1h is None or len(df_1h) == 0:
            log.warning("Could not fetch 1H bars for %s", asset)
            continue

        # Find all bars newer than stored_ts, replay in order
        new_bars = df_1h[df_1h.index > stored_ts].sort_index()
        if len(new_bars) == 0:
            continue

        log.debug("%s: %d new 1H bar(s) to process", asset, len(new_bars))

        for ts, row in new_bars.iterrows():
            bar = row.copy()
            bar.name = ts

            # Push to signal engine buffer
            engine.push_bar(asset, bar)

            # Score the bar (only if no open trade and no active monitor)
            has_open_trade   = asset in trade_mgr.open_trades
            has_active_monitor = asset in monitors and monitors[asset].phase not in ("ENTERED", "CANCELLED")

            if not has_open_trade and not has_active_monitor:
                result = engine.score_bar(asset)
                if result is not None and result["signal"] != 0:
                    # Start A+B monitor for this signal
                    atr_dollars = result["atr_pct"] * result["close"]
                    monitor = MonitorState(
                        asset        = asset,
                        signal       = result["signal"],
                        signal_ts    = ts.isoformat(),
                        signal_close = result["close"],
                        atr_1h       = atr_dollars,
                    )
                    monitors[asset] = monitor
                    log.info(
                        "%s  1H signal=%+d  prob=%.3f  close=%.6f  atr=%.6f  → A_WATCHING",
                        asset, result["signal"],
                        result["buy_prob"] if result["signal"] == 1 else result["sell_prob"],
                        result["close"], atr_dollars,
                    )
                    dirty = True

            # Retrain if scheduled
            if engine.should_retrain(asset):
                engine.retrain(asset)

            # Update last processed 1H timestamp
            state["processed_1h_bars"][asset] = ts.isoformat()
            dirty = True

    # ── Step 2: process new 15m bars ──────────────────────────────────────────
    new_15m_ts = _last_closed_15m_ts(now)

    for asset in config_live.ASSETS:
        stored_ts_str = state["processed_15m_bars"].get(asset, "")
        stored_ts     = _parse_ts(stored_ts_str)

        if new_15m_ts <= stored_ts:
            continue   # no new 15m bar

        df_15m = feed.get_15m_bars(asset)
        if df_15m is None or len(df_15m) == 0:
            log.warning("Could not fetch 15m bars for %s", asset)
            continue

        new_bars = df_15m[df_15m.index > stored_ts].sort_index()
        if len(new_bars) == 0:
            continue

        log.debug("%s: %d new 15m bar(s) to process", asset, len(new_bars))

        # Pre-fetch live ticker once per asset tick when a scaled trade is open.
        # Used for time exits in scaled mode so the fill reflects real bid/ask.
        # Only 1 API call per asset per tick — negligible overhead.
        asset_ticker = None
        if (config_live.EXIT_MODE == "scaled"
                and asset in trade_mgr.open_trades):
            asset_ticker = feed.get_ticker_price(asset)
            if asset_ticker is None:
                log.warning(
                    "%s: could not fetch live ticker for scaled exit — "
                    "time exit will fall back to bar close",
                    asset,
                )

        for ts, row in new_bars.iterrows():
            bar = row.copy()
            bar.name = ts

            # ── Advance active monitor ────────────────────────────────────────
            if asset in monitors:
                mon = monitors[asset]
                if mon.phase not in ("ENTERED", "CANCELLED"):
                    mon = advance_monitor(mon, bar)
                    monitors[asset] = mon
                    dirty = True

                    # Monitor just reached ENTERED → open trade at LIVE MARKET PRICE
                    if mon.phase == "ENTERED" and asset not in trade_mgr.open_trades:
                        # ── Fetch real-time price (not candle close) ──────────
                        # The state_machine stored candle.close in mon.entry_price
                        # purely as a pattern-detection anchor.  The actual fill
                        # must come from the live orderbook, not a past candle.
                        ticker = feed.get_ticker_price(asset)
                        if ticker is None:
                            log.error(
                                "%s: live ticker unavailable — skipping entry to "
                                "avoid filling at a stale candle price",
                                asset,
                            )
                            del monitors[asset]
                            dirty = True
                            continue

                        # Taker fill: long buys at ask, short sells at bid
                        candle_ref   = mon.entry_price   # from state_machine (for logging)
                        direction    = mon.signal
                        live_fill    = ticker["ask"] if direction == 1 else ticker["bid"]

                        # Overwrite the candle-derived price with the live fill
                        mon.entry_price = live_fill
                        monitors[asset] = mon

                        log.info(
                            "%s  candle_ref=%.6f  live_%s=%.6f  "
                            "spread_vs_candle=%.4f%%",
                            asset, candle_ref,
                            "ask" if direction == 1 else "bid", live_fill,
                            abs(live_fill - candle_ref) / candle_ref * 100,
                        )

                        tid   = store.next_trade_id(state)
                        trade = trade_mgr.open_trade(
                            monitor=mon,
                            trade_id=tid,
                            equity=state["equity"],
                        )
                        dirty = True

                    # Monitor CANCELLED → remove it
                    elif mon.phase == "CANCELLED":
                        log.info("%s monitor CANCELLED  (A-filter failed)", asset)
                        del monitors[asset]
                        dirty = True

            # ── Update open trade ─────────────────────────────────────────────
            closed_trade = trade_mgr.update_bar(asset, bar, ticker=asset_ticker)
            if closed_trade is not None:
                # Trade just closed: write CSV, update equity
                csv_writer.append(closed_trade)
                pnl = closed_trade.net_pnl_usd or 0.0
                state["equity"] = state.get("equity", config_live.INITIAL_CAPITAL) + pnl
                log.info(
                    "Equity updated: $%.2f  (trade #%d  PnL=$%.2f)",
                    state["equity"], closed_trade.trade_id, pnl,
                )

                # Clear monitor for this asset (trade lifecycle is complete)
                monitors.pop(asset, None)
                dirty = True

            # Update last processed 15m timestamp
            state["processed_15m_bars"][asset] = ts.isoformat()
            dirty = True

    # ── Persist updated state ─────────────────────────────────────────────────
    if dirty:
        # Serialise monitors and open trades back to state dict
        state["monitors"]    = {a: m.to_dict() for a, m in monitors.items()}
        state["open_trades"] = {
            a: t.to_dict()
            for a, t in trade_mgr.open_trades.items()
        }
        store.save(state)


# ── Timestamp helpers ──────────────────────────────────────────────────────────

def _last_closed_1h_ts(now: datetime) -> pd.Timestamp:
    """
    Return the open timestamp of the last fully closed 1H bar.

    Example: now=15:43 → last complete bar opened at 14:00
    """
    current_open = now.replace(minute=0, second=0, microsecond=0)
    return pd.Timestamp(current_open - timedelta(hours=1))


def _last_closed_15m_ts(now: datetime) -> pd.Timestamp:
    """
    Return the open timestamp of the last fully closed 15m bar.

    Example: now=15:43 → last complete bar opened at 15:15
    """
    minutes = (now.minute // 15) * 15
    current_open = now.replace(minute=minutes, second=0, microsecond=0)
    return pd.Timestamp(current_open - timedelta(minutes=15))


def _parse_ts(ts_str: str) -> pd.Timestamp:
    """Parse an ISO timestamp string; return epoch if empty/invalid."""
    if not ts_str:
        return pd.Timestamp("1970-01-01")
    try:
        return pd.Timestamp(ts_str)
    except Exception:
        return pd.Timestamp("1970-01-01")

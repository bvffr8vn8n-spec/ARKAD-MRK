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

import random
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from paper_trading import config_live
from paper_trading import scheduler
from paper_trading.csv_writer import CsvWriter
from paper_trading.data_feed import LiveFeed
from paper_trading.logger import get_logger
from paper_trading.runtime import is_shutdown_requested, shutdown_reason
from paper_trading.signal_engine import SignalEngine
from paper_trading.signal_log import SignalLog
from paper_trading.state_machine import MonitorState, advance_monitor
from paper_trading.state_store import StateStore
from paper_trading.trade_manager import TradeManager, TradeState

log = get_logger()

# If we hit this many consecutive _tick() exceptions, log loudly and pause for
# a longer cooldown rather than spinning hot. Set high enough that ordinary
# Bybit hiccups (handled by http_resilient) never trip it.
_MAX_CONSECUTIVE_TICK_ERRORS = 50
_TICK_ERROR_COOLDOWN_S       = 60.0


# ── Public entry point ────────────────────────────────────────────────────────

def run_poll_loop(
    engine: SignalEngine,
    store: StateStore,
    feed: LiveFeed,
    csv_writer: CsvWriter,
    signal_log: SignalLog,
) -> None:
    """
    Blocking main loop.  Runs until interrupted (KeyboardInterrupt / SIGINT).

    The caller constructs all dependencies and passes them in; this makes the
    loop unit-testable without a live network connection.
    """
    log.info("Poll loop started.  Monitoring: %s", ", ".join(config_live.ASSETS))
    log.info(
        "Scheduler: bar-close wake-up (15m grid + %.0f-%.0fs grace + 0-%.0fs jitter); "
        "1H warmup bars: %d | assets: %d",
        scheduler.GRACE_MIN_S, scheduler.GRACE_MAX_S, scheduler.JITTER_MAX_S,
        config_live.WARMUP_1H_BARS, len(config_live.ASSETS),
    )

    consecutive_errors = 0

    while True:
        # Honour external shutdown request between ticks (SIGTERM / SIGINT
        # handler in paper_trader.py sets the flag in paper_trading.runtime).
        if is_shutdown_requested():
            log.info("Shutdown requested (%s) — exiting poll loop cleanly.",
                     shutdown_reason() or "no reason")
            break

        # Random pre-tick jitter de-aligns this loop from the exact HH:MM:SS
        # second mark where the Bybit IP rate-limit window otherwise gets hit
        # every hour by the bar-close burst from many simultaneous clients.
        if config_live.POLL_JITTER_S > 0:
            time.sleep(random.uniform(0.0, config_live.POLL_JITTER_S))

        try:
            _tick(engine, store, feed, csv_writer, signal_log)
            consecutive_errors = 0   # reset on any successful tick
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received — shutting down.")
            break
        except Exception as exc:
            consecutive_errors += 1
            log.error("Unexpected error in tick (consecutive=%d): %s",
                      consecutive_errors, exc, exc_info=True)

            if consecutive_errors >= _MAX_CONSECUTIVE_TICK_ERRORS:
                log.critical(
                    "Hit %d consecutive tick failures — pausing %.0fs before "
                    "next attempt to avoid log spam.  Investigate root cause.",
                    consecutive_errors, _TICK_ERROR_COOLDOWN_S,
                )
                time.sleep(_TICK_ERROR_COOLDOWN_S)
                continue

        # Sleep until just after the next 15m bar boundary (grace + jitter
        # baked into the scheduler).  Replaces the old fixed POLL_INTERVAL_S
        # cadence: ~4 wake-ups/hour instead of ~40, hourly burst removed by
        # construction.  POLL_JITTER_S above is now redundant safety belt.
        now      = datetime.utcnow()
        sleep_s  = scheduler.seconds_until_next_bar_wake(now)
        log.debug("Next wake in %.1fs (target boundary %s)",
                  sleep_s, scheduler.next_wake_label(now))
        time.sleep(sleep_s)


# ── Private ────────────────────────────────────────────────────────────────────

def _tick(
    engine: SignalEngine,
    store: StateStore,
    feed: LiveFeed,
    csv_writer: CsvWriter,
    signal_log: SignalLog,
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
    fetched_any_1h = False

    for asset in config_live.ASSETS:
        stored_ts_str = state["processed_1h_bars"].get(asset, "")
        stored_ts     = _parse_ts(stored_ts_str)

        if new_1h_ts <= stored_ts:
            continue   # no new 1H bar for this asset

        # Stagger sequential kline requests so 4 symbols don't burst into the
        # same Bybit 5-second IP-rate window (root cause of retCode 10006 at
        # every HH:00 bar-close).  No delay before the first fetch.
        if fetched_any_1h and config_live.INTER_ASSET_DELAY_S > 0:
            time.sleep(config_live.INTER_ASSET_DELAY_S)
        fetched_any_1h = True

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

            # Score the bar UNCONDITIONALLY — needed for live↔backtest parity
            # diagnostics.  The actual monitor-start decision is still gated on
            # no open trade + no active monitor; we just don't want the live
            # trade state to hide what the model would have emitted on this bar.
            has_open_trade     = asset in trade_mgr.open_trades
            has_active_monitor = asset in monitors and monitors[asset].phase not in ("ENTERED", "CANCELLED")

            result = engine.score_bar(asset)
            # Only log when the buffer actually advanced to this bar.  If
            # push_bar was a no-op (this ts was already in the buffer because
            # the CSV seed includes post-cutoff bars and Bybit returned an
            # overlapping bar), score_bar reflects buffer state ending at a
            # LATER ts than `ts` — logging it would mis-label the row.
            buf_tail_ts = engine._buffers[asset].index[-1] if asset in engine._buffers else None
            if result is not None and buf_tail_ts == ts:
                signal_log.append(
                    bar_ts             = ts.isoformat(),
                    asset              = asset,
                    signal             = result["signal"],
                    buy_prob           = result["buy_prob"],
                    sell_prob          = result["sell_prob"],
                    atr_pct            = result["atr_pct"],
                    open_              = float(bar["open"]),
                    high               = float(bar["high"]),
                    low                = float(bar["low"]),
                    close              = result["close"],
                    volume             = float(bar.get("volume", 0.0)),
                    had_open_trade     = has_open_trade,
                    had_active_monitor = has_active_monitor,
                )

            if (not has_open_trade and not has_active_monitor
                    and result is not None and result["signal"] != 0
                    and buf_tail_ts == ts):
                # Start A+B monitor for this signal.  The buf_tail_ts == ts
                # check prevents starting a monitor from a stale-buffer score
                # (push_bar dedup-skipped because the bar was already in seed).
                atr_dollars = result["atr_pct"] * result["close"]
                monitor = MonitorState(
                    asset               = asset,
                    signal              = result["signal"],
                    signal_ts           = ts.isoformat(),
                    signal_close        = result["close"],
                    atr_1h              = atr_dollars,
                    # Entry-state snapshot at SIGNAL time — for lifetime research.
                    signal_buy_prob     = result["buy_prob"],
                    signal_sell_prob    = result["sell_prob"],
                    entry_atr_pct       = result["atr_pct"],
                    entry_vol_ratio     = result.get("entry_vol_ratio"),
                    entry_rsi           = result.get("entry_rsi"),
                    entry_bb_pos        = result.get("entry_bb_pos"),
                    entry_macd_hist     = result.get("entry_macd_hist"),
                    entry_sma_50_ratio  = result.get("entry_sma_50_ratio"),
                    entry_vol_expansion = result.get("entry_vol_expansion"),
                    entry_volume        = result.get("entry_volume"),
                    entry_trend         = result.get("entry_trend"),
                    entry_vol_regime    = result.get("entry_vol_regime"),
                    entry_session       = result.get("entry_session"),
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
    fetched_any_15m = False

    for asset in config_live.ASSETS:
        stored_ts_str = state["processed_15m_bars"].get(asset, "")
        stored_ts     = _parse_ts(stored_ts_str)

        if new_15m_ts <= stored_ts:
            continue   # no new 15m bar

        # Same stagger logic as the 1H loop above — keeps the per-tick burst
        # under the Bybit IP-rate window.
        if fetched_any_15m and config_live.INTER_ASSET_DELAY_S > 0:
            time.sleep(config_live.INTER_ASSET_DELAY_S)
        fetched_any_15m = True

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

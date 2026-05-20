"""
paper_trading/config_live.py — Configuration for the automated paper trader.

All file paths, asset list, API parameters, and execution constants live here.
Edit this file to change assets, sizing, or timing without touching logic files.
"""

import os

# ── Assets ────────────────────────────────────────────────────────────────────
# Tier 1 deployment set: all have Win3 WF PF >= 1.0 with >= 198 WF trades.
# AVAX: WF=1.337 Win3=1.254  |  ADA: WF=1.327 Win3=1.317
# SOL : WF=1.134 Win3=1.109  |  XRP: WF=1.083 Win3=1.071
ASSETS = ["AVAXUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(_ROOT, "data")
PT_DIR     = os.path.join(_ROOT, "paper_trading")

STATE_FILE       = os.path.join(PT_DIR, "state.json")
CSV_FILE         = os.path.join(_ROOT, "paper_trades_tier1.csv")
LOG_FILE         = os.path.join(PT_DIR, "paper_trader.log")
# Per-bar signal log — populated for every scored 1H bar, used by parity test.
SIGNAL_LOG_FILE  = os.path.join(PT_DIR, "signal_log.csv")

# ── Bybit API ─────────────────────────────────────────────────────────────────
BYBIT_CATEGORY = "linear"   # USDT perpetual futures
INTERVAL_1H    = "60"       # 60-minute bars
INTERVAL_15M   = "15"       # 15-minute bars

# ── Signal engine ─────────────────────────────────────────────────────────────
# Number of 1H bars kept in the rolling scoring buffer.
# Must be > SMA-200 warmup (200) + ATR-14 + RSI-14 + safety margin.
WARMUP_1H_BARS = 320

# How many recent 1H / 15m bars to fetch per poll (live polling, NOT warmup).
# We only need enough to cover any gap since the last processed bar; one new
# bar is the normal case, a handful of bars covers brief outages.  Smaller =
# less bandwidth per request, no impact on the trading logic.
FETCH_1H_BARS_LIVE  = 3
FETCH_15M_BARS_LIVE = 6
FETCH_15M_BARS      = FETCH_15M_BARS_LIVE   # back-compat alias

# Poll interval in seconds — how often the main loop wakes up.
# 90s keeps comfortable margin vs. 15m trading granularity (10× margin) and
# de-aligns ticks from the HH:00 second mark where every running instance
# converges (the previous 60s value was bursting at every hour boundary).
POLL_INTERVAL_S = 90

# Pause between symbols within a single poll tick.  Spaces 4 sequential
# kline requests over ~1.6s instead of firing all of them in the same
# Bybit 5-second IP-rate window.  Removes the HH:00 burst that was tripping
# retCode 10006.
INTER_ASSET_DELAY_S = 0.4

# Random jitter (seconds) added at the start of every tick.  Avoids the
# situation where the loop drifts onto exactly HH:00:0X seconds and stays
# there, colliding with the bar-close burst from other IPs / services.
POLL_JITTER_S = 5.0

# Retrain the signal model every N newly processed 1H bars (0 = never retrain).
# At startup the model is always trained from the full historical CSV.
RETRAIN_EVERY_N_BARS = 0   # disabled during paper trading

# ── 15m-AB execution parameters ───────────────────────────────────────────────
A_FILTER_BARS        = 4   # number of 15m bars to evaluate for Approach A
A_FILTER_MIN_ALIGNED = 2   # minimum aligned bars to pass A filter

B_WAIT_BARS = 8            # max 15m bars to scan for pullback (Approach B)

# ── Position sizing and costs ─────────────────────────────────────────────────
INITIAL_CAPITAL    = 10_000.0   # paper-account starting equity ($)
POSITION_SIZE_PCT  = 0.10       # 10% of equity per trade
COMMISSION_PCT     = 0.001      # round-trip commission (0.1% each side)
# Applied on top of the live bid/ask price at entry (not on candle close).
# 6 bps covers spread + market-impact on Bybit USDT perps for mid-cap assets.
SLIPPAGE_PCT       = 0.0006     # 6 bps per side (~0.06%)

# ── ATR-based stop / take-profit ──────────────────────────────────────────────
STOP_LOSS_ATR_MULT   = 1.5      # R = 1.5 × ATR (SL distance)
TAKE_PROFIT_ATR_MULT = 2.5      # TP = 2.5 × ATR  →  R:R = 1.667

# Maximum hold time in hours from entry time (matches HOLD_BARS in config.py).
HOLD_HOURS = 24

# ── Exit mode ──────────────────────────────────────────────────────────────────
# "original" : single TP at TAKE_PROFIT_ATR_MULT × ATR (default, unchanged)
# "scaled"   : three partial exits with BE stop after TP1
EXIT_MODE = "scaled"

# Scaled exit parameters  (used only when EXIT_MODE == "scaled")
# 1R is defined as abs(entry_price - stop_price)
SCALED_TP1_R    = 0.65   # TP1 level in R-units  → close TP1_FRAC of position
SCALED_TP2_R    = 1.00   # TP2 level in R-units  → close TP2_FRAC of position
SCALED_TP3_R    = 1.67   # TP3 level in R-units  → close remaining  (≈ original TP)
SCALED_TP1_FRAC = 0.50   # fraction of original position closed at TP1
SCALED_TP2_FRAC = 0.25   # fraction of original position closed at TP2
SCALED_TP3_FRAC = 0.25   # fraction of original position closed at TP3

"""
config.py — Global settings for the ARKAD MRK research pipeline.

Calibrated for BTCUSDT 1-hour bars.
Edit these values to tune the pipeline without touching any logic files.
"""

# ── Data ─────────────────────────────────────────────────────────────────────
DATE_COLUMN = "date"          # Name of the date/datetime column in the CSV
OHLCV_COLUMNS = {             # Expected column name mapping
    "open":   "open",
    "high":   "high",
    "low":    "low",
    "close":  "close",
    "volume": "volume",
}

# ── Features ─────────────────────────────────────────────────────────────────
# Windows expressed in 1H bars:
#   4 bars  =  4 hours
#  12 bars  = 12 hours
#  24 bars  =  1 day
#  48 bars  =  2 days
RETURN_WINDOWS = [4, 12, 24, 48]

# SMA windows at 1H:
#   20 bars  =  20H  (~1 day)
#   50 bars  =  50H  (~2 days)
#  200 bars  = 200H  (~8 days)
SMA_WINDOWS    = [20, 50, 200]

RSI_WINDOW = 14               # Standard 14-period RSI
ATR_WINDOW = 14               # 14 × 1H = 14 hours of true-range history
BB_WINDOW  = 20               # Bollinger Band lookback
BB_STD     = 2.0              # Bollinger Band standard deviation multiplier

# ── Market regime ────────────────────────────────────────────────────────────
# Trend direction at 1H scale:
#   FAST =  24 bars =  1 day
#   SLOW = 168 bars =  7 days (1 week)
REGIME_TREND_SMA_FAST  = 24
REGIME_TREND_SMA_SLOW  = 168

# Volatility regime: compare current ATR to its rolling median.
# 72 bars = 72H = 3 days of history for a stable baseline.
VOL_REGIME_WINDOW      = 72
HIGH_VOL_THRESHOLD     = 1.5    # atr / median_atr >= this  → high_vol
LOW_VOL_THRESHOLD      = 0.75   # atr / median_atr <= this  → low_vol

# Gate: set True to also block entries during high_vol bars
BLOCK_HIGH_VOL         = False

# Set True to block entries during low_vol bars (insufficient ATR to reach TP/SL).
# False = allow low_vol bars; edge survives at high model confidence (>=0.55) even
# in low-vol environments given the 24H hold window.
BLOCK_LOW_VOL          = True

# Per-volatility-regime BUY probability thresholds for regime-aware filtering.
# None means the regime is blocked entirely (no trades allowed).
VOL_REGIME_THRESHOLDS = {
    "low_vol":    0.40,   # allow at standard threshold (BLOCK_LOW_VOL=False)
    "normal_vol": 0.40,   # standard threshold
    "high_vol":   0.40,   # allow — wide ATR makes TP/SL levels meaningful
}

# ── Sessions (1H granularity) ─────────────────────────────────────────────────
# At 1H resolution each bar falls cleanly into one session bucket.
#
#   asia         :  00:00 – 07:59 UTC   (low volume, choppy)
#   london_open  :  08:00 – 10:59 UTC   (London open, strong momentum)
#   eu_mid       :  11:00 – 12:59 UTC   (quiet EU mid-session)
#   us_open      :  13:00 – 16:59 UTC   (US open, highest volume)
#   us_afternoon :  17:00 – 20:59 UTC   (continued US liquidity)
#   late         :  21:00 – 23:59 UTC   (thin market)
#
# All sessions open: the pipeline's regime/session analysis will surface
# which sessions contribute positively at sufficient frequency.
ALLOWED_SESSIONS = ["asia", "london_open", "eu_mid", "us_open", "us_afternoon", "late"]

# Trend regimes where trading is allowed.
# All three regimes open: filter relaxation sweep showed that "range" bars are
# slightly profitable at high model confidence (PF=1.33 vs 1.26 with range blocked).
# Range bars appear to be valid mean-reversion setups the 24H model correctly identifies.
ALLOWED_TRENDS = ["trend_up", "trend_down", "range"]

# ── Labelling ────────────────────────────────────────────────────────────────
# 24 bars × 1H = 24 hours ahead.
FORWARD_RETURN_WINDOW = 24

# ATR multiplier: 0.85 ATR over 24 bars labels clean directional moves.
# (median 24H forward move ~1.87 ATR; 0.85 captures the directional majority)
LABEL_ATR_MULT        = 0.85

# ── Training basis (shared between live paper trader and backtest) ──────────
# Single source of truth for what data the model is allowed to see at fit time.
#
# Convention:
#   train: bars where TRAINING_START_DATE <= bar.index < TRAINING_CUTOFF_DATE
#   test : bars where bar.index >= TRAINING_CUTOFF_DATE   (live, BT-OOS, parity)
#
# Both ends are fixed because data/download_all.py uses YEARS_BACK from the
# CURRENT date — a `--force` redownload silently shifts the CSV start forward,
# which used to invalidate any model trained from the older CSV.  Anchoring
# both START and CUTOFF makes the training set CSV-refresh-proof: if the CSV
# happens to start earlier than TRAINING_START_DATE the extra bars are dropped;
# if it starts later, a warning is logged so the operator can re-download.
#
# When bumping CUTOFF (recommended every 3-6 months), review START too —
# keeping a 3-year span (e.g., START=2023-01-01 ↔ CUTOFF=2026-01-01) preserves
# walk-forward comparability across model refreshes.
TRAINING_START_DATE  = "2022-06-01"
TRAINING_CUTOFF_DATE = "2026-01-01"

# ── Model ────────────────────────────────────────────────────────────────────
TEST_SIZE          = 0.2           # Fraction of data held out for evaluation
RANDOM_STATE       = 42
N_ESTIMATORS       = 200           # RandomForest trees

# Minimum samples required at each leaf node.
# At 1H the signal-to-noise ratio is higher than 5m; 20 is appropriate.
RF_MIN_SAMPLES_LEAF = 20

# Probability threshold for trade entry signals.
# With a 2-class model P(BUY) + P(SELL) = 1, so any T < 0.50 creates a large
# conflict zone [T, 1-T] that traps most bars as flat.  T = 0.55 eliminates
# all conflicts: BUY when P(BUY) >= 0.55, SELL when P(BUY) <= 0.45.
BUY_PROB_THRESHOLD  = 0.55          # Min predicted P(BUY)  to emit a long signal
SELL_PROB_THRESHOLD = 0.55          # Min predicted P(SELL) to emit a short signal

# ── Backtest ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL      = 10_000.0    # Starting equity ($)
POSITION_SIZE_PCT    = 0.10        # Fraction of equity per trade (10%)
COMMISSION_PCT       = 0.001       # Round-trip commission (0.1%)

# Taker slippage: 2 bps per side (same as 5m — Bybit perpetuals).
SLIPPAGE_PCT         = 0.0002      # 2 basis points per side

# 24 bars × 1H = 24 hours max hold — matches FORWARD_RETURN_WINDOW.
HOLD_BARS            = 24

# ATR-based exit levels scaled for the 24H forward window.
# Median 24H forward move ≈ 1.87 ATR; TP=2.5 is reachable for labeled (≥0.85 ATR) moves.
# R:R = 2.5 / 1.5 = 1.67  (break-even at 37.5% WR; at 50% WR → PF ≈ 1.67)
STOP_LOSS_ATR_MULT   = 1.5         # Stop loss:   1.5 × ATR
TAKE_PROFIT_ATR_MULT = 2.5         # Take profit: 2.5 × ATR

# ── Exit mode (single TP vs scaled exit) ─────────────────────────────────────
# "single" — classic single TP at TAKE_PROFIT_ATR_MULT × ATR (legacy)
# "scaled" — three partial exits + BE stop after TP1 (DEFAULT, mirrors paper_trading)
#
# Validated on 2023-2025 walk-forward (native engine, AVAX/ADA/SOL/XRP):
#                  single        scaled        delta
#   trades         5,136         6,076         +940 (BE-exits free position earlier)
#   WR             48.8%         74.1%         +25.3pp
#   Avg R          +0.212R       +0.217R       +0.005R
#   Sum R          +1090.62R     +1319.26R     +228.64R
#   PF             1.426         1.845         +0.419
#   Max DD         17.98%        5.12%         -12.86pp  ← 71% drawdown reduction
#   CAGR ($10K/10%) +40.3%        +59.0%        +18.75pp
BACKTEST_EXIT_MODE   = "scaled"

# Scaled exit parameters (used only when BACKTEST_EXIT_MODE == "scaled").
# Must match paper_trading/config_live.py for backtest ↔ live parity.
SCALED_TP1_R         = 0.65        # TP1 level in R-units (close TP1_FRAC)
SCALED_TP2_R         = 1.00        # TP2 level in R-units
SCALED_TP3_R         = 1.67        # TP3 level = TAKE_PROFIT_ATR_MULT/STOP_LOSS_ATR_MULT
SCALED_TP1_FRAC      = 0.50        # Fraction closed at TP1
SCALED_TP2_FRAC      = 0.25        # Fraction closed at TP2
SCALED_TP3_FRAC      = 0.25        # Fraction closed at TP3 (= 1 − TP1 − TP2)

# ── 4H Context Layer — SMA crossover (proven, WF PF = 0.988) ─────────────────
# Set USE_4H_CONTEXT=True in the main pipeline to gate 1H signals by a 4H
# directional bias (bull / bear / neutral).  See features/context_4h.py.
#
# SMA windows expressed in 4H bars:
#   fast = 20 × 4H = 80  1H bars ≈ 3.3 days
#   slow = 50 × 4H = 200 1H bars ≈ 8.3 days
USE_4H_CONTEXT      = False       # True = enable 4H bias gate in main pipeline
CONTEXT_4H_MODE     = "strict"    # "strict"  = only aligned direction (no neutral)
                                   # "relaxed" = aligned + neutral allowed
CONTEXT_4H_SMA_FAST = 20          # 4H SMA fast window (in 4H bars)
CONTEXT_4H_SMA_SLOW = 50          # 4H SMA slow window (in 4H bars)

# ── 4H Context Layer — ADX gate (Design A, experimental) ─────────────────────
# See features/context_4h_adx.py and experiments/context_4h_adx_test.py.
#
# bull   : close > SMA-slow  AND  ADX > ADX_THRESHOLD  (trending up)
# bear   : close < SMA-slow  AND  ADX > ADX_THRESHOLD  (trending down)
# neutral: ADX <= ADX_THRESHOLD  (confirmed range — not warmup/unknown)
#
# In "adx_range" filter mode, neutral bars are allowed for both directions
# (hypothesis: low-ADX range bars are valid mean-reversion setups).
CONTEXT_4H_ADX_WINDOW    = 14    # Wilder's ADX period in 4H bars (~2.3 days)
CONTEXT_4H_ADX_THRESHOLD = 20    # Classic trend/range boundary (Wilder's standard)

# ── Experiments ──────────────────────────────────────────────────────────────
REPORTS_DIR        = "reports"
EXPERIMENTS_DIR    = "experiments"

# ── Strategy selection ────────────────────────────────────────────────────────
# Minimum trades in the test window for a strategy to be eligible for selection.
# Corresponds to ~1 trade/day over the 20% test period (~290 days of 1H data).
# Strategies below this threshold are disqualified regardless of PF.
MIN_TRADES = 200

# Thresholds for confidence-filter sweep.
# 2-class model: thresholds below 0.50 create conflict zones; sweep from 0.50 up.
SWEEP_THRESHOLDS   = [0.50, 0.52, 0.55, 0.58, 0.60]

# Walk-forward validation windows (anchored / expanding-window variant).
WF_TRAIN_FRACTIONS = [0.60, 0.70, 0.80]
WF_TEST_FRACTION   = 0.10

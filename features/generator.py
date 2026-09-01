"""
features/generator.py — Generates candidate technical features from OHLCV data.

Design philosophy:
  - No feature is "the strategy". All features are candidates; the model decides
    which ones carry predictive signal.
  - Features are purely look-back (no future leakage).
  - Adding new features is as simple as appending to _add_*() helpers.

Feature groups
--------------
1. Return-based        : past n-bar log returns
2. Moving averages     : price relative to SMA; SMA crossovers
3. Momentum (RSI)      : normalised relative strength
4. Volatility (ATR)    : average true range normalised by price
5. Bollinger Bands     : price position within the band
6. Volume              : volume relative to rolling averages
7. Candle microstructure: body/wick ratios, range compression, vol expansion
8. Higher-timeframe    : 1H trend, momentum, and volatility context
9. Momentum extras     : MACD histogram + slope, RSI slope, ATR expansion (14-bar)
"""

import numpy as np
import pandas as pd
import config


# ── Public API ────────────────────────────────────────────────────────────────

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all candidate features to the OHLCV DataFrame in-place."""
    df = df.copy()
    df = _add_return_features(df)
    df = _add_sma_features(df)
    df = _add_rsi(df)
    df = _add_atr(df)              # must come before candle + momentum features
    df = _add_bollinger(df)
    df = _add_volume_features(df)
    df = _add_candle_features(df)  # requires atr_pct → after _add_atr
    df = _add_momentum_features(df)  # requires rsi + atr_pct → after _add_rsi/_add_atr
    # _add_htf_features is only meaningful when input bars are finer than 1H.
    # On 1H data the 1H resample is a no-op; the resulting features are
    # 1-bar-lagged duplicates of existing columns and add no information.
    # Uncomment the line below when running on sub-hourly (e.g. 5m) data.
    # df = _add_htf_features(df)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a 'label' column using ATR-normalised forward returns.

    The raw forward return is divided by the current bar's ATR (expressed as
    a fraction of price) to produce a unit-free score: how many ATRs did price
    move over the next FORWARD_RETURN_WINDOW bars?

      fwd_return_atr = (close[t+n] / close[t] - 1) / atr_pct[t]

    Labels:
      +1  →  fwd_return_atr >= +LABEL_ATR_MULT   (BUY)
      -1  →  fwd_return_atr <= -LABEL_ATR_MULT   (SELL)
       0  →  otherwise                            (NEUTRAL)

    At 1H, LABEL_ATR_MULT=0.5 requires price to move ±0.5 ATR over the next
    24 bars (1 day), which is a meaningful and learnable threshold.

    The last FORWARD_RETURN_WINDOW rows will have NaN labels and should be
    dropped before training.
    """
    if "atr_pct" not in df.columns:
        raise ValueError(
            "add_labels() requires 'atr_pct' column. "
            "Call generate_features() before add_labels()."
        )

    df = df.copy()
    fwd  = config.FORWARD_RETURN_WINDOW
    mult = config.LABEL_ATR_MULT

    fwd_return     = df["close"].shift(-fwd) / df["close"] - 1
    fwd_return_atr = fwd_return / df["atr_pct"]

    df["fwd_return"]     = fwd_return
    df["fwd_return_atr"] = fwd_return_atr

    df["label"] = 0
    df.loc[fwd_return_atr >=  mult, "label"] =  1
    df.loc[fwd_return_atr <= -mult, "label"] = -1

    return df


def add_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (2, 4, 8, 12),
    base_mult: float | None = None,
    base_horizon: int | None = None,
) -> pd.DataFrame:
    """
    Add per-horizon labels for multi-horizon shadow-mode research.

    For each horizon h in `horizons`, adds columns:
        label_h{h}       — {-1, 0, +1} using scaled ATR multiplier
        fwd_return_h{h}  — raw forward return over h bars

    Scaled multiplier (variant B, sqrt-of-time):
        mult(h) = base_mult * sqrt(h / base_horizon)

    Rationale: price movement ~ sqrt(t) under Brownian-motion assumptions,
    so 0.85 ATR over 24 bars corresponds to ~0.24 ATR over 2 bars.  Keeps
    label semantics comparable across horizons — "meaningful directional
    move for THIS horizon" rather than "same absolute move regardless of h".

    Same semantics as `add_labels`, applied per-horizon.  Called by
    signal_engine._train_from_df only when multi-horizon shadow-mode is
    enabled; primary 24H model still uses `label` from `add_labels`.
    """
    import math

    if base_mult is None:
        base_mult = config.LABEL_ATR_MULT
    if base_horizon is None:
        base_horizon = config.FORWARD_RETURN_WINDOW

    if "atr_pct" not in df.columns:
        raise ValueError(
            "add_multi_horizon_labels() requires 'atr_pct' column. "
            "Call generate_features() first."
        )

    df = df.copy()
    for h in horizons:
        mult = base_mult * math.sqrt(h / base_horizon)
        fwd  = df["close"].shift(-h) / df["close"] - 1
        fwd_atr = fwd / df["atr_pct"]
        df[f"fwd_return_h{h}"] = fwd
        df[f"label_h{h}"] = 0
        df.loc[fwd_atr >=  mult, f"label_h{h}"] =  1
        df.loc[fwd_atr <= -mult, f"label_h{h}"] = -1
    return df


# ── Private helpers ───────────────────────────────────────────────────────────

def _add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in config.RETURN_WINDOWS:
        df[f"ret_{w}"] = np.log(df["close"] / df["close"].shift(w))
    return df


def _add_sma_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in config.SMA_WINDOWS:
        sma = df["close"].rolling(w).mean()
        df[f"sma_{w}_ratio"] = df["close"] / sma - 1   # distance from SMA

    # Crossover signal: short SMA vs long SMA (first two windows)
    if len(config.SMA_WINDOWS) >= 2:
        short, long_ = config.SMA_WINDOWS[0], config.SMA_WINDOWS[1]
        df[f"sma_cross_{short}_{long_}"] = (
            df["close"].rolling(short).mean() /
            df["close"].rolling(long_).mean() - 1
        )
    return df


def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    w = config.RSI_WINDOW
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = (-delta.clip(upper=0)).rolling(w).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_norm"] = df["rsi"] / 100 - 0.5   # centre around 0
    return df


def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    w = config.ATR_WINDOW
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(w).mean()
    df["atr_pct"] = atr / df["close"]   # ATR as % of price
    return df


def _add_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    w   = config.BB_WINDOW
    std = config.BB_STD
    mid  = df["close"].rolling(w).mean()
    band = df["close"].rolling(w).std() * std
    df["bb_pos"]   = (df["close"] - mid) / band.replace(0, np.nan)  # –1 to +1 roughly
    df["bb_width"] = (2 * band) / mid                                # band width as % of price
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    vol_ma = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)
    df["vol_log"]   = np.log1p(df["volume"])
    return df


def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle microstructure features — informative on 5m bars where individual
    candle shape carries information about short-term order flow.

    Features added
    --------------
    body_ratio      : |close - open| / (high - low)
                      High  → strong directional bar (few wicks)
                      Low   → doji / indecision

    range_ratio     : (high - low) / close
                      Normalised single-bar range (volatility proxy)

    upper_wick      : (high - max(open, close)) / (high - low)
                      High → selling pressure at the top of the bar

    lower_wick      : (min(open, close) - low) / (high - low)
                      High → buying pressure at the bottom of the bar

    vol_expansion   : atr_pct / rolling_mean(atr_pct, 5) - 1
                      Positive → volatility expanding (breakout candidate)
                      Negative → volatility compressing (mean reversion setup)

    vol_spike_5     : volume / rolling_mean(volume, 5)
                      Spike in volume relative to the last 5 bars

    range_vs_mean   : (high - low) / rolling_mean(high - low, 20)
                      How the current bar's range compares to the recent average
                      Used to detect range compression / expansion regimes
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    body = (df["close"] - df["open"]).abs()
    df["body_ratio"]  = body / candle_range
    df["range_ratio"] = (df["high"] - df["low"]) / df["close"]

    upper_wick_raw = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick_raw = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick"] = upper_wick_raw / candle_range
    df["lower_wick"] = lower_wick_raw / candle_range

    # Short-term volatility expansion: how fast has ATR grown in the last 5 bars?
    atr_ma5 = df["atr_pct"].rolling(5).mean()
    df["vol_expansion"] = df["atr_pct"] / atr_ma5.replace(0, np.nan) - 1

    # Short-term volume spike over the last 5 bars
    vol_ma5 = df["volume"].rolling(5).mean()
    df["vol_spike_5"] = df["volume"] / vol_ma5.replace(0, np.nan)

    # Rolling range compression / expansion vs 20-bar mean range
    bar_range    = df["high"] - df["low"]
    mean_range20 = bar_range.rolling(20).mean()
    df["range_vs_mean"] = bar_range / mean_range20.replace(0, np.nan)

    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum-based features: MACD histogram, RSI slope, ATR expansion (14-bar).

    Requires _add_rsi() and _add_atr() to have been called first.

    Features added
    --------------
    macd_hist        : (EMA12 − EMA26 − EMA9_of_MACD) / close
                       Histogram of the MACD oscillator, price-normalised.
                       Positive → short-term momentum above its 9-bar average.
                       Negative → weakening or reversing momentum.

    macd_hist_slope  : 3-bar difference of macd_hist
                       Captures acceleration / deceleration of MACD momentum.
                       Rising → momentum building; falling → momentum fading.

    rsi_slope        : (RSI_t − RSI_{t-3}) / 100
                       Rate of change of RSI over the last 3 bars.
                       Adds directionality beyond the level signal in rsi_norm.

    atr_expansion_14 : atr_pct / rolling_mean(atr_pct, 14) − 1
                       Compares current volatility to its 14-bar mean.
                       Complements vol_expansion (5-bar) with a medium-term view.
                       Positive → volatility expanding vs recent baseline.
    """
    close = df["close"]

    # ── MACD histogram ────────────────────────────────────────────────────────
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - signal_line
    df["macd_hist"]       = macd_hist / close                  # price-normalised
    df["macd_hist_slope"] = df["macd_hist"].diff(3)            # 3-bar acceleration

    # ── RSI slope ─────────────────────────────────────────────────────────────
    df["rsi_slope"] = df["rsi"].diff(3) / 100                  # normalised to [−1, +1]

    # ── ATR expansion (14-bar) ────────────────────────────────────────────────
    atr_ma14 = df["atr_pct"].rolling(14).mean()
    df["atr_expansion_14"] = df["atr_pct"] / atr_ma14.replace(0, np.nan) - 1

    return df


def _add_htf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Higher-timeframe (1H) context features, computed by resampling the 5m data.

    Rationale
    ---------
    On 5m bars, the local price action is noisy. Including 1H-scale trend and
    volatility context allows the model to condition on the broader market
    environment without requiring a separate 1H CSV file.

    Look-ahead prevention
    ---------------------
    After computing each 1H series, we shift it forward by one 1H period
    (shift(1)) before forward-filling to 5m resolution.  This ensures that
    the value visible to a 5m bar at time t uses only 1H candles whose CLOSE
    occurred strictly before t.

    Example:
      1H bar labeled 13:00 covers 5m bars 13:00 – 13:55 (last close at 13:55).
      After shift(1), the 1H value labeled 14:00 holds the data from 13:00.
      Forward-filling populates 5m bars 14:00 – 14:55 with that value.
      The 5m bar at 14:00 therefore sees only data available at 13:55 — safe.

    Features added
    --------------
    htf_sma20_ratio : 1H close / 1H SMA-20 − 1   (short-term 1H trend)
    htf_sma_cross   : 1H SMA-20 / 1H SMA-50 − 1  (1H trend direction score)
    htf_rsi_norm    : 14-period RSI on 1H, centred around 0
    htf_vol_ratio   : 1H ATR / median(1H ATR, 50) (1H volatility regime score)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("_add_htf_features() requires a DatetimeIndex.")

    # ── Resample 5m → 1H ─────────────────────────────────────────────────────
    close_1h  = df["close"].resample("1h", closed="left", label="left").last()
    high_1h   = df["high"].resample("1h",  closed="left", label="left").max()
    low_1h    = df["low"].resample("1h",   closed="left", label="left").min()

    # ── 1H ATR (14-period) ────────────────────────────────────────────────────
    hl_1h  = high_1h  - low_1h
    hc_1h  = (high_1h - close_1h.shift()).abs()
    lc_1h  = (low_1h  - close_1h.shift()).abs()
    tr_1h  = pd.concat([hl_1h, hc_1h, lc_1h], axis=1).max(axis=1)
    atr_1h = tr_1h.rolling(14).mean()
    atr_pct_1h = atr_1h / close_1h.replace(0, np.nan)

    # ── 1H SMAs and trend score ───────────────────────────────────────────────
    sma20_1h = close_1h.rolling(20).mean()
    sma50_1h = close_1h.rolling(50).mean()

    htf_sma20_ratio = close_1h / sma20_1h.replace(0, np.nan) - 1
    htf_sma_cross   = sma20_1h / sma50_1h.replace(0, np.nan) - 1

    # ── 1H RSI (14-period) ────────────────────────────────────────────────────
    delta_1h = close_1h.diff()
    gain_1h  = delta_1h.clip(lower=0).rolling(14).mean()
    loss_1h  = (-delta_1h.clip(upper=0)).rolling(14).mean()
    rs_1h    = gain_1h / loss_1h.replace(0, np.nan)
    rsi_1h   = 100 - (100 / (1 + rs_1h))
    htf_rsi_norm = rsi_1h / 100 - 0.5   # centred around 0

    # ── 1H volatility regime score ────────────────────────────────────────────
    atr_median_1h  = atr_pct_1h.rolling(50).median()
    htf_vol_ratio  = atr_pct_1h / atr_median_1h.replace(0, np.nan)

    # ── Shift by 1H and forward-fill to 5m ───────────────────────────────────
    def _to_5m(series_1h: pd.Series, fill_value: float) -> pd.Series:
        """Shift 1H series by 1 period, reindex to 5m, forward-fill, fill NaN."""
        return (
            series_1h.shift(1)
            .reindex(df.index, method="ffill")
            .fillna(fill_value)
        )

    df = df.copy()
    df["htf_sma20_ratio"] = _to_5m(htf_sma20_ratio, fill_value=0.0)
    df["htf_sma_cross"]   = _to_5m(htf_sma_cross,   fill_value=0.0)
    df["htf_rsi_norm"]    = _to_5m(htf_rsi_norm,     fill_value=0.0)
    df["htf_vol_ratio"]   = _to_5m(htf_vol_ratio,    fill_value=1.0)

    return df

"""
features/market_regime.py — Market regime detection and analysis layer.

This module classifies every bar into a trend regime and a volatility regime.
The regime columns are used downstream for analysis (experiments/regime_analysis.py)
rather than as a hard trade gate.

Design principles
-----------------
- All computations use only past data (rolling windows, no look-ahead).
- Regime columns are added alongside features but excluded from ML inputs
  via _NON_FEATURE_COLS in models/classifier.py.
- The model never sees regime labels as input features — it implicitly learns
  regime-like structure through the SMA ratio and ATR features it already has.
- This module has no imports from backtest/ or models/ — it is a pure data
  transformation layer.

Columns added to df by add_regime_columns()
-------------------------------------------
  trend        : "trend_up" | "trend_down" | "range"
  vol_regime   : "high_vol" | "normal_vol" | "low_vol"
  regime_gate  : 1 (historically favourable) | 0 (historically unfavourable)
                 Kept for reference and future use; not applied as a hard block.

Trend detection
---------------
  sma_fast = close.rolling(REGIME_TREND_SMA_FAST).mean()
  sma_slow = close.rolling(REGIME_TREND_SMA_SLOW).mean()

  trend_up   : close > sma_slow  AND  sma_fast > sma_slow
  trend_down : close < sma_slow  AND  sma_fast < sma_slow
  range      : neither (or SMA NaN during warmup)

Volatility regime
-----------------
  vol_ratio = atr_pct / rolling_median(atr_pct, VOL_REGIME_WINDOW)

  high_vol   : vol_ratio >= HIGH_VOL_THRESHOLD   (default 1.5)
  low_vol    : vol_ratio <= LOW_VOL_THRESHOLD     (default 0.75)
  normal_vol : otherwise (including NaN vol_ratio → safe fallback)
"""

import numpy as np
import pandas as pd

import config


# ── Public API ────────────────────────────────────────────────────────────────

def add_session_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a 'session' column based on the UTC hour of each bar's timestamp.

    Six buckets calibrated for 5m BTCUSDT trading.  Three are allowed
    (see config.ALLOWED_SESSIONS); three are blocked by apply_session_filter().

      asia         : 00:00 – 07:59 UTC   low volume, choppy
      london_open  : 08:00 – 10:59 UTC   London open spike
      eu_mid       : 11:00 – 12:59 UTC   quiet EU mid-session
      us_open      : 13:00 – 16:59 UTC   US open, highest volume
      us_afternoon : 17:00 – 20:59 UTC   continued US session
      late         : 21:00 – 23:59 UTC   thin market, wide spreads

    Assumes df.index is a UTC DatetimeIndex (naive or tz-aware UTC).
    Must be called after load_ohlcv(); has no dependency on other feature columns.

    Parameters
    ----------
    df : DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame — original df plus 'session' column.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("add_session_column() requires a DatetimeIndex.")

    hour = df.index.hour

    # Default: asia (00:00–07:59, blocked)
    session = pd.Series("asia", index=df.index, dtype=object)

    session[(hour >= 8)  & (hour <= 10)] = "london_open"   # 08:00–10:59
    session[(hour >= 11) & (hour <= 12)] = "eu_mid"         # 11:00–12:59
    session[(hour >= 13) & (hour <= 16)] = "us_open"        # 13:00–16:59
    session[(hour >= 17) & (hour <= 20)] = "us_afternoon"   # 17:00–20:59
    session[hour >= 21]                  = "late"            # 21:00–23:59

    df = df.copy()
    df["session"] = session
    return df


def apply_trend_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a trend filter to trade signals.

    Zeroes out signals on bars whose trend is not in config.ALLOWED_TRENDS.
    Preserves the original 'signal' column unchanged.
    Adds 'signal_trend_filtered' column.

    Works with signal values of -1 (short), 0 (flat), and 1 (long):
    multiplying by a boolean mask converts out-of-regime signals to 0
    while leaving -1 (short) and 1 (long) intact inside the allowed regime.

    Parameters
    ----------
    signals_df : DataFrame from generate_signals(), must contain
                 'signal' and 'trend' columns.

    Returns
    -------
    pd.DataFrame — copy with 'signal_trend_filtered' added.
    """
    for col in ("signal", "trend"):
        if col not in signals_df.columns:
            raise ValueError(
                f"apply_trend_filter() requires '{col}' column. "
                f"Call generate_signals() and add_regime_columns() first."
            )

    result  = signals_df.copy()
    allowed = result["trend"].isin(config.ALLOWED_TRENDS)
    result["signal_trend_filtered"] = result["signal"] * allowed.astype(int)
    return result


def apply_vol_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a volatility filter to trade signals.

    Zeroes out signals on bars where vol_regime == "low_vol".
    Trades are only allowed when vol_regime is "normal_vol" or "high_vol".

    The original 'signal' column is preserved intact.
    A new 'signal_vol_filtered' column is added.

    Parameters
    ----------
    signals_df : DataFrame from generate_signals(), must contain
                 'signal' and 'vol_regime' columns.

    Returns
    -------
    pd.DataFrame — copy with 'signal_vol_filtered' added.
    """
    for col in ("signal", "vol_regime"):
        if col not in signals_df.columns:
            raise ValueError(
                f"apply_vol_filter() requires '{col}' column. "
                f"Call generate_signals() and add_regime_columns() first."
            )

    result = signals_df.copy()
    if config.BLOCK_LOW_VOL:
        allowed = result["vol_regime"].isin(["normal_vol", "high_vol"])
    else:
        # low_vol bars allowed; only regime_threshold_filter applies per-regime gates
        allowed = pd.Series(True, index=result.index)
    result["signal_vol_filtered"] = result["signal"] * allowed.astype(int)
    return result


def apply_regime_threshold_filter(
    signals_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Apply per-volatility-regime BUY probability thresholds to trade signals.

    For each vol_regime bucket, the threshold from `thresholds` is applied to
    `buy_prob`. A value of None for a regime blocks all trades in that bucket.
    Defaults to config.VOL_REGIME_THRESHOLDS.

    The original 'signal' and 'signal_vol_filtered' columns are not modified.
    A new 'signal_regime_filtered' column is added.

    Parameters
    ----------
    signals_df : DataFrame from generate_signals() with 'buy_prob' and
                 'vol_regime' columns present.
    thresholds : dict mapping vol_regime label to float threshold or None.
                 Defaults to config.VOL_REGIME_THRESHOLDS.

    Returns
    -------
    pd.DataFrame — copy with 'signal_regime_filtered' added.
    """
    for col in ("buy_prob", "vol_regime"):
        if col not in signals_df.columns:
            raise ValueError(
                f"apply_regime_threshold_filter() requires '{col}' column. "
                f"Call generate_signals() and add_regime_columns() first."
            )

    if thresholds is None:
        thresholds = config.VOL_REGIME_THRESHOLDS

    result = signals_df.copy()
    # Start from the existing signal (-1 / 0 / 1) and selectively block bars.
    # This preserves both long and short signals through the filter.
    signal = result["signal"].copy().astype(int)

    for regime, threshold in thresholds.items():
        mask = result["vol_regime"] == regime
        if threshold is None:
            # Block all signals in this regime (e.g. low_vol)
            signal[mask] = 0
        else:
            # Block long signals that don't meet the buy conviction bar
            buy_mask  = mask & (signal == 1)
            if "buy_prob" in result.columns:
                signal[buy_mask & (result["buy_prob"] < threshold)] = 0
            # Block short signals that don't meet the sell conviction bar
            sell_mask = mask & (signal == -1)
            if "sell_prob" in result.columns:
                signal[sell_mask & (result["sell_prob"] < threshold)] = 0

    result["signal_regime_filtered"] = signal
    return result


def apply_session_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a session filter to trade signals.

    Zeroes out signals on bars whose session is not in config.ALLOWED_SESSIONS.
    Trades are only allowed during permitted sessions (default: 'us' and 'late').

    The original 'signal', 'signal_vol_filtered', and 'signal_regime_filtered'
    columns are preserved intact.
    A new 'signal_session_filtered' column is added.

    Parameters
    ----------
    signals_df : DataFrame with 'signal' and 'session' columns.

    Returns
    -------
    pd.DataFrame — copy with 'signal_session_filtered' added.
    """
    for col in ("signal", "session"):
        if col not in signals_df.columns:
            raise ValueError(
                f"apply_session_filter() requires '{col}' column. "
                f"Call generate_signals() and add_session_column() first."
            )

    result = signals_df.copy()
    allowed = result["session"].isin(config.ALLOWED_SESSIONS)
    result["signal_session_filtered"] = result["signal"] * allowed.astype(int)
    return result


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append trend, vol_regime, and regime_gate columns to df.

    Must be called after generate_features() (requires atr_pct).
    Must be called before add_labels() and model training.

    Parameters
    ----------
    df : DataFrame with at least 'close', 'high', 'low', 'atr_pct' columns.

    Returns
    -------
    pd.DataFrame — original df plus three new columns.
    """
    if "atr_pct" not in df.columns:
        raise ValueError(
            "add_regime_columns() requires 'atr_pct'. "
            "Call generate_features() first."
        )

    df = df.copy()
    df = _add_trend(df)
    df = _add_vol_regime(df)
    df = _add_gate(df)
    return df


def print_regime_stats(df: pd.DataFrame) -> None:
    """
    Print bar counts and percentages for each trend and volatility regime.

    Parameters
    ----------
    df : Full DataFrame with 'trend' and 'vol_regime' columns.
    """
    n = len(df)
    sep = "-" * 52

    print(f"\n  Regime Distribution  ({n:,} bars total)")
    print(f"  {sep}")

    print(f"  Trend regime:")
    for label in ["trend_up", "range", "trend_down"]:
        count = (df["trend"] == label).sum()
        pct   = count / n * 100
        print(f"    {label:<14} {count:>6} bars  ({pct:>5.1f}%)")

    print(f"\n  Volatility regime:")
    for label in ["low_vol", "normal_vol", "high_vol"]:
        count = (df["vol_regime"] == label).sum()
        pct   = count / n * 100
        print(f"    {label:<14} {count:>6} bars  ({pct:>5.1f}%)")

    gate_on = (df["regime_gate"] == 1).sum()
    print(f"\n  regime_gate = 1 (historically favourable):  "
          f"{gate_on:,}  ({gate_on/n*100:.1f}%)")
    print(f"  {sep}")


# ── Private helpers ───────────────────────────────────────────────────────────

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    fast_w = config.REGIME_TREND_SMA_FAST
    slow_w = config.REGIME_TREND_SMA_SLOW

    sma_fast = df["close"].rolling(fast_w).mean()
    sma_slow = df["close"].rolling(slow_w).mean()

    # Default: range (covers NaN warmup bars automatically)
    trend = pd.Series("range", index=df.index, dtype=object)

    valid = sma_fast.notna() & sma_slow.notna()
    trend_up_mask   = valid & (df["close"] > sma_slow) & (sma_fast > sma_slow)
    trend_down_mask = valid & (df["close"] < sma_slow) & (sma_fast < sma_slow)

    trend[trend_up_mask]   = "trend_up"
    trend[trend_down_mask] = "trend_down"

    df["trend"] = trend
    return df


def _add_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    w = config.VOL_REGIME_WINDOW

    # Rolling median gives a stable "normal" baseline, resistant to spikes
    atr_median = df["atr_pct"].rolling(w).median()
    vol_ratio  = df["atr_pct"] / atr_median.replace(0, np.nan)

    # Default: normal_vol (covers NaN during warmup)
    vol_regime = pd.Series("normal_vol", index=df.index, dtype=object)

    vol_regime[vol_ratio >= config.HIGH_VOL_THRESHOLD] = "high_vol"
    vol_regime[vol_ratio <= config.LOW_VOL_THRESHOLD]  = "low_vol"
    # NaN vol_ratio stays "normal_vol" — already the default

    df["vol_regime"] = vol_regime
    return df


def _add_gate(df: pd.DataFrame) -> pd.DataFrame:
    gate = df["trend"] == "trend_up"

    if config.BLOCK_HIGH_VOL:
        gate = gate & (df["vol_regime"] != "high_vol")

    df["regime_gate"] = gate.astype(int)
    return df

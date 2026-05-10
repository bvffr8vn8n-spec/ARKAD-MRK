"""
features/context_4h_adx.py — 4H ADX-gated directional context layer (Design A).

Design A replaces the SMA-fast crossover confirmation with ADX trend strength.
The key difference from the proven SMA crossover (context_4h.py):

  Crossover (proven):
    bull = close > SMA-slow  AND  SMA-fast > SMA-slow
    → neutral when price is trending but fast hasn't crossed slow yet (3–5 day lag)

  Design A (ADX gate):
    bull    = close > SMA-slow  AND  ADX > ADX_THRESHOLD
    bear    = close < SMA-slow  AND  ADX > ADX_THRESHOLD
    neutral = ADX <= ADX_THRESHOLD  (confirmed range, not unknown/warmup)

Neutral semantics
-----------------
In the crossover, "neutral" is an ambiguous catch-all (transition, range, warmup).
In Design A, "neutral" specifically means the market is ranging (ADX < threshold).
This distinction motivates the "adx_range" filter mode: range bars may be valid
mean-reversion setups (prior filter_relax_sweep showed range PF = 1.33), so
allowing both directions in confirmed-range periods could recover frequency
without degrading quality.

Filter modes
------------
  strict    : long when bull, short when bear.  Neutral = blocked.
              Directly comparable to the proven crossover-strict variant.
  adx_range : long when bull OR neutral, short when bear OR neutral.
              Neutral bars are confirmed low-ADX range periods — allowed for both.

Column added
------------
  bias_4h_adx : 'bull' | 'bear' | 'neutral'
                Excluded from model training via _NON_FEATURE_COLS in
                models/classifier.py.

No look-ahead
-------------
Same shift(1) + ffill pattern as context_4h.py.  All 4H bars are shifted one
4H bar forward before being broadcast to 1H resolution.

ADX computation
---------------
Uses Wilder's original smoothing (seed = sum of first n values; subsequent bars
use the n-1/n recursive formula).  Computed on resampled 4H OHLCV bars.
Requires 'high' and 'low' columns in addition to 'close'.
"""

import numpy as np
import pandas as pd

import config


# ── Public API ────────────────────────────────────────────────────────────────

def add_4h_context_adx(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ADX-gated 4H directional bias and append 'bias_4h_adx' column.

    Must be called after generate_features() (requires DatetimeIndex and
    'high', 'low', 'close' columns).

    Parameters
    ----------
    df_1h : DataFrame with DatetimeIndex and 'high', 'low', 'close' columns.

    Returns
    -------
    pd.DataFrame — copy with 'bias_4h_adx' column appended.
    """
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        raise ValueError("add_4h_context_adx() requires a DatetimeIndex.")
    for col in ("high", "low", "close"):
        if col not in df_1h.columns:
            raise ValueError(f"add_4h_context_adx() requires a '{col}' column.")

    slow_w   = config.CONTEXT_4H_SMA_SLOW
    adx_w    = config.CONTEXT_4H_ADX_WINDOW
    adx_thr  = config.CONTEXT_4H_ADX_THRESHOLD

    # ── Resample 1H → 4H ─────────────────────────────────────────────────────
    close_4h = (
        df_1h["close"]
        .resample("4h", closed="left", label="left")
        .last()
        .dropna()
    )
    high_4h = (
        df_1h["high"]
        .resample("4h", closed="left", label="left")
        .max()
        .reindex(close_4h.index)
    )
    low_4h = (
        df_1h["low"]
        .resample("4h", closed="left", label="left")
        .min()
        .reindex(close_4h.index)
    )

    # ── SMA-slow (direction anchor) ───────────────────────────────────────────
    sma_slow = close_4h.rolling(slow_w).mean()

    # ── ADX on 4H bars ────────────────────────────────────────────────────────
    adx = _compute_adx(high_4h, low_4h, close_4h, adx_w)

    # ── Bias classification ───────────────────────────────────────────────────
    valid = sma_slow.notna() & adx.notna()

    bias_4h = pd.Series("neutral", index=close_4h.index, dtype=object)
    bias_4h[valid & (close_4h > sma_slow) & (adx > adx_thr)] = "bull"
    bias_4h[valid & (close_4h < sma_slow) & (adx > adx_thr)] = "bear"
    # neutral = ADX <= threshold (confirmed range) OR warmup (ADX NaN)

    # ── Shift 1 bar + forward-fill to 1H resolution ───────────────────────────
    bias_1h = (
        bias_4h.shift(1)
        .reindex(df_1h.index, method="ffill")
        .fillna("neutral")
    )

    result = df_1h.copy()
    result["bias_4h_adx"] = bias_1h
    return result


def apply_4h_context_adx_filter(
    signals_df: pd.DataFrame,
    mode: str = "strict",
) -> pd.DataFrame:
    """
    Gate trade signals by the ADX-based 4H directional bias.

    Reads the 'signal' column and writes 'signal_4h_adx_filtered'.
    The original 'signal' column is preserved unchanged.

    Parameters
    ----------
    signals_df : DataFrame with 'signal' and 'bias_4h_adx' columns.
    mode       : 'strict'    — long on bull, short on bear; neutral blocked.
                 'adx_range' — long on bull or neutral, short on bear or neutral.
                               Neutral = confirmed range → both directions allowed.

    Returns
    -------
    pd.DataFrame — copy with 'signal_4h_adx_filtered' column added.
    """
    for col in ("signal", "bias_4h_adx"):
        if col not in signals_df.columns:
            raise ValueError(
                f"apply_4h_context_adx_filter() requires '{col}' column. "
                f"Call add_4h_context_adx() before generating signals."
            )

    if mode not in ("strict", "adx_range"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'strict' or 'adx_range'.")

    result = signals_df.copy()
    sig    = result["signal"].copy().astype(int)
    bias   = result["bias_4h_adx"]

    if mode == "strict":
        long_ok  = bias == "bull"
        short_ok = bias == "bear"
    else:  # adx_range: neutral = confirmed range → allow both directions
        long_ok  = bias.isin(["bull",  "neutral"])
        short_ok = bias.isin(["bear",  "neutral"])

    sig[(sig ==  1) & ~long_ok]  = 0
    sig[(sig == -1) & ~short_ok] = 0

    result["signal_4h_adx_filtered"] = sig
    return result


def print_4h_adx_bias_stats(df: pd.DataFrame) -> None:
    """Print distribution of ADX-based 4H bias labels."""
    if "bias_4h_adx" not in df.columns:
        return

    n   = len(df)
    sep = "-" * 52

    print(f"\n  4H ADX Context Bias Distribution  ({n:,} bars)")
    print(f"  {sep}")
    for label in ["bull", "neutral", "bear"]:
        count = (df["bias_4h_adx"] == label).sum()
        pct   = count / n * 100
        print(f"    {label:<10} {count:>6,} bars  ({pct:>5.1f}%)")
    print(f"  {sep}")


# ── Private helpers ───────────────────────────────────────────────────────────

def _wilder_smooth(s: pd.Series, n: int) -> pd.Series:
    """
    Wilder's smoothing: seed = sum of first n values; subsequent bars use
    the (n-1)/n recursive formula.  This is the standard initialization used
    by Wilder for ATR, +DI, -DI, and ADX.
    """
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)

    # Find first non-NaN position
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1

    seed_end = start + n
    if seed_end > len(arr):
        return pd.Series(out, index=s.index)

    seed_window = arr[start:seed_end]
    if np.any(np.isnan(seed_window)):
        return pd.Series(out, index=s.index)

    out[seed_end - 1] = seed_window.sum()
    factor = (n - 1) / n
    for i in range(seed_end, len(arr)):
        v = arr[i]
        out[i] = out[i - 1] * factor + (v if not np.isnan(v) else 0.0)

    return pd.Series(out, index=s.index)


def _compute_adx(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    n: int,
) -> pd.Series:
    """
    Compute Wilder's ADX(n) from 4H OHLCV series.

    Returns a pd.Series of ADX values aligned to the input index.
    NaN for the warmup period (approximately 2n bars from the first valid bar).
    """
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        (high  - low).abs(),
        (high  - prev_close).abs(),
        (low   - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   =  high.diff()
    down_move = -low.diff()

    dm_plus  = up_move.copy()
    dm_minus = down_move.copy()

    # +DM: up_move positive and strictly greater than down_move; else 0
    dm_plus[ (dm_plus  < 0) | (dm_plus  <= dm_minus)] = 0.0
    # -DM: down_move positive and strictly greater than up_move; else 0
    dm_minus[(dm_minus < 0) | (dm_minus <= dm_plus )] = 0.0

    # Wilder-smoothed TR, +DM, -DM
    atr_s  = _wilder_smooth(tr,       n)
    sdm_p  = _wilder_smooth(dm_plus,  n)
    sdm_m  = _wilder_smooth(dm_minus, n)

    # +DI / -DI
    safe_atr = atr_s.replace(0.0, np.nan)
    di_plus  = 100.0 * sdm_p / safe_atr
    di_minus = 100.0 * sdm_m / safe_atr

    # DX
    di_sum = (di_plus + di_minus).replace(0.0, np.nan)
    dx     = 100.0 * (di_plus - di_minus).abs() / di_sum

    # ADX = Wilder smoothing of DX, normalized to 0-100.
    # _wilder_smooth seeds with sum(first n values), so its output ≈ n × true_ADX.
    # Dividing by n recovers the standard 0-100 ADX scale.
    adx = _wilder_smooth(dx, n) / n
    return adx

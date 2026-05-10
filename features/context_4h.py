"""
features/context_4h.py — 4H directional context layer.

Derives a macro directional bias from 1H OHLCV data by resampling to 4H bars
and computing a dual-SMA crossover on that timeframe.  The bias gates 1H trade
signals: in strict mode, longs are only allowed during a 4H bull phase and
shorts only during a 4H bear phase.

Motivation
----------
Walk-forward testing of the 1H/24H model consistently yields PF 0.58–0.68
across all time windows — below breakeven.  A suspected cause: the model fires
directional setups that are misaligned with the prevailing macro trend (e.g., a
bullish 1H signal during a 4H downtrend).  A slow-moving 4H context gate may
suppress these counter-trend entries without materially reducing
aligned-direction frequency, because the 4H trend changes at most 1–3× per day.

4H SMA configuration
--------------------
  SMA-fast : CONTEXT_4H_SMA_FAST × 4H bars  =  20 × 4H =  80 1H bars ≈ 3.3 days
  SMA-slow : CONTEXT_4H_SMA_SLOW × 4H bars  =  50 × 4H = 200 1H bars ≈ 8.3 days

  bull    : close > SMA-slow  AND  SMA-fast > SMA-slow
  bear    : close < SMA-slow  AND  SMA-fast < SMA-slow
  neutral : neither condition met (transition, range, or warmup)

No look-ahead
-------------
After computing each 4H series, it is shifted forward by 1 bar (shift(1)) then
forward-filled to 1H resolution.  This ensures that the value visible to a 1H
bar at time T uses only 4H candles whose CLOSE occurred strictly before T.

  Example:
    4H bar labeled 12:00 covers 1H bars 12:00–14:00 (last close at 14:00).
    After shift(1) the 4H value at label 16:00 holds the 12:00 bar data.
    Forward-fill gives 1H bars 16:00–19:00 that value.
    The 1H bar at 16:00 therefore sees only 4H data available at 14:00 — safe.

Filter modes
------------
  strict  : long when bias_4h == 'bull' only;  short when bias_4h == 'bear' only.
            All 'neutral' bars are blocked from both directions.
  relaxed : long when bias_4h in ('bull', 'neutral');
            short when bias_4h in ('bear', 'neutral').
            Only direct contradictions are blocked (e.g., no long during bear).

Columns added
-------------
  bias_4h : 'bull' | 'bear' | 'neutral'
            Excluded from model feature inputs via _NON_FEATURE_COLS in
            models/classifier.py — it is a filtering column, not a learned input.
"""

import pandas as pd

import config


# ── Public API ────────────────────────────────────────────────────────────────

def add_4h_context(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4H directional bias from 1H OHLCV data and append 'bias_4h' column.

    Must be called after generate_features() (requires DatetimeIndex and 'close').
    Must be called before add_labels() so the column is present throughout the
    full DataFrame used for training and backtesting.

    bias_4h is a categorical string column excluded from model training via
    _NON_FEATURE_COLS in models/classifier.py.

    Parameters
    ----------
    df_1h : DataFrame with DatetimeIndex and 'close' column.

    Returns
    -------
    pd.DataFrame — copy with 'bias_4h' column appended.
    """
    if not isinstance(df_1h.index, pd.DatetimeIndex):
        raise ValueError("add_4h_context() requires a DatetimeIndex.")
    if "close" not in df_1h.columns:
        raise ValueError("add_4h_context() requires a 'close' column.")

    fast_w = config.CONTEXT_4H_SMA_FAST
    slow_w = config.CONTEXT_4H_SMA_SLOW

    # ── Resample 1H → 4H (last close of each 4-hour window) ──────────────────
    close_4h = (
        df_1h["close"]
        .resample("4h", closed="left", label="left")
        .last()
        .dropna()
    )

    # ── Dual-SMA crossover on 4H bars ─────────────────────────────────────────
    sma_fast = close_4h.rolling(fast_w).mean()
    sma_slow = close_4h.rolling(slow_w).mean()

    valid = sma_fast.notna() & sma_slow.notna()

    bias_4h = pd.Series("neutral", index=close_4h.index, dtype=object)
    bias_4h[valid & (close_4h > sma_slow) & (sma_fast > sma_slow)] = "bull"
    bias_4h[valid & (close_4h < sma_slow) & (sma_fast < sma_slow)] = "bear"

    # ── Shift 1 bar then forward-fill to 1H resolution ───────────────────────
    # shift(1) ensures the 1H bar at T sees only closed 4H bars before T.
    bias_1h = (
        bias_4h.shift(1)
        .reindex(df_1h.index, method="ffill")
        .fillna("neutral")
    )

    result = df_1h.copy()
    result["bias_4h"] = bias_1h
    return result


def apply_4h_context_filter(
    signals_df: pd.DataFrame,
    mode: str | None = None,
) -> pd.DataFrame:
    """
    Gate trade signals by 4H directional bias.

    Reads the existing 'signal' column and writes 'signal_4h_filtered':
      - Long  signals (+1) that contradict the 4H bias are zeroed.
      - Short signals (-1) that contradict the 4H bias are zeroed.
      - The original 'signal' column is preserved unchanged.

    Parameters
    ----------
    signals_df : DataFrame with 'signal' and 'bias_4h' columns.
    mode       : 'strict' or 'relaxed'.  Defaults to config.CONTEXT_4H_MODE.

    Returns
    -------
    pd.DataFrame — copy with 'signal_4h_filtered' column added.
    """
    for col in ("signal", "bias_4h"):
        if col not in signals_df.columns:
            raise ValueError(
                f"apply_4h_context_filter() requires '{col}' column. "
                f"Call add_4h_context() before generating signals."
            )

    if mode is None:
        mode = config.CONTEXT_4H_MODE

    if mode not in ("strict", "relaxed"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'strict' or 'relaxed'.")

    result = signals_df.copy()
    sig    = result["signal"].copy().astype(int)
    bias   = result["bias_4h"]

    if mode == "strict":
        long_ok  = bias == "bull"
        short_ok = bias == "bear"
    else:  # relaxed
        long_ok  = bias.isin(["bull",  "neutral"])
        short_ok = bias.isin(["bear",  "neutral"])

    sig[(sig ==  1) & ~long_ok]  = 0
    sig[(sig == -1) & ~short_ok] = 0

    result["signal_4h_filtered"] = sig
    return result


def print_4h_bias_stats(df: pd.DataFrame) -> None:
    """
    Print distribution of 4H bias labels across a DataFrame.

    Parameters
    ----------
    df : DataFrame with 'bias_4h' column present.
    """
    if "bias_4h" not in df.columns:
        return

    n   = len(df)
    sep = "-" * 48

    print(f"\n  4H Context Bias Distribution  ({n:,} bars)")
    print(f"  {sep}")
    for label in ["bull", "neutral", "bear"]:
        count = (df["bias_4h"] == label).sum()
        pct   = count / n * 100
        print(f"    {label:<10} {count:>6,} bars  ({pct:>5.1f}%)")
    print(f"  {sep}")

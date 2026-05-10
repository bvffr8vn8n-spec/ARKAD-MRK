"""
features/execution_15m.py — 15m execution layer (Approach B: pullback entry).

Role
----
The 1H RF model fires a directional signal at bar close.  The current backtest
enters at that close price immediately.  This module looks at the first 2 hours
of 15m bars AFTER the signal and tries to find a better entry:

  Pullback entry (Approach B)
  ---------------------------
  After a 1H long signal, wait for any 15m bar to close below the 1H signal
  bar close (a short counter-move / pullback).  Once that happens, wait for the
  next 15m bar that closes above its own open (bullish resumption).  Enter there.

  If no pullback occurs within max_wait bars: fall back to the 1H close price
  (baseline entry).  The trade always executes — no signals are dropped.

Output column added to signals_df
----------------------------------
  entry_price_15m : float  — override entry price for the backtest engine.
                             NaN means "use the default 1H close".

Look-ahead safety
-----------------
  entry_start = ts + 1H (one full bar after the signal bar closes).
  Only 15m bars with index >= entry_start are examined.
  No intra-bar look-ahead.

15m data coverage
-----------------
  The 15m dataset covers 2024-01-01 onward.  For 1H signals before that date,
  df_15m[index >= entry_start] is empty → fallback to baseline automatically.
  Use coverage_stats() to report how many signals had 15m data available.
"""

import math

import numpy as np
import pandas as pd


# 14-bar rolling mean of True Range on 15m bars (3.5 hours of lookback).
_ATR_WINDOW_15M = 14


# ── Public API ────────────────────────────────────────────────────────────────

def load_15m_data(path: str) -> pd.DataFrame:
    """
    Load 15m OHLCV CSV and attach a rolling ATR column.

    Expected CSV columns: date, open, high, low, close, volume
    The 'date' column is parsed as the index.
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df = _add_atr(df)
    return df


def load_5m_as_15m(path: str) -> pd.DataFrame:
    """
    Load a 5m OHLCV CSV, resample to 15m bars, and attach a rolling ATR.

    Each group of three consecutive 5m bars is aggregated into one 15m bar:
      open   = first  bar open
      high   = max    bar high
      low    = min    bar low
      close  = last   bar close
      volume = sum    bar volume

    Bars where any of open/high/low/close is missing after aggregation are
    dropped (start-of-dataset warmup).

    Use this instead of load_15m_data() when you have a 5m file but not a
    native 15m file.  The ATR is computed on the resampled 15m bars, so it
    matches what load_15m_data() would produce.
    """
    df5 = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df15 = (
        df5.resample("15min", closed="left", label="left")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["open", "close"])
    )
    df15 = _add_atr(df15)
    return df15


def annotate_signals_B(
    signals_df: pd.DataFrame,
    df_15m:     pd.DataFrame,
    max_wait:   int = 8,
) -> pd.DataFrame:
    """
    Apply Approach B (pullback entry) to every non-zero signal in signals_df.

    Parameters
    ----------
    signals_df : 1H signal DataFrame with 'signal' and 'close' columns.
    df_15m     : 15m OHLCV DataFrame with DatetimeIndex (from load_15m_data()).
    max_wait   : max number of 15m bars to scan per signal (default 8 = 2 hours).

    Returns
    -------
    pd.DataFrame — copy of signals_df with 'entry_price_15m' column added.
      NaN  → no pullback found (or no 15m data) → engine uses 1H close.
      float → pullback entry price → engine uses this price instead.
    """
    result = signals_df.copy()
    result["entry_price_15m"] = np.nan

    # Pre-sort 15m index for fast binary search
    idx_15m = df_15m.index

    for ts, row in result.iterrows():
        sig = int(row["signal"])
        if sig == 0:
            continue

        direction   = sig                               # +1 long, -1 short
        entry_start = ts + pd.Timedelta(hours=1)       # first 15m bar after signal

        # Guard: entry_start must fall within the 15m dataset's date range.
        # searchsorted returns 0 when entry_start is before the first bar,
        # which would silently look up future 15m data — a look-ahead bug.
        if len(idx_15m) == 0 or entry_start < idx_15m[0]:
            continue                                    # before 15m data starts → fallback

        # Fast lookup: binary search for entry_start in 15m index
        pos = idx_15m.searchsorted(entry_start)
        end = min(pos + max_wait, len(idx_15m))
        if pos >= len(idx_15m):
            continue                                    # beyond 15m data → fallback

        window = df_15m.iloc[pos:end]
        if len(window) == 0:
            continue                                    # no 15m data → fallback

        ref_price = float(row["close"])
        entry_px  = _find_pullback(window, direction, ref_price)

        if entry_px is not None:
            result.at[ts, "entry_price_15m"] = entry_px

    return result


def annotate_signals_A(
    signals_df: pd.DataFrame,
    df_15m:     pd.DataFrame,
    k_bars:     int = 4,
    min_aligned: int = 2,
) -> pd.DataFrame:
    """
    Apply Approach A (momentum consistency filter) to every non-zero signal.

    Looks at the first k_bars 15m bars after the 1H signal close.
    Counts how many bars close in the signal direction (close > open for long,
    close < open for short).  If fewer than min_aligned bars align → zero out
    the signal (skip the trade entirely).

    Parameters
    ----------
    signals_df  : 1H signal DataFrame with 'signal' column.
    df_15m      : 15m OHLCV DataFrame with DatetimeIndex (from load_15m_data()).
    k_bars      : number of 15m bars to examine (default 4 = 1 hour).
    min_aligned : minimum aligned bars required to take the trade (default 2).

    Returns
    -------
    pd.DataFrame — copy of signals_df with 'signal_15m_A' column added.
      Inherits original signal value when:
        (a) no 15m data is available (look-ahead guard → keep trade)
        (b) aligned bars >= min_aligned (momentum consistent → take trade)
      Set to 0 when aligned bars < min_aligned (counter-momentum → skip).

    Fallback behaviour
    ------------------
    When entry_start is before the 15m dataset's first bar, the signal is
    kept unchanged (baseline pass-through).  This is intentional: if we
    cannot evaluate 15m momentum, we default to taking the trade rather
    than skipping it.
    """
    result  = signals_df.copy()
    result["signal_15m_A"] = result["signal"].copy()

    idx_15m = df_15m.index

    for ts, row in result.iterrows():
        sig = int(row["signal"])
        if sig == 0:
            continue

        direction   = sig
        entry_start = ts + pd.Timedelta(hours=1)

        # Look-ahead guard: do not use 15m bars from before the signal date.
        if len(idx_15m) == 0 or entry_start < idx_15m[0]:
            continue                    # no 15m data → keep signal (baseline)

        pos = idx_15m.searchsorted(entry_start)
        if pos >= len(idx_15m):
            continue                    # after 15m data ends → keep signal

        end    = min(pos + k_bars, len(idx_15m))
        window = df_15m.iloc[pos:end]

        if len(window) < k_bars:
            continue                    # insufficient bars → keep signal

        aligned = int(((window["close"] - window["open"]) * direction > 0).sum())
        if aligned < min_aligned:
            result.at[ts, "signal_15m_A"] = 0

    return result


def annotate_signals_AB(
    signals_df: pd.DataFrame,
    df_15m:     pd.DataFrame,
    k_bars:     int   = 4,
    min_aligned: int  = 2,
    max_wait:   int   = 8,
) -> pd.DataFrame:
    """
    Apply Approach A then Approach B in sequence.

    Step 1 — A (hard filter): zero out signals where fewer than min_aligned
      of the first k_bars 15m bars close in the signal direction.

    Step 2 — B (pullback entry): for every signal that survived A, search
      for a pullback-then-resumption entry price.  Fallback to 1H close if
      no pullback found (no trade lost from the A-filtered set).

    Columns added
    -------------
    signal_15m_A    : A-filtered signal (0 where A blocked, original otherwise)
    entry_price_15m : pullback entry price for signals that passed A
                      (NaN = no pullback found or no 15m data → 1H baseline price)

    The backtest engine reads both:
      signal          ← caller overwrites from signal_15m_A
      entry_price_15m ← engine reads automatically when column exists
    """
    # Step 1: apply A → get signal_15m_A
    ann_a = annotate_signals_A(signals_df, df_15m,
                                k_bars=k_bars, min_aligned=min_aligned)

    # Step 2: apply B only on A-surviving signals.
    # Build a view with signal = signal_15m_A so annotate_signals_B skips A-blocked rows.
    temp = signals_df.copy()
    temp["signal"] = ann_a["signal_15m_A"]
    ann_b = annotate_signals_B(temp, df_15m, max_wait=max_wait)

    # Combine into a single result
    result = signals_df.copy()
    result["signal_15m_A"]    = ann_a["signal_15m_A"]
    result["entry_price_15m"] = ann_b["entry_price_15m"]
    return result


def coverage_stats(
    signals_df:  pd.DataFrame,
    annotated:   pd.DataFrame,
    df_15m:      pd.DataFrame,
    window_label: str = "test",
) -> dict:
    """
    Return a dict of coverage statistics for this signals_df window.

    Counts
    ------
    total_signals    : non-zero signals in the window
    signals_with_15m : signals whose entry_start falls within df_15m date range
    pullback_entries : signals that got an actual pullback price override
    baseline_fallback: signals with 15m data but no pullback found (used 1H price)
    no_15m_data      : signals outside 15m coverage entirely
    """
    if len(df_15m) == 0:
        min_15m = pd.Timestamp.max
        max_15m = pd.Timestamp.min
    else:
        min_15m = df_15m.index[0]
        max_15m = df_15m.index[-1]

    active = signals_df[signals_df["signal"] != 0]
    total  = len(active)

    if total == 0:
        return {
            "window":           window_label,
            "total_signals":    0,
            "signals_with_15m": 0,
            "pullback_entries": 0,
            "baseline_fallback":0,
            "no_15m_data":      0,
            "pullback_rate":    float("nan"),
        }

    entry_starts      = active.index + pd.Timedelta(hours=1)
    has_15m           = (entry_starts >= min_15m) & (entry_starts <= max_15m)
    n_with_15m        = int(has_15m.sum())
    n_no_15m          = total - n_with_15m

    # Pullback count: has 15m data AND entry_price_15m is not NaN
    if "entry_price_15m" in annotated.columns:
        ann_active    = annotated.loc[active.index]
        n_pullback    = int(ann_active["entry_price_15m"].notna().sum())
    else:
        n_pullback    = 0

    n_fallback = n_with_15m - n_pullback

    return {
        "window":           window_label,
        "total_signals":    total,
        "signals_with_15m": n_with_15m,
        "pullback_entries": n_pullback,
        "baseline_fallback":n_fallback,
        "no_15m_data":      n_no_15m,
        "pullback_rate":    n_pullback / n_with_15m if n_with_15m > 0 else float("nan"),
    }


def print_coverage_stats(stats: dict) -> None:
    """Print a formatted coverage stats block."""
    sep = "-" * 52
    print(f"\n  15m Coverage — {stats['window']}")
    print(f"  {sep}")
    print(f"    Total signals         : {stats['total_signals']:>4}")
    print(f"    With 15m data         : {stats['signals_with_15m']:>4}")
    print(f"    Pullback entries found: {stats['pullback_entries']:>4}"
          f"  (rate={_fmt_pct(stats['pullback_rate'])})")
    print(f"    Baseline fallback     : {stats['baseline_fallback']:>4}")
    print(f"    No 15m data           : {stats['no_15m_data']:>4}")
    print(f"  {sep}")


# ── Private helpers ───────────────────────────────────────────────────────────

def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 14-bar simple rolling ATR to the 15m DataFrame."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    out = df.copy()
    out["atr_15m"] = tr.rolling(_ATR_WINDOW_15M).mean()
    return out


def _find_pullback(
    window:    pd.DataFrame,
    direction: int,
    ref_price: float,
) -> float | None:
    """
    Scan up to len(window) 15m bars for a pullback-then-resumption pattern.

    State machine:
      phase 0 → waiting for counter-move past ref_price
      phase 1 → counter-move seen; waiting for a bar that closes in signal dir
                (close > open for long, close < open for short)

    Returns
    -------
    float  — close price of the confirmation bar (entry price override)
    None   — no pattern found within window → use baseline
    """
    phase = 0
    for _, bar in window.iterrows():
        close = float(bar["close"])
        open_ = float(bar["open"])

        if phase == 0:
            # Detect pullback: price moves AGAINST the signal direction past ref_price
            if direction == 1  and close < ref_price:
                phase = 1
            elif direction == -1 and close > ref_price:
                phase = 1

        elif phase == 1:
            # Detect resumption: bar closes in signal direction
            if direction == 1  and close > open_:
                return close
            elif direction == -1 and close < open_:
                return close

    return None


def _fmt_pct(v: float) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v * 100:.0f}%"

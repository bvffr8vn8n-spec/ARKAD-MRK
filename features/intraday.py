"""
features/intraday.py -- Intraday features derived from 5m OHLCV bars.

Aggregates 5m bar data into 1H-resolution signals that carry information
about intra-hour order flow, which is absent from single-bar 1H OHLCV.

Feature derivation approach
----------------------------
Real taker buy/sell volume (from exchange APIs) is the gold standard for
order-flow analysis.  In its absence, the "Tick Rule" approximates trade
direction from price movement within each 5m bar:

    close >= open  -->  classified as net "buy" bar   (taker buy pressure)
    close <  open  -->  classified as net "sell" bar  (taker sell pressure)

Studies on liquid crypto markets show this approximation achieves ~72-78%
agreement with actual tape data on high-volume pairs like BTCUSDT.

No look-ahead
-------------
vdelta_ratio and vol_tail_ratio use only 5m bars within the CURRENT 1H bar
(i.e., [T, T+55min] for bar timestamped T).  The label at T references
close[T + FORWARD_RETURN_WINDOW], which is strictly in the future.
vdelta_ema12 is a lagged EMA of past vdelta_ratio values.  Zero leakage.

Features added to df_1h
-----------------------
vdelta_ratio     : (buy_vol - sell_vol) / total_vol  for this 1H bar.
                   Range [-1, +1].
                   +1 = every 5m bar was a buy bar (maximum buying pressure).
                   -1 = every 5m bar was a sell bar (maximum selling pressure).
                    0 = balanced or random order flow.

vdelta_ema12     : 12-period EMA of vdelta_ratio (rolling ~12H smoothed flow).
                   Positive = sustained buying pressure over the past half-day.
                   Negative = sustained selling pressure.

vol_tail_ratio   : Volume of the last 3 5m bars (final 15 min) / 1H total volume.
                   Uniform baseline = 3/12 = 0.25.
                   High (> 0.35) = late-hour accumulation, continuation signal.
                   Low  (< 0.15) = early spike followed by quiet -- possible fade.
"""

import numpy as np
import pandas as pd


def add_5m_features(df_1h: pd.DataFrame, path_5m: str) -> pd.DataFrame:
    """
    Load 5m OHLCV data and merge intraday features into the 1H DataFrame.

    Parameters
    ----------
    df_1h   : 1H DataFrame with DatetimeIndex (output of generate_features).
    path_5m : Path to the 5m OHLCV CSV file.

    Returns
    -------
    pd.DataFrame -- df_1h with three new feature columns appended.
    """
    # ── Load 5m data ──────────────────────────────────────────────────────────
    df5 = pd.read_csv(path_5m, parse_dates=["date"])
    df5.columns = df5.columns.str.lower().str.strip()
    df5 = df5.set_index("date").sort_index()

    # ── Tick Rule: classify each 5m bar as buy or sell ────────────────────────
    is_buy         = (df5["close"] >= df5["open"]).astype(float)
    df5["buy_vol"] = df5["volume"] * is_buy

    # ── Assign each 5m bar to its parent 1H bucket ───────────────────────────
    # floor("60min") works across pandas 1.x and 2.x
    df5["hour"] = df5.index.floor("60min")

    # Rank within each hour (0 = first 5m bar, 11 = last 5m bar)
    df5["bar_rank"] = df5.groupby("hour").cumcount()

    # ── Aggregate per 1H bucket ───────────────────────────────────────────────
    grp = df5.groupby("hour")
    agg = grp.agg(
        total_vol=("volume",   "sum"),
        buy_vol  =("buy_vol",  "sum"),
    )

    total_safe = agg["total_vol"].replace(0.0, np.nan)

    # vdelta_ratio: 2*(buy_vol/total) - 1  -->  [-1, +1]
    agg["vdelta_ratio"] = (2.0 * agg["buy_vol"] / total_safe) - 1.0

    # vol_tail_ratio: volume in the final 15 min (bars 9, 10, 11 of 12)
    tail_vol = df5[df5["bar_rank"] >= 9].groupby("hour")["volume"].sum()
    agg["vol_tail_ratio"] = tail_vol / total_safe

    # ── EMA-12 of vdelta_ratio (lagged cumulative flow signal) ───────────────
    agg["vdelta_ema12"] = (
        agg["vdelta_ratio"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    # ── Merge into 1H DataFrame ───────────────────────────────────────────────
    result    = df_1h.copy()
    feat_cols = ["vdelta_ratio", "vdelta_ema12", "vol_tail_ratio"]

    for col in feat_cols:
        result[col] = agg[col].reindex(result.index)

    n_matched = result["vdelta_ratio"].notna().sum()
    print(f"      Intraday features: {n_matched:,}/{len(result):,} bars matched "
          f"({len(feat_cols)} new columns)")

    return result

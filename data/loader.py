"""
data/loader.py — Loads historical OHLCV data from a CSV file.

Expected CSV format (column names are configurable in config.py):
    date, open, high, low, close, volume

The loader normalises column names to lowercase, parses the date index,
and validates that all required OHLCV columns are present.
"""

import pandas as pd
import config


def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a clean OHLCV DataFrame indexed by date.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    df = pd.read_csv(path, parse_dates=[config.DATE_COLUMN])
    df.columns = df.columns.str.lower().str.strip()

    # Validate required columns
    required = list(config.OHLCV_COLUMNS.values()) + [config.DATE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df.set_index(config.DATE_COLUMN, inplace=True)
    df.sort_index(inplace=True)

    # Keep only OHLCV columns (drop extras silently)
    ohlcv = list(config.OHLCV_COLUMNS.values())
    df = df[ohlcv].copy()

    # Cast to float, drop rows with any NaN in OHLCV
    df = df.astype(float).dropna()

    return df

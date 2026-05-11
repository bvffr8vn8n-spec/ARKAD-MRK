"""
data/generate_sample.py — Generates a synthetic OHLCV CSV for testing.

Usage:
    python data/generate_sample.py
    python data/generate_sample.py --bars 2000 --out data/sample.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_ohlcv(n_bars: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Simulate a realistic price path with drift + noise + occasional jumps
    returns    = rng.normal(0.0003, 0.012, n_bars)
    jump_mask  = rng.random(n_bars) < 0.01          # 1% chance of a jump
    jump_size  = rng.choice([-1, 1], n_bars) * rng.uniform(0.02, 0.06, n_bars)
    returns   += jump_mask * jump_size

    close = 100 * np.exp(np.cumsum(returns))

    # Build OHLCV from close
    noise = rng.uniform(0.002, 0.015, n_bars)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.008, 0.008, n_bars))
    vol   = (rng.lognormal(10, 0.5, n_bars)).astype(int)

    dates = pd.bdate_range(end="2024-12-31", periods=n_bars)

    df = pd.DataFrame({
        "date":   dates,
        "open":   open_.round(4),
        "high":   high.round(4),
        "low":    low.round(4),
        "close":  close.round(4),
        "volume": vol,
    })
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=1500)
    parser.add_argument("--out",  default="data/sample.csv")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(exist_ok=True)
    df = generate_ohlcv(args.bars)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} bars → {args.out}")


if __name__ == "__main__":
    main()

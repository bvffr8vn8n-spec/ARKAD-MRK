# ARKAD MRK — Automated Trading Research Lab

A modular Python framework for discovering statistically promising trading patterns
from historical market data. No fixed strategy is hardcoded — patterns emerge from data.

## Project Structure

```
ARKAD MRK/
├── data/               # Data loading and storage utilities
├── features/           # Feature engineering (candidate signal generation)
├── models/             # ML model training and signal production
├── backtest/           # Trade simulation and performance metrics
├── experiments/        # Saved experiment configs and results
├── reports/            # Generated HTML/text performance reports
├── config.py           # Global settings and parameters
├── main.py             # Entry point — runs the full research pipeline
└── requirements.txt    # Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --data data/sample.csv
```

## Pipeline Overview

1. **Load** — Read OHLCV CSV into a clean DataFrame
2. **Features** — Generate technical and statistical candidate features
3. **Label** — Create forward-return targets for supervised learning
4. **Train** — Fit a baseline ML classifier to predict directional moves
5. **Signal** — Convert model probabilities into trade signals (buy/sell/hold)
6. **Backtest** — Simulate trades on out-of-sample data
7. **Report** — Print win rate, risk-reward, expectancy, profit factor, max drawdown

## Key Metrics Reported

| Metric | Description |
|---|---|
| Win Rate | % of trades that are profitable |
| Risk-Reward Ratio | Average win size / average loss size |
| Expectancy | Expected P&L per trade (edge per dollar risked) |
| Profit Factor | Gross profit / gross loss |
| Max Drawdown | Largest peak-to-trough equity decline |

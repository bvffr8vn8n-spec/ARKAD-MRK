"""
backtest/metrics.py — Computes performance statistics from a list of trades.

Metrics
-------
win_rate        : fraction of trades with net_pnl > 0
avg_win         : mean P&L of winning trades
avg_loss        : mean P&L of losing trades (positive number)
risk_reward     : avg_win / avg_loss
expectancy      : win_rate * avg_win - loss_rate * avg_loss  (per trade)
profit_factor   : gross_profit / gross_loss
max_drawdown    : largest peak-to-trough decline in the equity curve
"""

import pandas as pd


def compute_metrics(trades: list[dict], equity_curve: pd.Series) -> dict:
    if not trades:
        return {"error": "No trades to evaluate."}

    pnls   = [t["net_pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades    = len(pnls)
    win_rate    = len(wins) / n_trades
    loss_rate   = 1 - win_rate

    avg_win     = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss    = abs(sum(losses) / len(losses)) if losses else 0.0

    risk_reward   = avg_win / avg_loss if avg_loss > 0 else float("inf")
    expectancy    = win_rate * avg_win - loss_rate * avg_loss

    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_drawdown  = _max_drawdown(equity_curve)

    total_return  = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # Exit reason breakdown (present only if engine recorded exit_reason)
    exit_reasons = {}
    if trades and "exit_reason" in trades[0]:
        for reason in ("stop", "tp", "time"):
            exit_reasons[f"exit_{reason}"] = sum(
                1 for t in trades if t.get("exit_reason") == reason
            )

    return {
        "n_trades":      n_trades,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "risk_reward":   risk_reward,
        "expectancy":    expectancy,
        "profit_factor": profit_factor,
        "max_drawdown":  max_drawdown,
        "total_return":  total_return,
        "final_equity":  equity_curve.iloc[-1],
        **exit_reasons,
    }


def _max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown as a positive percentage (e.g. 15.3 = 15.3% drawdown)."""
    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100
    return abs(dd.min())

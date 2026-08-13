"""Performance metrics computed from a backtest.

Everything is calculated from two simple inputs: the equity curve (how
our account value changed day by day) and the list of trades. The result
is a dict of key figures like Sharpe ratio, max drawdown and win rate
that get stored and shown on the dashboard.
"""

import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    equity: pd.Series,
    trades: List[Dict[str, Any]],
    initial_capital: float,
) -> Dict[str, float]:
    """Compute the standard set of backtest performance metrics.

    Args:
        equity: daily account value indexed by date.
        trades: list of trade dicts with a pnl key per closed round trip.
        initial_capital: starting account balance.

    Returns:
        A dict of metric names to values, safe to store as JSON.
    """
    if equity.empty:
        raise ValueError("Equity curve is empty, nothing to measure")

    final_equity = float(equity.iloc[-1])
    total_return = final_equity / initial_capital - 1

    # Daily percentage returns drive everything that involves risk.
    daily_returns = equity.pct_change().dropna()

    # Compounded annual growth rate. n is the number of years of data.
    n_years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1e-9)
    cagr = (final_equity / initial_capital) ** (1 / n_years) - 1

    # Sharpe ratio: how much return we get per unit of risk, annualized.
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = 0.0

    # Max drawdown: the worst peak to trough drop of the account.
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    # Trade based metrics. A round trip ends with a SELL that carries pnl.
    pnls = [t.get("pnl", 0.0) for t in trades if t.get("action") == "SELL"]
    num_trades = len(pnls)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / num_trades if num_trades else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    # No losses means an infinite profit factor, which JSON cannot store
    # (PostgreSQL rejects it), so we send null instead.
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "final_equity": final_equity,
        "volatility": float(daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)),
    }
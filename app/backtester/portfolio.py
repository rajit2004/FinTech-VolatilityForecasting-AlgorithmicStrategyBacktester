"""Multi-asset portfolio backtesting.

Running a strategy on a single asset tells you nothing about how it
performs as part of a portfolio. Diversification across uncorrelated
assets reduces drawdown because losses in one asset are partially
offset by gains in another.

This module runs a backtest on each symbol independently, then
combines the results into a portfolio-level equity curve. The portfolio
starts with equal capital allocated to each symbol and the combined
equity is the sum of all individual equity curves.

This is a simplified approach: in a real portfolio you might rebalance
periodically or weight assets by risk, but equal weighting is the
honest baseline that everything else should beat.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from app.backtester.engine import BacktestConfig, run_backtest
from app.backtester.metrics import compute_metrics
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_portfolio_backtest(
    data: Dict[str, pd.DataFrame],
    strategy_name: str,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, Any]:
    """Run a strategy across multiple symbols and combine results.

    Each symbol gets an equal share of the initial capital. The
    individual backtests run independently (no cross-asset position
    sizing), and the portfolio equity is the sum of all accounts.

    Args:
        data: mapping of symbol to cleaned OHLCV DataFrame.
        strategy_name: which strategy to run on every symbol.
        params: strategy tuning parameters, shared across symbols.
        config: backtest settings. initial_capital is split equally.

    Returns:
        A dict with per-symbol results, combined equity curve, and
        portfolio-level metrics.

    Raises:
        ValueError: if data is empty.
    """
    if not data:
        raise ValueError("Cannot run a portfolio backtest with no symbols")

    symbols = sorted(data.keys())
    n_symbols = len(symbols)

    # Split initial capital equally across symbols.
    base_config = config or BacktestConfig()
    per_symbol_capital = base_config.initial_capital / n_symbols
    symbol_config = BacktestConfig(
        initial_capital=per_symbol_capital,
        commission=base_config.commission,
        position_pct=base_config.position_pct,
        slippage=base_config.slippage,
        target_volatility=base_config.target_volatility,
        forecast_volatility=base_config.forecast_volatility,
        stop_loss_pct=base_config.stop_loss_pct,
        take_profit_pct=base_config.take_profit_pct,
    )

    individual_results = {}
    equity_frames = []

    for symbol in symbols:
        try:
            result = run_backtest(
                symbol=symbol,
                df=data[symbol],
                strategy_name=strategy_name,
                params=params,
                config=symbol_config,
            )
            individual_results[symbol] = result

            # Build a Series of this symbol's equity for combination.
            equity_series = pd.Series(
                [e["equity"] for e in result.equity_curve],
                index=pd.to_datetime([e["date"] for e in result.equity_curve]),
                name=symbol,
            )
            equity_frames.append(equity_series)
        except Exception as exc:  # noqa: BLE001, one bad symbol should not kill the run
            logger.error("Portfolio backtest failed for %s: %s", symbol, exc)

    if not equity_frames:
        raise ValueError("All symbols failed in the portfolio backtest")

    # Align all equity curves on their common dates and sum them.
    combined = pd.concat(equity_frames, axis=1, join="outer")
    combined = combined.ffill().fillna(0.0)
    portfolio_equity = combined.sum(axis=1)

    # Compute portfolio-level metrics from the combined equity curve.
    all_trades = []
    for result in individual_results.values():
        all_trades.extend(result.trades)

    portfolio_metrics = compute_metrics(
        portfolio_equity, all_trades, base_config.initial_capital
    )

    equity_curve = [
        {"date": idx.date().isoformat(), "equity": round(float(val), 2)}
        for idx, val in portfolio_equity.items()
    ]

    return {
        "symbols": symbols,
        "n_symbols": n_symbols,
        "strategy_name": strategy_name,
        "params": params or {},
        "per_symbol_capital": per_symbol_capital,
        "portfolio_metrics": portfolio_metrics,
        "portfolio_equity_curve": equity_curve,
        "individual_results": {
            sym: {
                "metrics": res.metrics,
                "num_trades": len(res.trades),
            }
            for sym, res in individual_results.items()
        },
    }

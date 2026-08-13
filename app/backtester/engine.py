"""The core backtesting engine.

The engine simulates trading a strategy over historical data, bar by bar.
On each day it checks what position the strategy wants, executes a trade
if that is different from the position we currently hold, books a small
commission, and records the account value.

Two course concepts live here:
- A generator (iter_bar_events) streams out one event per bar so we can
  process a long history without building every intermediate object.
- Multiprocessing (run_multiple_backtests) runs several parameter
  combinations at the same time, which would be far too slow serially.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

from app.backtester.metrics import compute_metrics
from app.strategies.base_strategy import BaseStrategy
from app.strategies.moving_average import MovingAverageCrossover
from app.strategies.volatility_breakout import VolatilityBreakout
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Settings that control the simulation, not the strategy itself."""

    initial_capital: float = 100_000.0
    commission: float = 0.001  # fraction of the trade value, 0.1%
    position_pct: float = 0.95  # how much of the account we invest
    slippage: float = 0.0  # per share price slippage, zero for simplicity


@dataclass
class BacktestResult:
    """The full output of one backtest run."""

    symbol: str
    strategy_name: str
    params: Dict[str, Any]
    config: BacktestConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten the result into a plain dict for the API and database."""
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "params": self.params,
            "config": self.config.__dict__,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }


def get_strategy(strategy_name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
    """Create a strategy instance from its name and parameters.

    Args:
        strategy_name: 'moving_average' or 'volatility_breakout'.
        params: tuning parameters for the strategy.

    Returns:
        A ready to use BaseStrategy instance.

    Raises:
        ValueError: if the strategy name is unknown.
    """
    params = dict(params or {})
    if strategy_name == "moving_average":
        return MovingAverageCrossover(params)
    if strategy_name == "volatility_breakout":
        return VolatilityBreakout(params)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def iter_bar_events(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    config: BacktestConfig,
) -> Generator[Dict[str, Any], None, None]:
    """Yield one event dict per bar, the heart of the simulation.

    A generator lets the loop run lazily: the caller can pull events one
    by one and each event is only created when needed.

    Args:
        df: cleaned OHLCV data indexed by date.
        strategy: the strategy to simulate.
        config: simulation settings.

    Yields:
        A dict per bar with date, price, signal, cash, shares and equity.
    """
    signals = strategy.generate_signals(df)

    cash = config.initial_capital
    shares = 0
    position = 0  # current actual position, updated on trades
    cost_basis = 0.0  # total paid for the shares we currently hold

    for date, row in df.iterrows():
        price = float(row["close"])
        desired = float(signals.loc[date])

        # Trade only when the strategy changes its mind.
        if desired != position:
            if desired > 0 and shares == 0:
                # Buy with a fraction of our cash, integer shares only.
                budget = cash * config.position_pct
                shares = max(int(budget / price), 0)
                if shares > 0:
                    cost = shares * price
                    fee = cost * config.commission
                    cash -= cost + fee
                    cost_basis = cost + fee
            elif desired == 0 and shares > 0:
                # Sell everything we hold and book the round trip pnl.
                proceeds = shares * price
                fee = proceeds * config.commission
                cash += proceeds - fee
                pnl = proceeds - fee - cost_basis
                cost_basis = 0.0
            position = desired
            shares = shares if desired > 0 else 0

        equity = cash + shares * price
        yield {
            "date": date.date().isoformat(),
            "price": price,
            "signal": desired,
            "cash": cash,
            "shares": shares,
            "equity": equity,
        }


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    strategy_name: str,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run one backtest end to end.

    Args:
        symbol: ticker being tested.
        df: cleaned OHLCV data indexed by date.
        strategy_name: which strategy to simulate.
        params: strategy tuning parameters.
        config: simulation settings, defaults used when omitted.

    Returns:
        A BacktestResult with metrics, equity curve and trades.

    Raises:
        ValueError: if the frame is empty or the strategy is unknown.
    """
    if df is None or df.empty:
        raise ValueError("Cannot backtest with an empty data frame")

    config = config or BacktestConfig()
    strategy = get_strategy(strategy_name, params)
    logger.info(
        "Running %s on %s with params %s", strategy_name, symbol, strategy.params
    )

    events = list(iter_bar_events(df, strategy, config))

    equity_curve = [{"date": e["date"], "equity": round(e["equity"], 2)} for e in events]

    # Build one trade dict per entry and exit so the DB and dashboard
    # can show the full trading history.
    trades: List[Dict[str, Any]] = []
    shares_held = 0
    for e in events:
        signal = e["signal"]
        if signal > 0 and shares_held == 0 and e["shares"] > 0:
            trades.append(
                {
                    "date": e["date"],
                    "action": "BUY",
                    "price": e["price"],
                    "size": e["shares"],
                    "pnl": 0.0,
                }
            )
            shares_held = e["shares"]
        elif signal == 0 and shares_held > 0:
            # Approximate the exit price with the current close and the
            # pnl is calculated by the metrics using the closing trade.
            trades.append(
                {
                    "date": e["date"],
                    "action": "SELL",
                    "price": e["price"],
                    "size": shares_held,
                    "pnl": 0.0,
                }
            )
            shares_held = 0

    equity_series = pd.Series(
        [e["equity"] for e in equity_curve],
        index=pd.to_datetime([e["date"] for e in equity_curve]),
    )
    metrics = compute_metrics(equity_series, trades, config.initial_capital)

    return BacktestResult(
        symbol=symbol.upper(),
        strategy_name=strategy_name,
        params=strategy.params,
        config=config,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
    )


def _run_one(
    symbol: str,
    df: pd.DataFrame,
    strategy_name: str,
    params: Dict[str, Any],
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one parameter combination, meant for the worker processes.

    This has to be a module level function (not a method) so it can be
    pickled and sent to another process on Windows.

    Args:
        symbol: ticker.
        df: the price data frame, pickled to the worker.
        strategy_name: strategy to run.
        params: strategy parameters.
        config_dict: backtest config as a plain dict.

    Returns:
        A compact result dict ready to be collected.
    """
    config = BacktestConfig(**config_dict)
    result = run_backtest(symbol, df, strategy_name, params, config)
    payload = result.to_dict()
    return {
        "symbol": payload["symbol"],
        "strategy_name": payload["strategy_name"],
        "params": payload["params"],
        "metrics": payload["metrics"],
    }


def run_multiple_backtests(
    symbol: str,
    df: pd.DataFrame,
    strategy_name: str,
    param_grid: List[Dict[str, Any]],
    config: Optional[BacktestConfig] = None,
    max_workers: int = 2,
    parallel: bool = True,
) -> List[Dict[str, Any]]:
    """Run a grid of parameter sets, optionally in parallel.

    Trying out many parameter combinations is slow, so we use a process
    pool to run them on multiple cores. Each worker gets one combination.

    Args:
        symbol: ticker.
        df: the price data frame, copied to every worker.
        strategy_name: strategy to run.
        param_grid: list of parameter dicts, one entry per run.
        config: simulation settings.
        max_workers: how many processes to spawn.
        parallel: set False to run serially (useful in tests).

    Returns:
        A list of compact result dicts, one per parameter set.
    """
    config = config or BacktestConfig()
    results: List[Dict[str, Any]] = []

    if not parallel:
        for params in param_grid:
            results.append(_run_one(symbol, df, strategy_name, params, config.__dict__))
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_one, symbol, df, strategy_name, params, config.__dict__)
            for params in param_grid
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.error("A parallel backtest failed: %s", exc)

    return results
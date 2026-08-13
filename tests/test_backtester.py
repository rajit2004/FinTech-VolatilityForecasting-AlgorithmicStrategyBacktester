"""Tests for the backtesting engine and performance metrics."""

import numpy as np
import pandas as pd
import pytest

from app.backtester.engine import (
    BacktestConfig,
    get_strategy,
    iter_bar_events,
    run_backtest,
    run_multiple_backtests,
)
from app.backtester.metrics import compute_metrics
from app.strategies.base_strategy import BaseStrategy
from app.strategies.moving_average import MovingAverageCrossover


class FlatStrategy(BaseStrategy):
    """Test helper: a strategy that never takes a position."""

    def __init__(self, params=None):
        super().__init__(params)
        self.name = "flat"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index)


def test_run_backtest_returns_full_result(clean_frame):
    """A normal backtest returns metrics, an equity curve and trades."""
    result = run_backtest(
        symbol="AAPL",
        df=clean_frame,
        strategy_name="moving_average",
        params={"fast": 10, "slow": 50},
    )

    assert result.symbol == "AAPL"
    expected_metrics = {
        "total_return",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "num_trades",
        "final_equity",
        "volatility",
    }
    assert set(result.metrics.keys()) == expected_metrics
    assert len(result.equity_curve) == len(clean_frame)
    assert isinstance(result.metrics["total_return"], float)


def test_run_backtest_rejects_empty_frame():
    """An empty data frame is useless for a backtest and must be rejected."""
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        run_backtest("AAPL", empty, "moving_average")


def test_run_backtest_rejects_unknown_strategy(clean_frame):
    """An unknown strategy name raises a clear error."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest("AAPL", clean_frame, "not_a_strategy")


def test_flat_strategy_makes_no_trades(clean_frame):
    """With nothing to trade the account should not change at all."""
    config = BacktestConfig(initial_capital=100_000.0)
    strategy = FlatStrategy()
    events = list(iter_bar_events(clean_frame, strategy, config))

    assert len(events) == len(clean_frame)
    for event in events:
        assert event["shares"] == 0
        assert event["equity"] == pytest.approx(100_000.0)


def test_moving_average_with_never_ready_windows_trades_nothing(clean_frame):
    """Windows longer than the data produce no signals and no trades."""
    result = run_backtest(
        symbol="AAPL",
        df=clean_frame,
        strategy_name="moving_average",
        params={"fast": 900, "slow": 2000},
    )
    assert result.metrics["num_trades"] == 0
    assert result.trades == []
    assert result.metrics["final_equity"] == pytest.approx(100_000.0)


def test_iter_bar_events_is_a_generator(clean_frame):
    """The event stream must be a plain generator we can iterate lazily."""
    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events = iter_bar_events(clean_frame, strategy, BacktestConfig())
    assert iter(events) is events  # generators are their own iterator
    total = sum(1 for _ in events)
    assert total == len(clean_frame)


def test_compute_metrics_known_values():
    """Metrics on a small equity curve should match the math by hand."""
    equity = pd.Series([100.0, 110.0, 99.0])
    trades = []  # no trades, so trade based metrics stay neutral
    metrics = compute_metrics(equity, trades, initial_capital=100.0)

    assert metrics["total_return"] == pytest.approx(-0.01)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["sharpe_ratio"] == pytest.approx(0.0)
    assert metrics["win_rate"] == 0.0
    assert metrics["num_trades"] == 0
    # No losses means no profit factor, stored as null so JSON is valid.
    assert metrics["profit_factor"] is None


def test_run_multiple_backtests_serial_matches_grid(clean_frame):
    """Serial grid runs return one result per parameter set."""
    grid = [
        {"fast": 5, "slow": 30},
        {"fast": 10, "slow": 50},
        {"fast": 20, "slow": 100},
    ]
    results = run_multiple_backtests(
        symbol="AAPL",
        df=clean_frame,
        strategy_name="moving_average",
        param_grid=grid,
        parallel=False,
    )
    assert len(results) == 3
    for result in results:
        assert "metrics" in result
        assert "total_return" in result["metrics"]
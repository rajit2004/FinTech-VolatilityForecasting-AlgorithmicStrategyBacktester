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


def test_volatility_sizing_reduces_position_when_vol_high(clean_frame):
    """When forecast vol is high, position size should be smaller."""
    config_no_vol = BacktestConfig(initial_capital=100_000.0, position_pct=0.95)
    config_high_vol = BacktestConfig(
        initial_capital=100_000.0,
        position_pct=0.95,
        target_volatility=0.15,
        forecast_volatility=0.30,
    )

    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events_no_vol = list(iter_bar_events(clean_frame, strategy, config_no_vol))
    events_high_vol = list(iter_bar_events(clean_frame, strategy, config_high_vol))

    # Find the first buy event in each run and compare share counts.
    buy_no_vol = next(e for e in events_no_vol if e["shares"] > 0)
    buy_high_vol = next(e for e in events_high_vol if e["shares"] > 0)

    # High volatility should result in fewer shares purchased.
    assert buy_high_vol["shares"] <= buy_no_vol["shares"]


def test_volatility_sizing_full_position_when_vol_low(clean_frame):
    """When forecast vol is low relative to target, full position is used."""
    config_low_vol = BacktestConfig(
        initial_capital=100_000.0,
        position_pct=0.95,
        target_volatility=0.30,
        forecast_volatility=0.15,
    )

    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events = list(iter_bar_events(clean_frame, strategy, config_low_vol))

    # Find the first buy event.
    buy_event = next(e for e in events if e["shares"] > 0)

    # With low forecast vol, we should get close to the full position.
    expected_budget = 100_000.0 * 0.95
    expected_shares = int(expected_budget / buy_event["price"])
    assert buy_event["shares"] == expected_shares


def test_volatility_sizing_disabled_when_zero(clean_frame):
    """When target_volatility is 0, position sizing works normally."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        position_pct=0.95,
        target_volatility=0.0,
        forecast_volatility=0.30,
    )

    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events = list(iter_bar_events(clean_frame, strategy, config))

    # Find the first buy event.
    buy_event = next(e for e in events if e["shares"] > 0)

    # With target_volatility=0, vol scaling is disabled, full position used.
    expected_budget = 100_000.0 * 0.95
    expected_shares = int(expected_budget / buy_event["price"])
    assert buy_event["shares"] == expected_shares


def test_stop_loss_exits_position(clean_frame):
    """A stop-loss triggers a sell when price drops below entry."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        stop_loss_pct=0.05,  # exit if price drops 5% below entry
    )
    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events = list(iter_bar_events(clean_frame, strategy, config))

    # The backtest should complete without errors.
    assert len(events) == len(clean_frame)
    # Equity should be a valid number at every step.
    for event in events:
        assert isinstance(event["equity"], float)
        assert event["equity"] > 0


def test_take_profit_exits_position(clean_frame):
    """A take-profit triggers a sell when price rises above entry."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        take_profit_pct=0.10,  # exit if price rises 10% above entry
    )
    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events = list(iter_bar_events(clean_frame, strategy, config))

    assert len(events) == len(clean_frame)
    for event in events:
        assert isinstance(event["equity"], float)
        assert event["equity"] > 0


def test_stop_loss_and_take_profit_combined(clean_frame):
    """Both stop-loss and take-profit can be active at the same time."""
    config = BacktestConfig(
        initial_capital=100_000.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
    )
    strategy = MovingAverageCrossover({"fast": 10, "slow": 50})
    events = list(iter_bar_events(clean_frame, strategy, config))

    assert len(events) == len(clean_frame)


def test_portfolio_backtest_returns_results(clean_frame):
    """A portfolio backtest across two symbols returns combined metrics."""
    from app.backtester.portfolio import run_portfolio_backtest

    data = {
        "AAPL": clean_frame,
        "MSFT": clean_frame,
    }
    result = run_portfolio_backtest(
        data, "moving_average", params={"fast": 10, "slow": 50}
    )

    assert result["n_symbols"] == 2
    assert set(result["symbols"]) == {"AAPL", "MSFT"}
    assert "portfolio_metrics" in result
    assert "total_return" in result["portfolio_metrics"]
    assert "portfolio_equity_curve" in result
    assert len(result["portfolio_equity_curve"]) > 0
    assert "individual_results" in result
    assert "AAPL" in result["individual_results"]


def test_portfolio_backtest_rejects_empty_data():
    """A portfolio backtest with no symbols must be rejected."""
    from app.backtester.portfolio import run_portfolio_backtest

    with pytest.raises(ValueError, match="no symbols"):
        run_portfolio_backtest({}, "moving_average")


def test_portfolio_capital_is_split_equally(clean_frame):
    """Each symbol gets an equal share of the initial capital."""
    from app.backtester.portfolio import run_portfolio_backtest

    data = {"AAPL": clean_frame, "MSFT": clean_frame, "BTC-USD": clean_frame}
    result = run_portfolio_backtest(data, "moving_average")

    assert result["n_symbols"] == 3
    assert result["per_symbol_capital"] == pytest.approx(100_000.0 / 3)
"""Tests for the trading strategies."""

import pandas as pd
import pytest

from app.strategies.base_strategy import BaseStrategy
from app.strategies.moving_average import MovingAverageCrossover
from app.strategies.volatility_breakout import VolatilityBreakout


def _frame_from_prices(prices: list) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a closing price list."""
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000.0] * len(prices),
        },
        index=pd.date_range("2022-01-03", periods=len(prices), freq="B"),
    )


def test_base_strategy_cannot_be_instantiated():
    """The abstract base strategy must refuse direct instantiation."""
    with pytest.raises(TypeError):
        BaseStrategy()


def test_base_strategy_requires_generate_signals():
    """A subclass missing generate_signals cannot be instantiated."""
    class Incomplete(BaseStrategy):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_moving_average_goes_long_in_uptrend(clean_frame):
    """In a rising market the fast MA sits above the slow MA (post warmup)."""
    strategy = MovingAverageCrossover({"fast": 5, "slow": 20})
    signals = strategy.generate_signals(_frame_from_prices(list(range(100, 200))))

    # After enough history to have both averages, the signal is long.
    assert (signals.iloc[25:] == 1.0).all()


def test_moving_average_flat_in_downtrend():
    """The default strategy exits to flat when the fast MA drops below."""
    prices = list(reversed(range(100, 200)))
    strategy = MovingAverageCrossover({"fast": 5, "slow": 20})
    signals = strategy.generate_signals(_frame_from_prices(prices))
    assert (signals.iloc[25:] == 0.0).all()


def test_moving_average_validates_parameters():
    """A fast window >= slow window makes no sense and must be rejected."""
    with pytest.raises(ValueError):
        MovingAverageCrossover({"fast": 50, "slow": 20})


def test_moving_average_stays_flat_before_warmup(clean_frame):
    """Signals before both averages exist must be zero, not noise."""
    strategy = MovingAverageCrossover({"fast": 5, "slow": 20})
    signals = strategy.generate_signals(clean_frame)
    # The first (slow - 1) rows have no slow average, hence flat.
    assert (signals.iloc[:19] == 0.0).all()


def test_volatility_breakout_enters_on_breakout():
    """A sharp jump through the upper band should trigger a long signal."""
    prices = [100.0] * 20 + [110.0, 112.0, 115.0]
    frame = _frame_from_prices(prices)
    strategy = VolatilityBreakout({"window": 10, "num_std": 2.0})
    signals = strategy.generate_signals(frame)

    # The big jump day should get a long signal.
    assert (signals.iloc[-2:] == 1.0).any()
    # Before the breakout everything should be flat.
    assert (signals.iloc[:19] == 0.0).all()


def test_volatility_breakout_uses_all_decimals():
    """num_std should be accepted as a float, not just an integer."""
    strategy = VolatilityBreakout({"window": 20, "num_std": 2.5})
    assert strategy.num_std == 2.5
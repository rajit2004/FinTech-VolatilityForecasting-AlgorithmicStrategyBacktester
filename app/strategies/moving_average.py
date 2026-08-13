"""Moving average crossover strategy.

The classic trend following idea: compute a fast and a slow moving
average of the closing price. When the fast average crosses above the
slow one the trend is turning up, so we go long. When it crosses back
below we go flat (or short).

This is one of the simplest strategies that still works in practice,
and it is a good benchmark against the volatility breakout strategy.
"""

from typing import Any, Dict, Optional

import pandas as pd

from app.strategies.base_strategy import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    """Buy when the fast MA crosses above the slow MA.

    Parameters:
        fast: window for the fast moving average, default 10.
        slow: window for the slow moving average, default 50.
        allow_short: when True the signal goes to -1 below the slow MA,
            when False it just drops to 0 (flat).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.name = "moving_average"
        self.fast = int(self.params.get("fast", 10))
        self.slow = int(self.params.get("slow", 50))
        self.allow_short = bool(self.params.get("allow_short", False))

        if self.fast >= self.slow:
            raise ValueError("fast window must be smaller than slow window")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return +1 when fast MA is above slow MA, else -1 or 0.

        Args:
            df: cleaned OHLCV data indexed by date.

        Returns:
            A Series of target positions aligned with df.index.
        """
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()

        # Position is the sign of the difference between the two averages.
        difference = fast_ma - slow_ma
        signals = difference.apply(lambda value: 1.0 if value > 0 else 0.0)

        if self.allow_short:
            signals = difference.apply(lambda value: 1.0 if value > 0 else -1.0)

        # Before the slow MA has enough data there is no valid signal,
        # so we stay flat instead of pretending we know the trend.
        signals[slow_ma.isna()] = 0.0
        return signals

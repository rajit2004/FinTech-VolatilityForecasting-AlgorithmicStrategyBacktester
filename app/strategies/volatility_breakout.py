"""Volatility breakout strategy.

The idea behind a breakout strategy is that markets that were trading
quietly often start a strong move when they finally break out of their
range. We use Bollinger bands built from the rolling mean and standard
deviation of the price:

- Enter long when the price closes above the upper band, the market is
  breaking upward with expanding volatility.
- Exit (go flat) when the price falls back below the middle band, the
  breakout failed or the move is over.

This strategy leans on volatility itself, which ties nicely into the
volatility forecasting part of the project.
"""

from typing import Any, Dict, Optional

import pandas as pd

from app.strategies.base_strategy import BaseStrategy


class VolatilityBreakout(BaseStrategy):
    """Breakout trading on Bollinger bands.

    Parameters:
        window: rolling window for the mean and standard deviation.
        num_std: how many standard deviations wide the bands are.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.name = "volatility_breakout"
        self.window = int(self.params.get("window", 20))
        self.num_std = float(self.params.get("num_std", 2.0))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return +1 above the upper band, 0 once price returns to the mean.

        Args:
            df: cleaned OHLCV data indexed by date.

        Returns:
            A Series of target positions aligned with df.index.
        """
        close = df["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()

        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        # Start flat everywhere, then mark the entries and exits.
        signals = pd.Series(0.0, index=df.index)

        # Enter long when the close punches through the upper band.
        signals[close > upper] = 1.0

        # Exit back to flat when the close drops below the middle line.
        signals[close < mid] = 0.0

        # The first `window` rows have no bands, so we stay flat there.
        signals[mid.isna()] = 0.0
        return signals

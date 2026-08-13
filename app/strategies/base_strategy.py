"""Abstract base class for all trading strategies.

Every strategy in this project does one job: look at a cleaned price
frame and decide what position we should hold on each day. The position
is +1 (long) or 0 (flat) or -1 (short). The backtest engine takes that
signal series and turns it into simulated trades and profits.

By making this an abstract base class we guarantee every strategy has a
generate_signals method with the same input and output, so the engine
can run any strategy without knowing what it does inside.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseStrategy(ABC):
    """Blueprint for a trading strategy.

    Attributes:
        name: human readable strategy name, used in the API and DB.
        params: the tuning parameters as a dict, stored with results.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "base"
        self.params: Dict[str, Any] = dict(params or {})

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Compute the desired position for every row of the frame.

        Args:
            df: cleaned OHLCV data indexed by date.

        Returns:
            A Series of -1, 0 or +1 aligned with df.index. Positive means
            long, zero means flat, negative means short.
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

    def get_params(self) -> Dict[str, Any]:
        """Return the strategy tuning parameters.

        Returns:
            The params dict, useful for comparing runs.
        """
        return self.params
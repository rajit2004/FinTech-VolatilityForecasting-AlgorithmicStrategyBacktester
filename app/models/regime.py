"""Market regime detection using Gaussian Mixture Models.

Markets operate in distinct states: calm (low volatility trending
sideways), normal (moderate volatility with directional moves), and
volatile (high volatility with large swings). Knowing which regime the
market is in changes how you should trade.

This module uses sklearn's GaussianMixture to cluster rolling
volatility observations into discrete regimes. It is a simpler
alternative to Hidden Markov Models that achieves similar results for
this use case and avoids an extra dependency.

The detected regime can be used to:
- Switch strategy behavior (trade less in volatile regimes)
- Adjust model parameters
- Filter signals (only trade in favorable regimes)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import ANNUALIZATION_FACTOR, VOLATILITY_WINDOW
from app.features.engineering import realized_volatility, log_returns
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Human readable names for the regimes, ordered from calm to volatile.
REGIME_LABELS = ["calm", "normal", "volatile"]


class RegimeDetector:
    """Detects market regimes from rolling volatility.

    Uses a Gaussian Mixture Model to cluster volatility observations
    into discrete regimes. The model is fit on the rolling volatility
    series and then used to label each day.

    Parameters:
        n_regimes: how many regimes to detect (default 3).
        volatility_window: rolling window for volatility (default 20).
    """

    def __init__(self, n_regimes: int = 3, volatility_window: int = VOLATILITY_WINDOW):
        self.n_regimes = n_regimes
        self.volatility_window = volatility_window
        self.model = None
        self._vol_means: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit the regime model on historical price data.

        Args:
            df: cleaned OHLCV data indexed by date.

        Returns:
            self so calls can be chained.
        """
        from sklearn.mixture import GaussianMixture

        returns = log_returns(df["close"])
        vol = realized_volatility(returns, window=self.volatility_window).dropna()

        if len(vol) < self.n_regimes * 10:
            raise ValueError(
                f"Need at least {self.n_regimes * 10} volatility "
                f"observations, got {len(vol)}"
            )

        # Reshape for sklearn: each sample is a single volatility value.
        X = vol.values.reshape(-1, 1)

        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=42,
            n_init=3,
        )
        self.model.fit(X)

        # Order regimes by their mean volatility so label 0 is always
        # the calmest and label n_regimes-1 is always the most volatile.
        means = self.model.means_.flatten()
        order = np.argsort(means)
        self._vol_means = means[order]

        # Remap component labels to match the sorted order.
        self._label_mapping = {
            old_label: new_label
            for new_label, old_label in enumerate(order)
        }

        logger.info(
            "Regime detector fitted: %d regimes with mean vols %s",
            self.n_regimes,
            [round(float(m), 4) for m in self._vol_means],
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Label each day with its market regime.

        Args:
            df: cleaned OHLCV data indexed by date.

        Returns:
            A Series of regime labels (0=calm, 1=normal, 2=volatile)
            aligned with df.index. Days without enough history for
            volatility are labeled as regime 1 (normal) as a safe default.
        """
        if self.model is None:
            raise RuntimeError("Detector has not been fitted yet, call fit() first")

        returns = log_returns(df["close"])
        vol = realized_volatility(returns, window=self.volatility_window)

        # Label every day, including those with NaN volatility.
        labels = pd.Series(1, index=df.index, dtype=int)  # default to normal

        valid_vol = vol.dropna()
        if len(valid_vol) > 0:
            X = valid_vol.values.reshape(-1, 1)
            raw_labels = self.model.predict(X)
            # Remap to ordered labels (0=calm ... n-1=volatile).
            mapped = [self._label_mapping[l] for l in raw_labels]
            labels.loc[valid_vol.index] = mapped

        return labels

    def predict_latest(self, df: pd.DataFrame) -> Dict[str, object]:
        """Return the regime for the most recent day.

        Args:
            df: cleaned OHLCV data indexed by date.

        Returns:
            A dict with regime_id, regime_name, and the volatility
            values of each regime for context.
        """
        labels = self.predict(df)
        latest_id = int(labels.iloc[-1])
        latest_name = REGIME_LABELS[latest_id] if latest_id < len(REGIME_LABELS) else str(latest_id)

        return {
            "regime_id": latest_id,
            "regime_name": latest_name,
            "regime_vol_means": [float(m) for m in self._vol_means]
            if self._vol_means is not None
            else [],
            "label_counts": labels.value_counts().to_dict(),
        }


def detect_regimes(df: pd.DataFrame, n_regimes: int = 3) -> Dict[str, object]:
    """Convenience function: fit and predict in one call.

    Args:
        df: cleaned OHLCV data indexed by date.
        n_regimes: number of regimes to detect.

    Returns:
        A dict with the latest regime info and the full label series.
    """
    detector = RegimeDetector(n_regimes=n_regimes)
    detector.fit(df)
    labels = detector.predict(df)
    result = detector.predict_latest(df)
    result["labels"] = labels
    return result

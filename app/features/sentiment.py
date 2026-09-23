"""Sentiment features for volatility forecasting.

Research shows that news and social media sentiment have measurable
predictive power for volatility, especially during market stress
(Saravanos & Kanavos 2025, Zhang et al. 2024). Sentiment features
capture information that price-derived features alone miss.

This module provides two approaches:

1. Simulated sentiment (works offline): derives a sentiment score
   from price action itself. Up days with high volume get positive
   sentiment, down days with high volume get negative sentiment.
   This is a reasonable proxy because price and sentiment are
   correlated, and it keeps the project fully reproducible.

2. Lexicon sentiment (for text data): a simple financial lexicon
   scorer that can be applied to news headlines or social media posts
   when you have them. It uses a small built-in lexicon of financial
   terms instead of requiring a large NLP model.

Both approaches produce a sentiment_score feature in the range [-1, 1]
that can be added to the feature set for the models.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from app.config import VOLATILITY_WINDOW
from app.utils.logger import get_logger

logger = get_logger(__name__)


# A small financial sentiment lexicon. In a production system you would
# use FinBERT or VADER, but this demonstrates the concept without
# requiring a large model download. Words are scored from -1 (very
# negative) to +1 (very positive).
_FINANCIAL_LEXICON = {
    "surge": 0.8, "rally": 0.7, "gain": 0.5, "rise": 0.4, "up": 0.3,
    "bullish": 0.7, "optimism": 0.6, "growth": 0.5, "profit": 0.6,
    "beat": 0.5, "strong": 0.5, "record": 0.4, "breakout": 0.5,
    "soar": 0.9, "jump": 0.6, "boost": 0.5, "recover": 0.4,
    "crash": -0.9, "plunge": -0.8, "drop": -0.6, "fall": -0.5,
    "down": -0.3, "bearish": -0.7, "fear": -0.7, "panic": -0.9,
    "loss": -0.6, "miss": -0.5, "weak": -0.5, "decline": -0.5,
    "slump": -0.7, "tumble": -0.8, "sink": -0.7, "wipeout": -0.9,
    "recession": -0.8, "inflation": -0.4, "crisis": -0.8,
    "risk": -0.3, "volatile": -0.2, "uncertainty": -0.4,
}


def simulate_sentiment(close: pd.Series, volume: pd.Series, window: int = VOLATILITY_WINDOW) -> pd.Series:
    """Generate a sentiment score from price action.

    This is the offline-friendly approach. The logic:
    - Up days contribute positive sentiment, down days negative.
    - High volume amplifies the sentiment (more people are trading).
    - A rolling window smooths the score so it does not jump around.

    The result is a score in [-1, 1] where positive means bullish
    sentiment and negative means bearish.

    Args:
        close: closing prices.
        volume: trading volumes.
        window: rolling window for smoothing.

    Returns:
        A Series of sentiment scores in [-1, 1].
    """
    # Daily return direction, scaled by relative volume.
    returns = close.pct_change()
    vol_ratio = volume / volume.rolling(window=window).mean()

    # Raw sentiment: return direction amplified by volume ratio.
    # When volume is high and price moves up, sentiment is strongly positive.
    raw = returns * vol_ratio.fillna(1.0)

    # Clip extreme values and smooth with a rolling mean.
    raw = raw.clip(-1.0, 1.0)
    smoothed = raw.rolling(window=window, min_periods=1).mean()

    # Scale to [-1, 1] using the observed range.
    max_abs = smoothed.abs().max()
    if max_abs > 0:
        smoothed = smoothed / max_abs

    return smoothed.fillna(0.0)


def score_text_sentiment(texts: List[str]) -> List[float]:
    """Score a list of texts using the financial lexicon.

    This is a demonstration of lexicon-based sentiment analysis.
    Each word in the text is looked up in the lexicon, and the scores
    are averaged. In a production system you would use FinBERT or
    VADER for much better accuracy.

    Args:
        texts: list of text strings (e.g. news headlines).

    Returns:
        A list of sentiment scores in [-1, 1], one per text.
    """
    scores = []
    for text in texts:
        words = str(text).lower().split()
        word_scores = [
            _FINANCIAL_LEXICON[word]
            for word in words
            if word in _FINANCIAL_LEXICON
        ]
        if word_scores:
            scores.append(float(np.mean(word_scores)))
        else:
            scores.append(0.0)  # no known words, neutral
    return scores


def add_sentiment_features(
    df: pd.DataFrame,
    window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Add sentiment features to the feature frame.

    Adds two columns:
    - sentiment: the simulated sentiment score in [-1, 1].
    - sentiment_momentum: rate of change of sentiment, captures
      whether sentiment is becoming more bullish or bearish.

    Args:
        df: cleaned OHLCV data indexed by date.
        window: rolling window for smoothing.

    Returns:
        A copy of df with the new sentiment columns.
    """
    result = df.copy()
    result["sentiment"] = simulate_sentiment(
        result["close"], result["volume"], window=window
    )
    result["sentiment_momentum"] = result["sentiment"].diff(periods=5).fillna(0.0)
    return result

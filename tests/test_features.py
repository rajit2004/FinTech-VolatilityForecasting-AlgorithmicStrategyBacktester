"""Tests for the feature engineering module."""

import math

import numpy as np
import pandas as pd
import pytest

from app.features.engineering import (
    build_features,
    ewma_volatility,
    future_realized_volatility,
    log_returns,
    make_lstm_sequences,
    prepare_for_model,
    realized_volatility,
    recursive_ema,
    rolling_mean_deque,
    rsi,
)


def test_log_returns_values():
    """Log returns for a known series should match the math by hand."""
    close = pd.Series([100.0, 105.0, 105.0])
    returns = log_returns(close)
    expected_first = np.log(105 / 100)
    returns_float = returns.dropna().tolist()

    assert returns.isna().iloc[0]  # first row has no previous price
    assert np.isclose(returns_float[0], expected_first)
    assert np.isclose(returns_float[1], 0.0)  # flat day gives zero return


def test_realized_volatility_small_for_stable_prices():
    """A steady price series should produce near zero realized volatility."""
    close = pd.Series([100.0] * 30)
    returns = log_returns(close)
    vol = realized_volatility(returns, window=20, annualize=False)
    assert vol.dropna().max() < 1e-6


def test_recursive_ema_matches_iterative():
    """The recursive EMA must agree with the iterative definition."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    alpha = 0.5

    result = recursive_ema(values, alpha)
    assert len(result) == len(values)

    # Compare against a manual loop with the same recurrence.
    manual = [values[0]]
    for value in values[1:]:
        manual.append(alpha * value + (1 - alpha) * manual[-1])
    assert np.allclose(result, manual)


def test_recursive_ema_rejects_long_series():
    """Too long a series must raise instead of blowing the recursion stack."""
    with pytest.raises(ValueError, match="recursive_ema"):
        recursive_ema(list(range(100000)), 0.5)


def test_rolling_mean_deque_matches_pandas():
    """The deque based rolling mean equals pandas' rolling mean."""
    values = pd.Series([float(i) for i in range(1, 11)])
    deque_based = rolling_mean_deque(values, window=3)
    pandas_based = values.rolling(3).mean()

    assert np.allclose(deque_based.dropna(), pandas_based.dropna())
    # First two values have no full window, so they are NaN.
    assert deque_based.isna().sum() == 2


def test_build_features_produces_expected_columns(clean_frame):
    """build_features adds all the engineered features plus the target."""
    horizon = 5
    features = build_features(clean_frame, horizon=horizon)

    expected = [
        "returns",
        "realized_vol",
        "ewma_vol",
        "rsi",
        "atr",
        "bollinger_position",
        "momentum",
        "volume_ratio",
        "day_of_week",
        "sentiment",
        "target",
    ]
    for column in expected:
        assert column in features.columns

    # The target is forward looking, so the last `horizon` rows are NaN.
    assert features["target"].iloc[-horizon:].isna().all()


def test_future_realized_volatility_is_shifted(clean_frame):
    """The target for day t must only depend on returns after day t."""
    from app.features.engineering import log_returns

    returns = log_returns(clean_frame["close"])
    target = future_realized_volatility(returns, horizon=5, annualize=False)
    known = target.dropna()
    assert len(known) < len(target)  # last rows must be NaN
    assert known.shape[0] <= len(returns) - 5


def test_prepare_for_model_drops_nan_rows(clean_frame):
    """prepare_for_model returns clean arrays with no missing values."""
    features = build_features(clean_frame, horizon=5)
    X, y, names = prepare_for_model(features)

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(names) == 10
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_make_lstm_sequences_shapes():
    """Sequence building produces the expected 3D window shape."""
    X = np.arange(60).reshape(10, 6).astype(float)
    y = np.arange(10).astype(float)
    X_seq, y_seq = make_lstm_sequences(X, y, sequence_length=3)

    assert X_seq.shape == (7, 3, 6)
    assert y_seq.shape == (7,)
    # The first sequence covers rows 0..2, so its target is y[2].
    assert y_seq[0] == y[2]


def test_make_lstm_sequences_needs_enough_rows():
    """Sequences cannot be built with fewer rows than the window size."""
    X = np.zeros((4, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        make_lstm_sequences(X, y, sequence_length=5)


def test_simulated_sentiment_range(clean_frame):
    """Simulated sentiment scores stay within [-1, 1]."""
    from app.features.sentiment import simulate_sentiment

    scores = simulate_sentiment(clean_frame["close"], clean_frame["volume"])

    assert len(scores) == len(clean_frame)
    assert scores.min() >= -1.0
    assert scores.max() <= 1.0


def test_score_text_sentiment():
    """Lexicon scorer gives positive scores for bullish text."""
    from app.features.sentiment import score_text_sentiment

    scores = score_text_sentiment([
        "Stocks rally on strong earnings",
        "Market crashes amid panic selling",
        "Weather is nice today",
    ])

    assert scores[0] > 0  # bullish words
    assert scores[1] < 0  # bearish words
    assert scores[2] == 0.0  # no known words, neutral


def test_add_sentiment_features(clean_frame):
    """add_sentiment_features adds sentiment columns to the frame."""
    from app.features.sentiment import add_sentiment_features

    result = add_sentiment_features(clean_frame)

    assert "sentiment" in result.columns
    assert "sentiment_momentum" in result.columns
    assert len(result) == len(clean_frame)
"""Tests for the volatility models and their evaluation."""

import numpy as np
import pandas as pd
import pytest

from app.features.engineering import build_features, make_lstm_sequences, prepare_for_model
from app.models.evaluator import (
    compare_models,
    evaluate_predictions,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from app.models.volatility_model import (
    LSTMVolatilityModel,
    RandomForestVolatilityModel,
    run_volatility_forecast,
    train_test_split_time,
)


def test_random_forest_predictions_match_input_length(clean_frame):
    """The random forest returns one prediction per input row."""
    features = build_features(clean_frame, horizon=5)
    X, y, _ = prepare_for_model(features)

    model = RandomForestVolatilityModel(n_estimators=50, max_depth=8)
    model.fit(X[:100], y[:100])
    predictions = model.predict(X[100:110])

    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))


def test_train_test_split_preserves_order():
    """The time split must keep the first part for training."""
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_fraction=0.2)

    assert len(X_train) == 80
    assert len(X_test) == 20
    # The test set comes after the training set in time.
    assert X_test[0][0] > X_train[-1][0]


def test_evaluator_metrics_known_values():
    """Metrics on a small hand computed example should match exactly."""
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([3.0, 4.0, 5.0])

    assert np.isclose(root_mean_squared_error(y_true, y_pred), np.sqrt(2 / 3))
    assert np.isclose(mean_absolute_error(y_true, y_pred), 2 / 3)
    assert np.isclose(r2_score(y_true, y_pred), 0.75)

    result = evaluate_predictions(y_true, y_pred)
    assert set(result.keys()) == {"rmse", "mae", "r2"}


def test_evaluator_raises_on_mismatched_lengths():
    """Evaluating must fail cleanly when arrays have different lengths."""
    with pytest.raises(ValueError):
        evaluate_predictions(np.array([1.0, 2.0]), np.array([1.0]))


def test_compare_models_picks_lowest_rmse():
    """compare_models should name the model with the smallest RMSE."""
    results = {
        "a": {"rmse": 0.5, "mae": 0.4, "r2": 0.8},
        "b": {"rmse": 0.2, "mae": 0.15, "r2": 0.9},
    }
    decision = compare_models(results)
    assert decision["best_model"] == "b"


def test_lstm_fit_and_predict_shapes(clean_frame):
    """The LSTM trains on sequences and predicts the same number of rows.

    This is deliberately tiny (one epoch, few units) so the test stays
    fast while still exercising the full Keras pipeline.
    """
    features = build_features(clean_frame, horizon=5)
    X, y, _ = prepare_for_model(features)
    X_seq, y_seq = make_lstm_sequences(X, y, sequence_length=15)

    model = LSTMVolatilityModel(sequence_length=15, epochs=1, batch_size=64, units=16)
    model.fit(X_seq[:150], y_seq[:150])
    predictions = model.predict(X_seq[150:160])

    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))


def test_run_volatility_forecast_end_to_end(clean_frame):
    """The full pipeline returns metrics and a next period forecast."""
    result = run_volatility_forecast(
        clean_frame, horizon=5, model_name="random_forest"
    )

    assert result["model_name"] == "random_forest"
    assert "rmse" in result["metrics"]
    assert isinstance(result["next_forecast"], float)
    assert result["next_forecast"] > 0
    assert len(result["test_predictions"]) == len(result["test_true"])


def test_run_volatility_forecast_rejects_unknown_model(clean_frame):
    """An unknown model name must raise a clear error."""
    with pytest.raises(ValueError, match="Unknown model"):
        run_volatility_forecast(clean_frame, model_name="not_a_model")
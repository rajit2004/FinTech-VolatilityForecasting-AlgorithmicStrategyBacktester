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
    GARCHVolatilityModel,
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


def test_garch_fit_and_predict():
    """The GARCH model trains on returns and forecasts volatility."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 500)

    model = GARCHVolatilityModel(p=1, q=1)
    model.fit(returns)
    forecast = model.predict(horizon=5)

    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))
    assert np.all(forecast > 0)


def test_garch_predict_before_fit_raises():
    """Calling predict before fit must raise a clear error."""
    model = GARCHVolatilityModel()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict()


def test_garch_next_convenience():
    """predict_next returns a single volatility value."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 500)

    model = GARCHVolatilityModel()
    model.fit(returns)
    value = model.predict_next()

    assert isinstance(value, float)
    assert value > 0


def test_run_volatility_forecast_garch(clean_frame):
    """The full pipeline works with the GARCH model."""
    result = run_volatility_forecast(
        clean_frame, horizon=5, model_name="garch"
    )

    assert result["model_name"] == "garch"
    assert "rmse" in result["metrics"]
    assert isinstance(result["next_forecast"], float)
    assert result["next_forecast"] > 0
    assert len(result["test_predictions"]) == len(result["test_true"])


def test_shap_explain_with_shap(clean_frame):
    """SHAP returns feature importance and per-row values."""
    from app.models.explainer import explain_with_shap

    features = build_features(clean_frame, horizon=5)
    X, y, feature_names = prepare_for_model(features)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)

    model = RandomForestVolatilityModel(n_estimators=50, max_depth=8)
    model.fit(X_train, y_train)

    result = explain_with_shap(model.model, X_test[:20], feature_names, X_train)

    assert "shap_values" in result
    assert "feature_importance" in result
    assert "base_value" in result
    assert len(result["feature_importance"]) == len(feature_names)
    # Top feature should have a higher mean abs SHAP than the bottom one.
    assert result["feature_importance"][0]["mean_abs_shap"] >= result["feature_importance"][-1]["mean_abs_shap"]


def test_shap_explain_forecast(clean_frame):
    """explain_forecast returns a summary and latest contributions."""
    from app.models.explainer import explain_forecast

    features = build_features(clean_frame, horizon=5)
    X, y, feature_names = prepare_for_model(features)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y)

    model = RandomForestVolatilityModel(n_estimators=50, max_depth=8)
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    result = explain_forecast(model, X_train, X_test, feature_names, test_predictions)

    assert "feature_importance" in result
    assert "latest_prediction_contributions" in result
    assert "summary" in result
    assert len(result["summary"]) > 0
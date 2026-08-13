"""Evaluation metrics for the volatility forecasting models.

These are the standard regression metrics used when comparing volatility
forecasts: root mean squared error, mean absolute error and R squared.
The comparisons are stored as dicts so they are easy to return through
the API and easy to test.
"""

from typing import Dict

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between predictions and truth.

    Squaring punishes big errors extra hard, which is what you want when
    a large missed volatility spike hurts a trading strategy.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between predictions and truth.

    Unlike RMSE this does not overweight big errors, it is just the
    average distance of the predictions from the real values.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R squared, how much of the variance the model explains.

    A value near 1 means the predictions track the true values closely,
    a value near 0 means the model is not better than predicting the mean.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1 - numerator / denominator)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics for one model run.

    Args:
        y_true: actual volatility values.
        y_pred: predicted volatility values.

    Returns:
        A dict with rmse, mae and r2 keys.
    """
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Predictions and true values must have the same length")

    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compare_models(results: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Pick the best model from a set of evaluation results.

    Args:
        results: mapping of model name to its metrics dict.

    Returns:
        A dict with the best_model name and a short summary.
    """
    best_name = None
    best_rmse = float("inf")
    for name, metrics in results.items():
        if metrics.get("rmse", float("inf")) < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name

    return {
        "best_model": best_name,
        "summary": f"Lowest RMSE was {best_rmse:.4f} from {best_name}",
    }

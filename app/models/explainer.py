"""Model explainability using SHAP.

SHAP (SHapley Additive exPlanations) is the standard tool for
explaining machine learning predictions. It tells you which features
pushed a prediction higher or lower, and by how much.

In financial forecasting, explainability matters for two reasons:
1. Trust: you need to know the model is not just memorizing noise.
2. Insight: knowing which features drive the forecast tells you
   something about the market regime.

This module provides a function that takes a trained random forest
model and the feature data, and returns per-row SHAP values and
overall feature importance rankings.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


def explain_with_shap(
    model,
    X: np.ndarray,
    feature_names: List[str],
    X_background: Optional[np.ndarray] = None,
    max_background_samples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values for a trained random forest model.

    SHAP values decompose each prediction into contributions from each
    feature. The sum of all feature contributions plus the base value
    equals the model's prediction for that row.

    Args:
        model: a trained scikit-learn RandomForestRegressor (or any
            tree-based model that supports the tree explainer).
        X: the data to explain, a 2D array of shape (n_samples, n_features).
        feature_names: list of feature names matching the columns of X.
        X_background: optional background dataset for the SHAP explainer.
            If None, a random subset of X is used.
        max_background_samples: maximum rows to use from X_background for
            the explainer background. More rows give better explanations
            but slower computation.

    Returns:
        A dict with:
        - shap_values: the per-row SHAP values as a 2D array.
        - feature_importance: a list of dicts with feature name, mean
          absolute SHAP value, and rank, sorted by importance.
        - base_value: the expected model output (the SHAP base value).
    """
    import shap

    # The background dataset is what SHAP uses to estimate the expected
    # model output. A small random subset is usually enough.
    if X_background is None:
        n_bg = min(max_background_samples, len(X))
        indices = np.random.choice(len(X), size=n_bg, replace=False)
        X_background = X[indices]

    # TreeExplainer is fast for tree-based models and exact for random
    # forests. It computes SHAP values in polynomial time instead of the
    # exponential time a naive approach would need.
    explainer = shap.TreeExplainer(model, data=X_background)
    shap_values = explainer.shap_values(X)

    # Feature importance: mean absolute SHAP value across all rows.
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs.sum()
    if total > 0:
        importance_fractions = mean_abs / total
    else:
        importance_fractions = np.zeros_like(mean_abs)

    ranked = sorted(
        zip(feature_names, mean_abs, importance_fractions),
        key=lambda item: item[1],
        reverse=True,
    )

    feature_importance = [
        {
            "feature": name,
            "mean_abs_shap": float(mean_val),
            "importance_pct": round(float(frac) * 100, 2),
            "rank": rank + 1,
        }
        for rank, (name, mean_val, frac) in enumerate(ranked)
    ]

    logger.info(
        "SHAP computed for %d samples, top feature: %s (%.1f%%)",
        len(X),
        feature_importance[0]["feature"],
        feature_importance[0]["importance_pct"],
    )

    return {
        "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        "feature_importance": feature_importance,
        "base_value": float(explainer.expected_value)
        if np.isscalar(explainer.expected_value)
        else float(np.mean(explainer.expected_value)),
    }


def explain_forecast(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    test_predictions: np.ndarray,
) -> Dict[str, Any]:
    """Explain the test set predictions using SHAP.

    This is the high-level function called by the API. It computes SHAP
    values for the test set and returns both the per-row explanations
    and the aggregate feature importance.

    Args:
        model: a trained RandomForestVolatilityModel.
        X_train: training features, used as background for SHAP.
        X_test: test features to explain.
        feature_names: list of feature names.
        test_predictions: the model's predictions on X_test.

    Returns:
        A dict with feature_importance, the most important features for
        the latest prediction, and a summary string.
    """
    # Use the underlying sklearn model for SHAP.
    raw_model = model.model

    # Subsample background for speed.
    n_bg = min(100, len(X_train))
    bg_indices = np.random.choice(len(X_train), size=n_bg, replace=False)
    X_background = X_train[bg_indices]

    result = explain_with_shap(raw_model, X_test, feature_names, X_background)

    # Highlight what mattered most for the most recent prediction.
    latest_shap = result["shap_values"][-1] if result["shap_values"] else []
    latest_contributions = []
    if latest_shap:
        for i, name in enumerate(feature_names):
            latest_contributions.append({
                "feature": name,
                "shap_value": float(latest_shap[i]),
            })
        latest_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    top_features = [f["feature"] for f in result["feature_importance"][:3]]
    result["latest_prediction_contributions"] = latest_contributions[:5]
    result["summary"] = (
        f"Top features driving predictions: {', '.join(top_features)}. "
        f"The model's average R-squared behavior is explained by these "
        f"features contributing the most to forecast changes."
    )

    return result

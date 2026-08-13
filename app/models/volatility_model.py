"""Volatility forecasting models.

Two models are implemented:

1. RandomForestVolatilityModel: a scikit-learn random forest regressor.
   It maps the engineered features directly to the future volatility.

2. LSTMVolatilityModel: a Keras LSTM that reads a window of past rows
   and predicts the volatility over the next horizon. We use Keras 3
   with the JAX backend because TensorFlow has no wheels for Python
   3.14 yet, but the Keras API is exactly the same.

Both models expose a common interface: fit(X, y) and predict(X), which
is all the rest of the app needs to use them interchangeably.
"""

import os
from typing import List, Optional, Tuple

# Keras 3 needs a backend chosen before it is imported. We use JAX
# because TensorFlow has no wheels for Python 3.14. setdefault keeps any
# explicit choice from an earlier import intact.
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

from app.config import FORECAST_HORIZON_DAYS, LSTM_EPOCHS, LSTM_SEQUENCE_LENGTH
from app.features.engineering import (
    build_features,
    make_lstm_sequences,
    prepare_for_model,
)
from app.models.evaluator import evaluate_predictions
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestVolatilityModel:
    """Random forest regressor that predicts future volatility.

    Random forests are great baseline models for tabular data. They do
    not need scaled features and they handle non linear relationships,
    which volatility definitely has.
    """

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 10, random_state: int = 42):
        from sklearn.ensemble import RandomForestRegressor

        self.name = "random_forest"
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestVolatilityModel":
        """Train the forest on the feature matrix.

        Args:
            X: 2D array of features.
            y: 1D array of target volatility values.

        Returns:
            self so calls can be chained.
        """
        self.model.fit(X, y)
        logger.info("Random forest trained with %d trees", self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict volatility for new feature rows.

        Args:
            X: 2D array of features.

        Returns:
            Array of predicted volatility values.
        """
        return np.asarray(self.model.predict(X))


class LSTMVolatilityModel:
    """A Keras LSTM for time series volatility forecasting.

    The LSTM expects sequences: each sample is a (sequence_length, n_features)
    window of past rows. Volatility has a strong memory effect, so the
    recurrent structure is a natural fit. Features and targets are scaled
    to 0..1 before training because neural nets train much better on
    normalized inputs, and scaled back on predict.
    """

    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = 32,
        units: int = 64,
    ):
        self.name = "lstm"
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.model = None
        self._feature_scaler = None
        self._target_scaler = None

    def _build(self, n_features: int):
        """Create the Keras Sequential model.

        Structure: an LSTM layer over the window, then a dense layer that
        shrinks the output to a single volatility number.
        """
        from keras.layers import Dense, LSTM
        from keras.models import Sequential
        from keras import Input

        self.model = Sequential(
            [
                Input(shape=(self.sequence_length, n_features)),
                LSTM(self.units, return_sequences=False),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> "LSTMVolatilityModel":
        """Train the LSTM on pre built sequences.

        Args:
            X_seq: 3D array shaped (samples, sequence_length, features).
            y: 1D array of target values.

        Returns:
            self so calls can be chained.
        """
        from sklearn.preprocessing import MinMaxScaler

        # Scale features along the feature axis, the shape is flattened to
        # a 2D matrix for the scaler then reshaped back.
        flat = X_seq.reshape(-1, X_seq.shape[2])
        self._feature_scaler = MinMaxScaler().fit(flat)
        scaled_flat = self._feature_scaler.transform(flat)
        scaled_seq = scaled_flat.reshape(X_seq.shape)

        self._target_scaler = MinMaxScaler().fit(y.reshape(-1, 1))
        scaled_y = self._target_scaler.transform(y.reshape(-1, 1)).ravel()

        self._build(X_seq.shape[2])
        self.model.fit(
            scaled_seq.astype("float32"),
            scaled_y.astype("float32"),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        logger.info(
            "LSTM trained with %d epochs, seq length %d",
            self.epochs,
            self.sequence_length,
        )
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict volatility from sequences.

        Args:
            X_seq: 3D array shaped (samples, sequence_length, features).

        Returns:
            Array of predicted volatility values in the original scale.
        """
        flat = X_seq.reshape(-1, X_seq.shape[2])
        scaled = self._feature_scaler.transform(flat).reshape(X_seq.shape)
        raw = self.model.predict(scaled.astype("float32"), verbose=0)
        return self._target_scaler.inverse_transform(raw).ravel()


def train_test_split_time(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data in time order, not randomly.

    Time series must never be shuffled before a split, the model has to
    learn from the past and be tested on the future. The last
    test_fraction of the rows becomes the test set.

    Args:
        X: feature array.
        y: target array.
        test_fraction: fraction of rows to hold out for testing.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    split = int(len(X) * (1 - test_fraction))
    return X[:split], X[split:], y[:split], y[split:]


def run_volatility_forecast(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON_DAYS,
    model_name: str = "random_forest",
    sequence_length: Optional[int] = None,
    epochs: Optional[int] = None,
) -> dict:
    """Run the whole forecasting pipeline for one symbol.

    Steps: build features, split in time order, train the chosen model,
    evaluate on the held out test set, and produce a one step ahead
    forecast for the period after the last known day.

    Args:
        df: cleaned OHLCV data indexed by date.
        horizon: forecast horizon in days.
        model_name: 'random_forest' or 'lstm'.
        sequence_length: LSTM window, only used for the lstm model.
        epochs: LSTM epochs, only used for the lstm model.

    Returns:
        A dict with predictions, true values, metrics and the next
        forecast, ready to store in the database and return via the API.
    """
    if model_name not in ("random_forest", "lstm"):
        raise ValueError(f"Unknown model: {model_name}")

    frame = build_features(df, horizon)
    X, y, feature_names = prepare_for_model(frame)
    logger.info("Prepared %d samples with %d features", len(X), len(feature_names))

    # The rows that survive NaN removal, their index gives us the dates.
    clean = frame.dropna(subset=feature_names + ["target"])
    dates = clean.index

    if model_name == "random_forest":
        X_train, X_test, y_train, y_test = train_test_split_time(X, y)
        model = RandomForestVolatilityModel()
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        # One step ahead: predict from the very last row we have features for.
        next_forecast = float(model.predict(X[-1:])[0])
    else:
        seq_len = sequence_length or LSTM_SEQUENCE_LENGTH
        epochs = epochs or LSTM_EPOCHS

        X_seq, y_seq = make_lstm_sequences(X, y, seq_len)
        X_train, X_test, y_train, y_test = train_test_split_time(X_seq, y_seq)
        model = LSTMVolatilityModel(sequence_length=seq_len, epochs=epochs)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        # The final sequence predicts the period after the last known day.
        next_forecast = float(model.predict(X_seq[-1:])[0])

    metrics = evaluate_predictions(y_test, test_predictions)

    # Align the test results with their real calendar dates. For the LSTM
    # each sequence ends at a known row, so its date is offset by the
    # sequence length. The test set is the tail end of the sample.
    split = int(len(X) * (1 - 0.2))
    if model_name == "random_forest":
        test_dates = [d.date().isoformat() for d in dates[split:]]
    else:
        test_dates = [d.date().isoformat() for d in dates[split + seq_len - 1:]]

    return {
        "model_name": model_name,
        "horizon": horizon,
        "feature_names": feature_names,
        "test_predictions": test_predictions.tolist(),
        "test_true": y_test.tolist(),
        "test_dates": test_dates,
        "metrics": metrics,
        "next_forecast": next_forecast,
    }

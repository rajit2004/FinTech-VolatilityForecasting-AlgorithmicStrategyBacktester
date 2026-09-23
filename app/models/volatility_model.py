"""Volatility forecasting models.

Three models are implemented:

1. RandomForestVolatilityModel: a scikit-learn random forest regressor.
   It maps the engineered features directly to the future volatility.

2. LSTMVolatilityModel: a Keras LSTM that reads a window of past rows
   and predicts the volatility over the next horizon. We use Keras 3
   with the JAX backend because TensorFlow has no wheels for Python
   3.14 yet, but the Keras API is exactly the same.

3. GARCHVolatilityModel: a GARCH(1,1) baseline from the arch library.
   GARCH is the classic statistical model for volatility. It models
   the conditional variance of returns directly, without any feature
   engineering. This is the benchmark that ML models should beat to
   prove they are adding value.

The random forest and LSTM share a common interface (fit on features,
predict from features). The GARCH model works differently: it fits on
raw log returns and forecasts conditional variance. The
run_volatility_forecast function hides this difference behind a
unified call.
"""

import os
from typing import List, Optional, Tuple

# Keras 3 needs a backend chosen before it is imported. We use JAX
# because TensorFlow has no wheels for Python 3.14. setdefault keeps any
# explicit choice from an earlier import intact.
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

from app.config import (
    ANNUALIZATION_FACTOR,
    FORECAST_HORIZON_DAYS,
    LSTM_EPOCHS,
    LSTM_SEQUENCE_LENGTH,
)
from app.features.engineering import (
    FEATURE_COLUMNS,
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


class TransformerVolatilityModel:
    """Attention-based transformer for volatility forecasting.

    Transformers use self-attention to weigh the importance of every
    time step against every other time step. Unlike the LSTM, which
    processes steps sequentially and can forget early information,
    attention lets the model look directly at any past observation.

    Recent research (2024-2025) shows transformers outperform LSTMs
    on financial time series because volatility patterns can span
    long ranges that recurrent networks struggle to capture.

    The architecture: multi-head attention over the input sequence,
    followed by global average pooling and dense layers. We keep it
    deliberately small to train quickly on a laptop.
    """

    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = 32,
        num_heads: int = 4,
        key_dim: int = 32,
    ):
        self.name = "transformer"
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.model = None
        self._feature_scaler = None
        self._target_scaler = None

    def _build(self, n_features: int):
        """Create the Keras model with multi-head attention.

        Uses the Functional API instead of Sequential because
        MultiHeadAttention takes both query and value as inputs, which
        Sequential cannot handle. Structure: project features to a
        higher dimension, apply multi-head self-attention over the
        time axis, pool the result, then dense layers that shrink to
        a single volatility number.
        """
        from keras import Input, Model
        from keras.layers import (
            Dense,
            GlobalAveragePooling1D,
            LayerNormalization,
            MultiHeadAttention,
            Dropout,
        )

        inputs = Input(shape=(self.sequence_length, n_features))
        # Project features to a space suitable for attention.
        x = Dense(self.key_dim * self.num_heads, activation="relu")(inputs)
        # Multi-head attention: each head learns a different aspect of
        # which time steps matter most. Query and value both come from x.
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
        )(query=x, value=x)
        # Normalize to keep training stable.
        x = LayerNormalization(epsilon=1e-6)(x)
        # Pool the sequence into a single vector.
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    def fit(self, X_seq: np.ndarray, y: np.ndarray) -> "TransformerVolatilityModel":
        """Train the transformer on pre built sequences.

        Args:
            X_seq: 3D array shaped (samples, sequence_length, features).
            y: 1D array of target values.

        Returns:
            self so calls can be chained.
        """
        from sklearn.preprocessing import MinMaxScaler

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
            "Transformer trained with %d epochs, %d heads, seq length %d",
            self.epochs,
            self.num_heads,
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


class GARCHVolatilityModel:
    """GARCH(1,1) baseline for volatility forecasting.

    GARCH models are the gold standard statistical baseline in volatility
    research. Every serious ML paper since 2020 compares against GARCH.
    The model fits the conditional variance of log returns using an
    autoregressive structure: today's variance depends on yesterday's
    squared return and yesterday's variance.

    Unlike the ML models, GARCH does not use engineered features. It
    works directly on the return series, which makes it a pure
    statistical baseline. If our ML models cannot beat GARCH, they are
    not adding real value.
    """

    def __init__(self, p: int = 1, q: int = 1):
        """Set up the GARCH order.

        Args:
            p: number of lagged variances (the GARCH part).
            q: number of lagged squared returns (the ARCH part).
        """
        self.name = "garch"
        self.p = p
        self.q = q
        self.model = None
        self._fitted = None

    def fit(self, returns: np.ndarray) -> "GARCHVolatilityModel":
        """Fit the GARCH model on a series of log returns.

        Args:
            returns: 1D array of daily log returns (not prices, not
                features). The arch library expects raw returns.

        Returns:
            self so calls can be chained.
        """
        from arch import arch_model

        # Scale returns to percentage so the optimizer has an easier time.
        # GARCH models are sensitive to the scale of the input, and daily
        # returns are typically in the range of a few percent, so scaling
        # to percentage points helps numerical stability.
        scaled = returns * 100.0
        self.model = arch_model(
            scaled,
            vol="Garch",
            p=self.p,
            q=self.q,
            mean="Constant",
            dist="Normal",
        )
        self._fitted = self.model.fit(disp="off", show_warning=False)
        logger.info(
            "GARCH(%d,%d) fitted on %d returns", self.p, self.q, len(returns)
        )
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Forecast conditional volatility for the next horizon days.

        Args:
            horizon: how many days ahead to forecast.

        Returns:
            Array of predicted annualized volatility values. The arch
            library returns variance, so we take the square root and
            convert back from percentage scale.
        """
        if self._fitted is None:
            raise RuntimeError("Model has not been fitted yet, call fit() first")

        # forecast() returns an object with .variance and .mean DataFrames.
        # We need the conditional variance (not residual variance).
        fcst = self._fitted.forecast(horizon=horizon)
        variance = fcst.variance.iloc[-1].values

        # Convert from percentage-squared variance to annualized volatility.
        # sqrt(variance) gives daily vol in percentage, multiply by sqrt(252)
        # to annualize, then divide by 100 to get back to decimal scale.
        daily_vol = np.sqrt(variance) / 100.0
        annualized = daily_vol * np.sqrt(ANNUALIZATION_FACTOR)
        return annualized

    def predict_next(self) -> float:
        """Convenience method: forecast volatility for the next single day.

        Returns:
            A single annualized volatility value.
        """
        return float(self.predict(horizon=1)[0])


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


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_splits: int = 5,
    min_train_fraction: float = 0.4,
) -> dict:
    """Walk-forward validation for time series models.

    Instead of a single train/test split, this trains the model on a
    rolling window and tests on the next chunk, then slides the window
    forward and repeats. This gives a much more honest picture of how
    the model performs across different market conditions.

    The approach: start with the first min_train_fraction of data as
    the initial training set. Split the remaining data into n_splits
    equal chunks. For each chunk, train on everything before it and
    test on that chunk. Collect predictions from all chunks.

    Args:
        X: feature array (2D for RF, 3D for sequence models).
        y: target array.
        model_factory: a callable that returns a fresh model instance.
            Must have fit(X, y) and predict(X) methods.
        n_splits: number of test chunks to evaluate on.
        min_train_fraction: minimum fraction of data used for the
            first training set.

    Returns:
        A dict with all_predictions, all_true, all_dates indices, and
        per fold metrics plus aggregate metrics.
    """
    n = len(X)
    initial_split = int(n * min_train_fraction)
    remaining = n - initial_split
    if remaining < n_splits:
        raise ValueError(
            f"Not enough data for {n_splits} walk-forward splits, "
            f"got {remaining} rows after initial training window"
        )

    chunk_size = remaining // n_splits

    all_predictions = []
    all_true = []
    fold_metrics = []

    for fold in range(n_splits):
        train_end = initial_split + fold * chunk_size
        test_end = train_end + chunk_size if fold < n_splits - 1 else n

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        if len(X_test) == 0:
            continue

        model = model_factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        all_predictions.extend(predictions.tolist())
        all_true.extend(y_test.tolist())

        fold_metrics.append(evaluate_predictions(y_test, predictions))
        logger.info(
            "Walk-forward fold %d/%d: RMSE=%.4f, R2=%.4f",
            fold + 1, n_splits,
            fold_metrics[-1]["rmse"],
            fold_metrics[-1]["r2"],
        )

    aggregate = evaluate_predictions(np.array(all_true), np.array(all_predictions))

    return {
        "all_predictions": all_predictions,
        "all_true": all_true,
        "test_start_index": initial_split,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate,
    }


def run_volatility_forecast(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON_DAYS,
    model_name: str = "random_forest",
    sequence_length: Optional[int] = None,
    epochs: Optional[int] = None,
    validation: str = "holdout",
    n_walk_forward_folds: int = 5,
) -> dict:
    """Run the whole forecasting pipeline for one symbol.

    Steps: build features, split in time order, train the chosen model,
    evaluate on the held out test set, and produce a one step ahead
    forecast for the period after the last known day.

    Args:
        df: cleaned OHLCV data indexed by date.
        horizon: forecast horizon in days.
        model_name: 'random_forest', 'lstm' or 'garch'.
        sequence_length: LSTM window, only used for the lstm model.
        epochs: LSTM epochs, only used for the lstm model.
        validation: 'holdout' for a single 80/20 split, or
            'walk_forward' for rolling window validation.
        n_walk_forward_folds: number of walk-forward folds when
            validation is 'walk_forward'.

    Returns:
        A dict with predictions, true values, metrics and the next
        forecast, ready to store in the database and return via the API.
    """
    valid_models = ("random_forest", "lstm", "garch", "transformer")
    if model_name not in valid_models:
        raise ValueError(f"Unknown model: {model_name}")
    if validation not in ("holdout", "walk_forward"):
        raise ValueError(f"Unknown validation mode: {validation}")

    frame = build_features(df, horizon)
    clean = frame.dropna(subset=FEATURE_COLUMNS + ["target"])
    dates = clean.index

    # GARCH works on raw returns, not engineered features. It is a
    # statistical baseline that models the conditional variance directly.
    if model_name == "garch":
        from app.features.engineering import log_returns

        returns = log_returns(df["close"]).dropna()
        returns_array = returns.values

        # 80/20 time split, same as the other models.
        split = int(len(returns_array) * 0.8)
        train_returns = returns_array[:split]
        test_returns = returns_array[split:]

        model = GARCHVolatilityModel()
        model.fit(train_returns)

        # Predict one-step-ahead volatility for each test day. In
        # production you would refit periodically, but for comparison
        # we refit once and roll forward with the same parameters.
        test_predictions = np.array([
            model.predict_next() for _ in range(len(test_returns))
        ])

        # The true volatility for comparison: realized vol over the same
        # horizon as the ML models, computed from the test returns.
        true_vol = np.array([
            np.std(test_returns[max(0, i - horizon + 1): i + 1])
            * np.sqrt(ANNUALIZATION_FACTOR)
            for i in range(len(test_returns))
        ])

        # The next period forecast uses all available data.
        model.fit(returns_array)
        next_forecast = model.predict_next()

        # Align test dates with the return series indices.
        test_dates = [d.date().isoformat() for d in dates[split:]]

        metrics = evaluate_predictions(true_vol, test_predictions)
        return {
            "model_name": model_name,
            "horizon": horizon,
            "feature_names": [],
            "validation": validation,
            "test_predictions": test_predictions.tolist(),
            "test_true": true_vol.tolist(),
            "test_dates": test_dates,
            "metrics": metrics,
            "next_forecast": next_forecast,
        }

    # For the ML models we need engineered features.
    X, y, feature_names = prepare_for_model(frame)
    logger.info("Prepared %d samples with %d features", len(X), len(feature_names))

    # Walk-forward validation: train on rolling windows, test on next chunk.
    if validation == "walk_forward":
        if model_name == "random_forest":
            factory = lambda: RandomForestVolatilityModel()  # noqa: E731
            wf = walk_forward_validate(X, y, factory, n_splits=n_walk_forward_folds)
        else:
            seq_len = sequence_length or LSTM_SEQUENCE_LENGTH
            epochs_val = epochs or LSTM_EPOCHS
            X_seq, y_seq = make_lstm_sequences(X, y, seq_len)
            if model_name == "transformer":
                factory = lambda: TransformerVolatilityModel(  # noqa: E731
                    sequence_length=seq_len, epochs=epochs_val
                )
            else:
                factory = lambda: LSTMVolatilityModel(  # noqa: E731
                    sequence_length=seq_len, epochs=epochs_val
                )
            wf = walk_forward_validate(
                X_seq, y_seq, factory, n_splits=n_walk_forward_folds
            )

        # Final model trained on all data for the next period forecast.
        if model_name == "random_forest":
            final_model = RandomForestVolatilityModel()
            final_model.fit(X, y)
            next_forecast = float(final_model.predict(X[-1:])[0])
            split_idx = wf["test_start_index"]
            test_dates = [d.date().isoformat() for d in dates[split_idx:]]
        else:
            if model_name == "transformer":
                final_model = TransformerVolatilityModel(
                    sequence_length=seq_len, epochs=epochs_val
                )
            else:
                final_model = LSTMVolatilityModel(
                    sequence_length=seq_len, epochs=epochs_val
                )
            final_model.fit(X_seq, y_seq)
            next_forecast = float(final_model.predict(X_seq[-1:])[0])
            split_seq = wf["test_start_index"]
            test_dates = [
                d.date().isoformat()
                for d in dates[split_seq + seq_len - 1 : len(dates) - 1]
            ]

        return {
            "model_name": model_name,
            "horizon": horizon,
            "feature_names": feature_names,
            "validation": validation,
            "test_predictions": wf["all_predictions"],
            "test_true": wf["all_true"],
            "test_dates": test_dates[: len(wf["all_predictions"])],
            "metrics": wf["aggregate_metrics"],
            "fold_metrics": wf["fold_metrics"],
            "next_forecast": next_forecast,
        }

    # Standard holdout validation: single 80/20 time split.
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

        if model_name == "transformer":
            model = TransformerVolatilityModel(sequence_length=seq_len, epochs=epochs)
        else:
            model = LSTMVolatilityModel(sequence_length=seq_len, epochs=epochs)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        # The final sequence predicts the period after the last known day.
        next_forecast = float(model.predict(X_seq[-1:])[0])

    metrics = evaluate_predictions(y_test, test_predictions)

    # Align the test results with their real calendar dates. For the LSTM
    # each sequence ends at a known row, so its date is offset by the
    # sequence length, and the last row is never a sequence target. The
    # split is computed on the rows that actually reached the model, which
    # differs between the two model types.
    if model_name == "random_forest":
        split = int(len(X) * (1 - 0.2))
        test_dates = [d.date().isoformat() for d in dates[split:]]
    else:
        split_seq = int(len(X_seq) * (1 - 0.2))
        test_dates = [
            d.date().isoformat() for d in dates[split_seq + seq_len - 1 : len(dates) - 1]
        ]

    return {
        "model_name": model_name,
        "horizon": horizon,
        "feature_names": feature_names,
        "validation": validation,
        "test_predictions": test_predictions.tolist(),
        "test_true": y_test.tolist(),
        "test_dates": test_dates,
        "metrics": metrics,
        "next_forecast": next_forecast,
    }

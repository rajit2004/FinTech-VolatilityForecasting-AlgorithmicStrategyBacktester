"""Feature engineering for the volatility forecasting models.

Raw OHLCV prices are not directly useful for a model. This module turns
them into features the models can learn from: simple returns, realized
volatility (our target), EWMA volatility, RSI, ATR, Bollinger position,
momentum and calendar features.

It also demonstrates two data structure concepts from the course:
- deques for rolling windows (rolling_mean_deque).
- recursion, used in recursive_ema to compute an exponential moving
  average through a self calling function instead of a loop.
"""

import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import ANNUALIZATION_FACTOR, VOLATILITY_WINDOW
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Recursion needs to stay under Python's default recursion limit. Daily
# histories are usually around 1000 rows, so we only use the recursive
# version for short series and switch to pandas for the long ones. This
# keeps the demonstration safe and fast.
RECURSION_SAFE_LIMIT = 700

# Columns that are numeric features, built by build_features().
FEATURE_COLUMNS = [
    "returns",
    "realized_vol",
    "ewma_vol",
    "rsi",
    "atr",
    "bollinger_position",
    "momentum",
    "volume_ratio",
    "day_of_week",
]


def log_returns(close: pd.Series) -> pd.Series:
    """Compute daily log returns from closing prices.

    Log returns are nice for volatility work because they are additive
    over time, which is not true for simple percentage returns.
    """
    return np.log(close / close.shift(1))


def realized_volatility(
    returns: pd.Series,
    window: int = VOLATILITY_WINDOW,
    annualize: bool = True,
) -> pd.Series:
    """Rolling realized volatility of returns.

    Realized volatility is just the rolling standard deviation of the
    returns, scaled up to a yearly number so different periods can be
    compared. We multiply by the square root of the number of trading
    days per year.

    Args:
        returns: daily log returns.
        window: rolling window size in days.
        annualize: multiply by sqrt(252) or not.

    Returns:
        A Series of volatility values.
    """
    rolling = returns.rolling(window=window).std()
    if annualize:
        rolling = rolling * math.sqrt(ANNUALIZATION_FACTOR)
    return rolling


def future_realized_volatility(
    returns: pd.Series,
    horizon: int,
    annualize: bool = True,
) -> pd.Series:
    """Volatility over the next `horizon` days, used as the target.

    For day t the target is the realized volatility of returns from
    t+1 up to t+horizon. Because it looks forward it has NaN values at
    the end of the series, those rows are dropped before training.

    Args:
        returns: daily log returns.
        horizon: how many days ahead the volatility covers.
        annualize: scale to a yearly number or not.

    Returns:
        A Series with the forward looking volatility.
    """
    # Rolling std centered at the future end of the window, then shift so
    # the value at day t only uses days strictly after t.
    forward = returns.rolling(window=horizon).std().shift(-horizon)
    if annualize:
        forward = forward * math.sqrt(ANNUALIZATION_FACTOR)
    return forward


def recursive_ema(values: List[float], alpha: float) -> List[float]:
    """Exponential moving average computed with recursion.

    The EMA recurrence is: ema[t] = alpha * value[t] + (1 - alpha) * ema[t-1].
    We express this as a self calling function: the EMA of a list is the
    combination of the last value with the EMA of everything before it.
    The base case is a list of one element, which returns that element
    unchanged.

    Recursion is a nice fit here because the formula is naturally
    recursive, one step depends on the previous step. We guard the depth
    so very long series never hit Python's recursion limit.

    Args:
        values: the raw values to smooth.
        alpha: smoothing factor between 0 and 1.

    Returns:
        A list of EMA values, same length as input.
    """
    if len(values) > RECURSION_SAFE_LIMIT:
        raise ValueError(
            "recursive_ema only supports series up to "
            f"{RECURSION_SAFE_LIMIT} values, use ewma_volatility for longer ones"
        )

    def step(items: List[float]) -> List[float]:
        # Base case: one value has no history, so the EMA is itself.
        if len(items) == 1:
            return [items[0]]
        # Recursive case: compute the EMA of the prefix, then blend the
        # last value with the previous EMA to get this step's value.
        previous = step(items[:-1])
        next_value = alpha * items[-1] + (1 - alpha) * previous[-1]
        previous.append(next_value)
        return previous

    return step(list(values))


def ewma_volatility(returns: pd.Series, span: int = 20) -> pd.Series:
    """Exponentially weighted volatility of returns.

    A fast moving average reacts to new information quickly, which fits
    how volatility clustering works in markets. For short series we use
    the recursive version to show recursion in action, otherwise pandas
    does the same math much faster.

    Args:
        returns: daily log returns.
        span: the EWMA span, roughly the effective window in days.

    Returns:
        A Series of annualized EWMA volatility.
    """
    alpha = 2.0 / (span + 1.0)
    clean = returns.dropna()

    if len(clean) <= RECURSION_SAFE_LIMIT:
        # variance, not std, because the recurrence works on squared moves
        squared = (clean.values ** 2).tolist()
        ema_of_squares = recursive_ema(squared, alpha)
        variance = pd.Series(ema_of_squares, index=clean.index)
        vol = np.sqrt(variance)
    else:
        vol = clean.pow(2).ewm(span=span, adjust=False).mean().apply(np.sqrt)

    return vol * math.sqrt(ANNUALIZATION_FACTOR)


def rolling_mean_deque(values: pd.Series, window: int = 20) -> pd.Series:
    """Rolling mean computed with a deque instead of pandas rolling.

    A deque with maxlen keeps only the last `window` values automatically.
    This is the structure you would use in a live trading feed where new
    bars keep arriving and old ones should fall out of the window. It is
    used here for the momentum feature to demonstrate the idea.

    Args:
        values: the series to smooth.
        window: rolling window size.

    Returns:
        A Series of rolling means, NaN until enough data has arrived.
    """
    buffer: deque = deque(maxlen=window)
    results = []
    for value in values.tolist():
        buffer.append(value)
        if len(buffer) == window:
            results.append(sum(buffer) / window)
        else:
            results.append(np.nan)
    return pd.Series(results, index=values.index)


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change of price over the window, a simple trend signal."""
    return close.pct_change(periods=window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, measures how overbought or oversold price is.

    Standard 0 to 100 oscillator. Above 70 is often seen as overbought,
    below 30 as oversold. We use it as a model feature, not a signal.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range, the biggest single day move including overnight gaps.

    The previous close matters, so the very first row has no true range.
    """
    previous_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - previous_close).abs()
    low_prev_close = (df["low"] - previous_close).abs()
    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range, a common measure of price volatility in dollars."""
    return true_range(df).rolling(window=window).mean()


def bollinger_position(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Where the price sits inside its Bollinger bands, from -1 to 1.

    The bands are a rolling mean plus and minus a multiple of the rolling
    standard deviation. This feature tells the model how stretched the
    price is relative to its recent range.
    """
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    spread = (upper - lower).replace(0, np.nan)
    return ((close - lower) / spread).fillna(0.0)


def build_features(
    df: pd.DataFrame,
    horizon: int,
    volatility_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Build the full feature set plus the target column.

    Args:
        df: cleaned OHLCV data indexed by date.
        horizon: forecast horizon in days for the target.
        volatility_window: window used for realized volatility.

    Returns:
        A DataFrame with the features in FEATURE_COLUMNS plus the target
        column named 'target'. NaN rows are not dropped here, callers
        decide when to drop them.
    """
    result = df.copy()
    result["returns"] = log_returns(result["close"])
    result["realized_vol"] = realized_volatility(
        result["returns"], window=volatility_window
    )
    result["ewma_vol"] = ewma_volatility(result["returns"], span=volatility_window)
    result["rsi"] = rsi(result["close"])
    result["atr"] = average_true_range(result)
    result["bollinger_position"] = bollinger_position(result["close"])
    result["momentum"] = rolling_mean_deque(
        momentum(result["close"], window=10), window=volatility_window
    )
    result["volume_ratio"] = result["volume"] / result["volume"].rolling(20).mean()
    result["day_of_week"] = result.index.dayofweek

    # The target is the volatility over the NEXT horizon days.
    result["target"] = future_realized_volatility(
        result["returns"], horizon=horizon
    )
    return result


def prepare_for_model(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Turn the feature frame into X, y arrays ready for training.

    Rows with any missing value are dropped. Missing values happen at the
    start (rolling windows need warm up) and at the end (the target looks
    forward).

    Args:
        df: output of build_features.
        feature_columns: which columns to use, defaults to FEATURE_COLUMNS.

    Returns:
        A tuple of X (2D array), y (1D array) and the feature names used.
    """
    features = feature_columns or FEATURE_COLUMNS
    clean = df.dropna(subset=features + ["target"])
    if clean.empty:
        raise ValueError("No complete rows after dropping NaN, increase the data range")

    X = clean[features].values.astype(np.float64)
    y = clean["target"].values.astype(np.float64)
    return X, y, features


def make_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding sequences for the LSTM model.

    The LSTM does not see one row at a time, it sees a window of past
    rows. For each position we take the last `sequence_length` rows as
    the input and the target at that same position as the output.

    Args:
        X: 2D feature array (samples, features).
        y: 1D target array.
        sequence_length: how many past rows each sample contains.

    Returns:
        X_seq shaped (samples, sequence_length, features) and y_seq.
    """
    if len(X) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} rows to build sequences, got {len(X)}"
        )

    features = X.shape[1]
    count = len(X) - sequence_length
    X_seq = np.zeros((count, sequence_length, features), dtype=np.float64)
    y_seq = np.zeros(count, dtype=np.float64)

    for i in range(count):
        X_seq[i] = X[i : i + sequence_length]
        y_seq[i] = y[i + sequence_length - 1]

    return X_seq, y_seq

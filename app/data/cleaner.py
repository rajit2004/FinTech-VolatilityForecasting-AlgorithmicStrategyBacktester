"""Cleaning and preprocessing of raw market data.

The Yahoo Finance downloads and our bundled CSV files can have holes:
missing rows, duplicate dates, or a handful of zero/NaN values around
holidays. This module fixes all of that before the data goes into
feature engineering.
"""

import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw OHLCV frame so it is safe for downstream use.

    Steps performed:
    - Keep only the expected columns in a known order.
    - Drop fully empty rows and duplicate dates.
    - Sort by date so everything is chronological.
    - Fill small gaps forward (e.g. a missing price on a quiet day).
    - Drop any rows that still have missing values after that.

    Args:
        df: raw data with columns open, high, low, close, volume.

    Returns:
        A cleaned DataFrame indexed by date.

    Raises:
        ValueError: if the frame has no data or is missing key columns.
    """
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df.copy()
    cleaned = cleaned[required]

    # Drop rows that are entirely empty, they only add noise.
    cleaned = cleaned.dropna(how="all")
    if cleaned.empty:
        raise ValueError("DataFrame is empty after removing empty rows")

    # Make the date the index and drop duplicates, keeping the first one.
    if "date" in df.columns:
        cleaned["date"] = pd.to_datetime(df["date"])
        cleaned = cleaned.set_index("date")

    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    cleaned = cleaned.sort_index()

    # A single missing value in a price column is usually a quiet day.
    # Forward fill only those, then drop anything still broken.
    cleaned = cleaned.ffill()
    cleaned = cleaned.dropna()

    if cleaned.empty:
        raise ValueError("DataFrame is empty after cleaning")

    logger.info("Cleaned data: %d rows from %s to %s", len(cleaned), cleaned.index[0], cleaned.index[-1])
    return cleaned


def validate_symbol(symbol: str) -> str:
    """Normalize and check a ticker symbol.

    Args:
        symbol: the raw ticker string from a user or the config.

    Returns:
        The uppercase, stripped symbol.

    Raises:
        ValueError: if the symbol is empty or contains spaces.
    """
    cleaned = str(symbol).strip().upper()
    if not cleaned:
        raise ValueError("Symbol cannot be empty")
    if " " in cleaned:
        raise ValueError(f"Symbol cannot contain spaces: {symbol!r}")
    return cleaned

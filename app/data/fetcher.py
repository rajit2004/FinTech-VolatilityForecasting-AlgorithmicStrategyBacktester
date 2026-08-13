"""Fetches historical price data for stocks and crypto.

The primary source is Yahoo Finance through the yfinance library. When the
machine is offline or the symbol is not found, we fall back to bundled CSV
files inside the dataset folder so the project still works end to end.

We also show off a few of the concepts from the course here:
- asyncio: fetch_many_async() downloads several tickers concurrently.
- decorators: fetch_ohlcv is wrapped with retry() and time_it().
- generators: stream_csv() yields rows one by one instead of loading
  the whole file into memory.
"""

import asyncio
import csv
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

from app.config import DATASET_DIR, END_DATE, START_DATE
from app.data.cleaner import clean_ohlcv, validate_symbol
from app.utils.decorators import retry, time_it
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _csv_path(symbol: str) -> Path:
    """Return the path of the bundled CSV file for a symbol."""
    return DATASET_DIR / f"{symbol}.csv"


def stream_csv(symbol: str) -> Iterator[dict]:
    """Stream rows from a bundled CSV file one by one.

    Using a generator means we never hold the whole file in memory. This
    matters for long histories, and it is a good demonstration of how
    generators help process large data sets.

    Args:
        symbol: the ticker to read.

    Yields:
        Each CSV row as a dict.

    Raises:
        FileNotFoundError: if there is no bundled file for the symbol.
    """
    path = _csv_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No bundled data file for {symbol} at {path}")

    logger.info("Streaming rows from %s", path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def load_from_csv(symbol: str) -> pd.DataFrame:
    """Load the bundled CSV data for a symbol into a cleaned DataFrame.

    Args:
        symbol: the ticker to load.

    Returns:
        Cleaned OHLCV data indexed by date.
    """
    symbol = validate_symbol(symbol)
    rows = list(stream_csv(symbol))
    if not rows:
        raise ValueError(f"No rows found for {symbol} in the dataset folder")

    df = pd.DataFrame(rows)
    # The bundled files are stored with a date column for readability.
    df = clean_ohlcv(df)
    return df


@time_it
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def _download_from_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single symbol from Yahoo Finance.

    Wrapped with time_it and retry, so a flaky network call is retried
    with backoff before we give up.

    Args:
        symbol: the ticker to download.
        start: start date as YYYY-MM-DD.
        end: end date as YYYY-MM-DD.

    Returns:
        Cleaned OHLCV data.

    Raises:
        RuntimeError: if the download fails or returns nothing useful.
    """
    import yfinance as yf

    logger.info("Downloading %s from Yahoo Finance", symbol)
    raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

    # yfinance returns a multi index on its columns in newer versions.
    # Flatten it so we always work with simple names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")

    raw = raw.reset_index()
    raw.columns = [str(c).lower() for c in raw.columns]
    raw = raw.rename(columns={"adj close": "close"})
    return clean_ohlcv(raw)


def fetch_ohlcv(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    use_csv_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch and clean historical data for one symbol.

    Strategy: try Yahoo Finance first. If that fails (offline, bad ticker,
    rate limit) and use_csv_fallback is on, load the bundled CSV instead.

    Args:
        symbol: ticker to fetch.
        start: start date YYYY-MM-DD.
        end: end date YYYY-MM-DD.
        use_csv_fallback: whether to fall back to bundled CSV data.

    Returns:
        Cleaned OHLCV data.

    Raises:
        RuntimeError: if both the live source and the fallback fail.
    """
    symbol = validate_symbol(symbol)
    try:
        return _download_from_yahoo(symbol, start, end)
    except Exception as exc:  # noqa: BLE001, we deliberately fall back
        if not use_csv_fallback:
            raise RuntimeError(f"Could not fetch {symbol} from Yahoo: {exc}") from exc
        logger.warning("Yahoo fetch failed for %s (%s), falling back to CSV", symbol, exc)
        try:
            return load_from_csv(symbol)
        except Exception as csv_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not fetch {symbol} from Yahoo or bundled CSV"
            ) from csv_exc


def fetch_many(
    symbols: List[str],
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict:
    """Fetch several symbols sequentially and return a dict of frames.

    Args:
        symbols: list of tickers.
        start: start date YYYY-MM-DD.
        end: end date YYYY-MM-DD.

    Returns:
        Mapping of symbol to cleaned DataFrame.
    """
    result: dict = {}
    for symbol in symbols:
        try:
            result[symbol] = fetch_ohlcv(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch %s: %s", symbol, exc)
    return result


async def _fetch_one_async(
    symbol: str,
    start: str,
    end: str,
    semaphore: asyncio.Semaphore,
) -> Optional[tuple]:
    """Fetch one symbol inside an async task.

    The semaphore limits how many downloads run at the same time so we do
    not overwhelm the rate limits. yfinance is a blocking library, so we
    run it in a thread with asyncio.to_thread.
    """
    async with semaphore:
        try:
            frame = await asyncio.to_thread(fetch_ohlcv, symbol, start, end)
            return (symbol, frame)
        except Exception as exc:  # noqa: BLE001
            logger.error("Async fetch failed for %s: %s", symbol, exc)
            return None


async def fetch_many_async(
    symbols: List[str],
    start: str = START_DATE,
    end: str = END_DATE,
    max_concurrency: int = 5,
) -> dict:
    """Fetch many symbols concurrently using asyncio.

    We create one task per symbol and gather them. This is much faster
    than fetching tickers one after another, and it demonstrates async
    programming for I/O bound work.

    Args:
        symbols: list of tickers.
        start: start date YYYY-MM-DD.
        end: end date YYYY-MM-DD.
        max_concurrency: how many downloads to allow at once.

    Returns:
        Mapping of symbol to cleaned DataFrame.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _fetch_one_async(symbol, start, end, semaphore)
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks)
    return {symbol: frame for symbol, frame in results if frame is not None}


def run_fetch_many_async(
    symbols: List[str],
    start: str = START_DATE,
    end: str = END_DATE,
    max_concurrency: int = 5,
) -> dict:
    """Convenience wrapper that runs the async fetch from sync code.

    Args:
        symbols: list of tickers.
        start: start date YYYY-MM-DD.
        end: end date YYYY-MM-DD.
        max_concurrency: how many downloads to allow at once.

    Returns:
        Mapping of symbol to cleaned DataFrame.
    """
    return asyncio.run(
        fetch_many_async(symbols, start=start, end=end, max_concurrency=max_concurrency)
    )

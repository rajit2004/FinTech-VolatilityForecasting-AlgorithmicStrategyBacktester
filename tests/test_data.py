"""Tests for the data pipeline: cleaning, CSV loading and the database.

These tests only use the bundled sample data, so they run fully offline.
"""

import pandas as pd
import pytest

from app.data import database
from app.data.cleaner import clean_ohlcv, validate_symbol
from app.data.fetcher import load_from_csv, run_fetch_many_async
from app.utils import decorators


def test_clean_deduplicates_and_fills_missing():
    """The cleaner drops duplicate dates and fills a single missing close.

    The duplicate 2022-01-03 row is removed. The missing close on the
    quiet 2022-01-04 day is forward filled from the previous close, so
    that row stays instead of producing a hole in the series.
    """
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, None],
            "volume": [100, 100, 100],
        },
        index=pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-04"]),
    )
    cleaned = clean_ohlcv(frame)

    # No duplicate dates remain.
    assert not cleaned.index.duplicated().any()
    assert list(cleaned.index) == [
        pd.Timestamp("2022-01-03"),
        pd.Timestamp("2022-01-04"),
    ]
    # The missing close was filled forward from the previous row.
    assert cleaned["close"].iloc[-1] == 10.5


def test_clean_rejects_missing_columns():
    """A frame without the required columns must raise a clear error."""
    frame = pd.DataFrame({"open": [1.0], "high": [2.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_ohlcv(frame)


def test_clean_rejects_empty_frame():
    """Cleaning an empty frame should fail loudly, not silently."""
    with pytest.raises(ValueError):
        clean_ohlcv(pd.DataFrame({"open": [], "close": []}))


def test_validate_symbol_normalizes_and_rejects_bad_input():
    """Tickers get uppercased, empty or space containing ones are rejected."""
    assert validate_symbol("  aapl ") == "AAPL"

    with pytest.raises(ValueError):
        validate_symbol("")
    with pytest.raises(ValueError):
        validate_symbol("bad symbol")


def test_database_save_and_load_roundtrip(configured_db, clean_frame):
    """Prices saved to the DB come back with the same rows and values."""
    symbol = "AAPL"
    saved = database.save_prices(symbol, clean_frame, source="csv")

    # Saving twice must not create duplicates, so the second call saves 0.
    saved_again = database.save_prices(symbol, clean_frame, source="csv")
    assert saved_again == 0

    loaded = database.load_prices(symbol)
    assert len(loaded) == len(clean_frame)
    assert saved > 0
    # Spot check a known value survives the round trip.
    assert loaded["close"].iloc[0] == clean_frame["close"].iloc[0]


def test_load_from_csv_with_monkeypatched_dataset(tmp_path, monkeypatch):
    """load_from_csv reads a bundled CSV file for a symbol correctly."""
    from app.data import fetcher

    csv_text = (
        "date,open,high,low,close,volume\n"
        "2022-01-03,100,105,99,104,1000\n"
        "2022-01-04,104,110,103,109,1200\n"
    )
    (tmp_path / "TEST.csv").write_text(csv_text, encoding="utf-8")
    monkeypatch.setattr(fetcher, "DATASET_DIR", tmp_path)

    frame = load_from_csv("TEST")
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 2


def test_fetch_many_async_falls_back_to_csv(tmp_path, monkeypatch):
    """The async fetch falls back to bundled CSV when the live source fails."""
    from app.data import fetcher

    csv_text = (
        "date,open,high,low,close,volume\n"
        "2022-01-03,50,52,49,51,500\n"
        "2022-01-04,51,53,50,52,600\n"
    )
    (tmp_path / "TEST.csv").write_text(csv_text, encoding="utf-8")
    monkeypatch.setattr(fetcher, "DATASET_DIR", tmp_path)

    # Force the live download to always fail so the fallback is exercised.
    def _broken_download(symbol, start, end):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(fetcher, "_download_from_yahoo", _broken_download)

    results = run_fetch_many_async(["TEST"])
    assert "TEST" in results
    assert len(results["TEST"]) == 2


def test_retry_decorator_retries_then_succeeds():
    """The retry decorator keeps calling until the function succeeds."""
    calls = {"count": 0}

    @decorators.retry(max_attempts=3, delay=0)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise ConnectionError("transient failure")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 2


def test_cached_decorator_computes_once():
    """The cached decorator computes once and reuses the result."""
    calls = {"count": 0}

    @decorators.cached()
    def expensive(value):
        calls["count"] += 1
        return value * 2

    assert expensive(3) == 6
    assert expensive(3) == 6
    assert calls["count"] == 1
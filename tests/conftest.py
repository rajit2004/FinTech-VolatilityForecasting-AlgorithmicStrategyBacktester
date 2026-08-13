"""Shared fixtures for all test modules.

The most important fixture is `clean_frame`, deterministic synthetic
OHLCV data produced by the same simulator the app uses offline. Tests
never hit the network, they work on this generated data.

`configured_db` points the database module at a fresh file in a temp
folder, so every test starts with a clean database and never touches the
real one.
"""

import sys
from pathlib import Path

# Make the project root importable so pytest, wherever it was launched
# from, can find the app package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from app.data import database  # noqa: E402
from app.data.cleaner import clean_ohlcv  # noqa: E402
from scripts.generate_sample_data import simulate_ohlcv  # noqa: E402


@pytest.fixture(scope="session")
def clean_frame() -> pd.DataFrame:
    """A reproducible cleaned OHLCV frame for the test symbol.

    Generated once per session since it is read only and identical for
    every test that uses it.
    """
    raw = simulate_ohlcv("AAPL", start="2022-01-01", end="2023-12-31", seed=3)
    return clean_ohlcv(raw)


@pytest.fixture
def configured_db(tmp_path) -> str:
    """Point the database at a fresh temp file and return its URL.

    Each test gets its own isolated database file so tests never leak
    rows into each other.
    """
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path.as_posix()}"
    database.configure_database(url)
    return url


@pytest.fixture
def app_client(configured_db):
    """A Flask test client bound to the isolated test database."""
    from app.main import create_app

    app = create_app(database_url=configured_db)
    return app.test_client()
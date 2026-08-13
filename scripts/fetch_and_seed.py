"""Helper script: generate sample data and load it into the database.

This is the fastest way to get the project into a usable state offline.

Usage:
    python scripts/fetch_and_seed.py
"""

import sys
from pathlib import Path

# Make the project root importable when this script runs directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import END_DATE, START_DATE, SYMBOLS
from app.data.database import init_db, save_prices
from app.data.fetcher import load_from_csv
from app.utils.logger import get_logger
from scripts.generate_sample_data import build_sample_dataset

logger = get_logger(__name__)


def main() -> None:
    """Generate the offline CSV data and store it in the database."""
    init_db()
    build_sample_dataset(SYMBOLS, start=START_DATE, end=END_DATE)

    total = 0
    for symbol in SYMBOLS:
        try:
            frame = load_from_csv(symbol)
            saved = save_prices(symbol, frame, source="csv")
            total += saved
            print(f"{symbol}: saved {saved} rows")
        except Exception as exc:  # noqa: BLE001, report and continue
            logger.error("Could not seed %s: %s", symbol, exc)

    print(f"Done, saved {total} rows in total.")


if __name__ == "__main__":
    main()
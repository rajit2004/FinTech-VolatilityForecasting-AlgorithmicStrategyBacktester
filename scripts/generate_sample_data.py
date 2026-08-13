"""Generate realistic sample OHLCV data into the dataset folder.

This gives the project an offline data source. The simulation is a
geometric random walk with volatility clustering, meaning calm periods
follow calm periods and wild periods follow wild periods, which is how
real markets behave. Each run with the same seed produces the same data,
so results are reproducible.

Usage:
    python scripts/generate_sample_data.py
"""

import random
import sys
from pathlib import Path

# Make the project root importable when this script runs directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.config import DATASET_DIR, END_DATE, START_DATE, SYMBOLS

# Different tickers get different base prices and day to day drift so the
# sample data does not all look identical.
_SYMBOL_PROFILES = {
    "AAPL": {"start_price": 180.0, "drift": 0.0004, "base_vol": 0.015},
    "MSFT": {"start_price": 330.0, "drift": 0.0005, "base_vol": 0.014},
    "GOOG": {"start_price": 140.0, "drift": 0.0003, "base_vol": 0.017},
    "BTC-USD": {"start_price": 30000.0, "drift": 0.0010, "base_vol": 0.035},
    "ETH-USD": {"start_price": 2000.0, "drift": 0.0009, "base_vol": 0.040},
}

# The default symbol set, used when a symbol has no profile entry.
_DEFAULT_PROFILE = {"start_price": 100.0, "drift": 0.0003, "base_vol": 0.020}


def _profile_for(symbol: str) -> dict:
    """Return the generation profile for a symbol, with a safe default."""
    return _SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE)


def simulate_ohlcv(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate OHLCV bars for one symbol between two dates.

    The random walk updates a volatility term each day using the classic
    GARCH style recurrence:

        sigma[t]^2 = w + a * return[t-1]^2 + b * sigma[t-1]^2

    This is what creates volatility clustering in the data.

    Args:
        symbol: ticker to simulate.
        start: start date YYYY-MM-DD.
        end: end date YYYY-MM-DD.
        seed: random seed for reproducible output.

    Returns:
        A DataFrame with date, open, high, low, close and volume columns.
    """
    rng = random.Random(seed)
    profile = _profile_for(symbol)

    days = pd.bdate_range(start=start, end=end, freq="D")
    current_price = profile["start_price"]
    volatility = profile["base_vol"]

    rows = []
    for day in days:
        # GARCH style update, big moves feed more big moves.
        shock = rng.gauss(0, 1)
        next_vol_squared = (
            0.01 * profile["base_vol"] ** 2
            + 0.10 * (shock * volatility) ** 2
            + 0.85 * volatility ** 2
        )
        volatility = max(next_vol_squared ** 0.5, 0.002)

        daily_return = profile["drift"] + volatility * shock
        open_price = current_price
        close_price = current_price * (1 + daily_return)
        high = max(open_price, close_price) * (1 + abs(rng.gauss(0, volatility / 3)))
        low = min(open_price, close_price) * (1 - abs(rng.gauss(0, volatility / 3)))

        volume = int(abs(rng.gauss(1_000_000, 300_000)) * (1 + 5 * volatility))
        rows.append(
            {
                "date": day.date().isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": volume,
            }
        )
        current_price = close_price

    return pd.DataFrame(rows)


def build_sample_dataset(
    symbols=None,
    start: str = START_DATE,
    end: str = END_DATE,
    out_dir: Path = DATASET_DIR,
    seed: int = 42,
) -> dict:
    """Generate CSV files for a set of symbols into the dataset folder.

    Args:
        symbols: list of tickers, defaults to the configured SYMBOLS.
        start: start date YYYY-MM-DD.
        end: end date YYYY-MM-DD.
        out_dir: where to write the files.
        seed: random seed, stable across symbols.

    Returns:
        A dict of symbol to the CSV path that was written.
    """
    symbols = symbols or SYMBOLS
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for index, symbol in enumerate(symbols):
        frame = simulate_ohlcv(symbol, start=start, end=end, seed=seed + index * 7)
        path = out_dir / f"{symbol.upper()}.csv"
        frame.to_csv(path, index=False)
        written[symbol.upper()] = path

    return written


if __name__ == "__main__":
    print(f"Generating sample data into {DATASET_DIR}")
    files = build_sample_dataset()
    for symbol, path in files.items():
        print(f"  {symbol}: {path}")
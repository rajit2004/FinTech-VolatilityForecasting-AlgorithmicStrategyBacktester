"""Central place for all configuration values.

Everything that could change between machines (database path, tickers,
dates, model settings) lives in a .env file. We load that file once here
with python-dotenv and expose the values as simple module level constants,
so the rest of the code can just import them.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from the .env file if it exists. Without it we still run,
# we just fall back to the default values below.
load_dotenv()

# The root folder of the project, computed from this file's location so it
# does not matter where the repo is cloned to.
BASE_DIR = Path(__file__).resolve().parent.parent

# The folder that holds the bundled CSV files used when we are offline.
DATASET_DIR = BASE_DIR / "dataset"

# The database connection for the app. This is PostgreSQL by default. The
# default below points at the local development database the project ships
# with. For any other environment, set DATABASE_URL in your .env file and
# it overrides this value.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://volforecaster:CHANGE_ME@localhost:5432/volforecaster",
)

# Symbols and date range for data downloads.
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,BTC-USD").split(",") if s.strip()]
START_DATE = os.getenv("START_DATE", "2021-01-01")
END_DATE = os.getenv("END_DATE", "2024-12-31")

# Logging setup.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Model settings. horizon_days decides how far ahead we forecast.
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "30"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "25"))
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "5"))

# Rolling windows used by the feature engineering and indicators.
VOLATILITY_WINDOW = 20
ANNUALIZATION_FACTOR = 252  # trading days in a year for daily data

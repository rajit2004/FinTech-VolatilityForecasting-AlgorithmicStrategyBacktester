# Complete Project Guide

## FinTech Volatility Forecasting and Algorithmic Strategy Backtester

This is a single document that explains every part of the project. You
should be able to read this top to bottom and talk about the project for
an hour. Every file, every feature, every model, every design decision
is covered here.

---

## 1. What This Project Does

Most people think about stock prices going up or down. Professional
traders think about how violently prices move. That movement is called
volatility. If you can predict when volatility is about to spike, you
can size your positions differently and manage risk better.

This project does three things:

1. It collects historical price data for stocks and crypto.
2. It uses machine learning to forecast future volatility.
3. It runs algorithmic trading strategies against that data and tells
   you how much money you would have made (or lost) with transaction
   costs included.

Everything is exposed through a REST API and a Streamlit dashboard,
so you can explore results without touching code.

The project runs fully offline using generated sample data, so anyone
can clone it and see it work immediately.

---

## 2. How the Program Flows

Here is the full pipeline from raw data to dashboard, step by step.

```text
Step 1: Data Collection
  Yahoo Finance downloads OHLCV bars (or bundled CSV fallback)
          |
          v
Step 2: Cleaning
  Duplicate dates removed, missing values filled, chronological order
          |
          v
Step 3: Storage
  Cleaned prices saved to PostgreSQL (or SQLite for tests)
          |
          v
Step 4: Feature Engineering
  Raw prices turned into 9 features:
    returns, realized vol, EWMA vol, RSI, ATR,
    Bollinger position, momentum, volume ratio, day of week
          |
          v
Step 5: Target Creation
  Future realized volatility over the next N days
          |
          v
Step 6: Model Training
  Random Forest, LSTM, or GARCH trained on features
          |
          v
Step 7: Prediction
  Model forecasts next period volatility
          |
          v
Step 8: Strategy Signals
  Trading strategy generates buy/sell signals
          |
          v
Step 9: Backtesting
  Engine simulates trades bar by bar with commission
          |
          v
Step 10: Metrics
  Sharpe ratio, max drawdown, win rate, total return computed
          |
          v
Step 11: Serving
  Flask API returns JSON, Streamlit dashboard renders charts
```

The key insight is that the forecasting and the trading are connected.
The forecast tells you how risky the next period is, and the backtester
uses that information to decide how much to trade.

---

## 3. File by File Walkthrough

### app/config.py

This file loads all settings from a .env file using python-dotenv. It
exposes constants that the rest of the app imports. Having one place
for configuration means you change the database URL or date range in
one spot and everything else picks it up.

Key values:
- DATABASE_URL: PostgreSQL connection string (default points at local
  dev database, .env overrides it)
- SYMBOLS: which tickers to fetch (default: AAPL, MSFT, BTC-USD)
- START_DATE, END_DATE: date range for data downloads
- LSTM_SEQUENCE_LENGTH: how many past rows the LSTM sees (default 30)
- LSTM_EPOCHS: training epochs for the LSTM (default 25)
- FORECAST_HORIZON_DAYS: how far ahead to forecast (default 5)
- VOLATILITY_WINDOW: rolling window for volatility calculation (20)
- ANNUALIZATION_FACTOR: 252 trading days per year

Why it matters: every module imports from config, so nothing is
hardcoded in the business logic. Tests can override values by
pointing at different .env files.

---

### app/main.py

This is the entry point. It does two things:

1. Starts the Flask API server when run with python -m app.main
2. Provides a CLI with three commands: fetch, forecast, backtest

The create_app function builds the Flask application, registers the
API blueprint, and configures CORS so the dashboard can talk to the
API from a browser. It accepts an optional database_url parameter so
tests can point at an isolated database.

The CLI commands are thin wrappers:
- fetch: downloads prices and stores them
- forecast: trains a model and prints the next period forecast
- backtest: runs a strategy and prints the metrics

The argparse parser supports --model with choices random_forest, lstm,
and garch. The backtest command supports --target-vol and --forecast-vol
for volatility-based position sizing.

---

### app/data/cleaner.py

Two functions here:

clean_ohlcv(df): Takes raw OHLCV data and makes it safe for
downstream use. Steps: keep only the 5 expected columns, drop fully
empty rows, drop duplicate dates, sort chronologically, forward fill
small gaps, drop any remaining broken rows. Raises ValueError if the
result is empty or missing required columns.

validate_symbol(symbol): Normalizes a ticker to uppercase and rejects
empty strings or strings with spaces. Used everywhere a user-provided
symbol enters the system.

Why it matters: Yahoo Finance data can have holes around holidays,
duplicate rows from timezone changes, or missing columns. This module
fixes all of that before anything else sees the data.

---

### app/data/fetcher.py

This is the data collection layer. It tries Yahoo Finance first and
falls back to bundled CSV files when offline.

Key functions:

fetch_ohlcv(symbol, start, end): The main function. Tries yfinance
download, falls back to CSV if that fails. Returns cleaned data.

_download_from_yahoo(symbol, start, end): Downloads from Yahoo Finance
using yfinance. Wrapped with @retry(3 attempts) and @time_it decorators.
Handles the multi-index column issue in newer yfinance versions.

load_from_csv(symbol): Reads a bundled CSV file from the dataset folder.
Uses the stream_csv generator to read rows one by one instead of
loading the whole file into memory.

stream_csv(symbol): A generator that yields CSV rows as dicts. This
demonstrates the generator concept from the course. It processes large
files without holding everything in memory.

fetch_many(symbols): Fetches several symbols sequentially and returns
a dict mapping symbol to DataFrame.

fetch_many_async(symbols): Uses asyncio to download several symbols
concurrently. Creates one task per symbol, uses a semaphore to limit
concurrency to 5 at a time. yfinance is blocking, so each download
runs in a thread via asyncio.to_thread. This is much faster than
sequential fetching for multiple symbols.

run_fetch_many_async: Convenience wrapper that runs the async fetch
from synchronous code.

Course concepts demonstrated: asyncio (concurrent fetching), generators
(streaming CSV rows), decorators (retry, time_it).

---

### app/data/database.py

The database layer. Uses SQLAlchemy ORM with PostgreSQL. Tests use
SQLite instead (same ORM, different driver).

Four ORM models:

Price: stores one OHLCV bar for one symbol on one date. Has a unique
constraint on (symbol, date) so the same bar cannot be inserted twice.

VolatilityForecast: stores one prediction made by a model. Records the
symbol, model name, forecast horizon, date, and predicted volatility.

BacktestRun: stores one complete backtest. Records the symbol,
strategy name, parameters (JSON), metrics (JSON), and created timestamp.

BacktestTrade: stores one trade from a backtest run. Linked to
BacktestRun via foreign key. Records date, action (BUY/SELL), price,
size, and PnL.

Key helper functions:

configure_database(url): Points the module at a different database URL.
Tests use this to point at a temporary SQLite file. Creates all tables.

save_prices(symbol, df): Inserts cleaned OHLCV rows, skipping dates
that already exist for this symbol. Returns the count of new rows.

load_prices(symbol, start, end): Loads stored prices into a DataFrame.
Optionally filters by date range.

iter_price_chunks(symbol, chunksize): A generator that streams stored
prices in chunks using yield_per. Memory stays flat even for very
long histories. This is the generators requirement for the course.

save_forecast, load_forecasts: Store and retrieve volatility predictions.

save_backtest, load_backtest_runs, load_backtest_run: Store and retrieve
backtest results with their trades. load_backtest_run uses selectinload
to eagerly load trades while the session is open.

---

### app/features/engineering.py

This is where raw prices become model inputs. It builds 9 features
plus the target variable.

The 9 features:

1. returns: daily log returns (log of price ratio). Log returns are
   additive over time, which is useful for volatility work.

2. realized_vol: rolling standard deviation of returns, annualized by
   multiplying by sqrt(252). This is the most basic volatility measure.

3. ewma_vol: exponentially weighted moving average volatility. Reacts
   faster to new information than simple rolling vol. For short series
   (under 700 values) it uses a recursive implementation to demonstrate
   recursion. For longer series it uses pandas ewm for speed.

4. rsi: Relative Strength Index. A 0 to 100 oscillator measuring
   overbought/oversold conditions. Above 70 is overbought, below 30
   is oversold. Used as a feature, not a signal.

5. atr: Average True Range. Measures the largest single-day price
   move including overnight gaps. A common volatility indicator.

6. bollinger_position: where the price sits inside its Bollinger
   bands, scaled from -1 (at lower band) to +1 (at upper band).
   Tells the model how stretched the price is.

7. momentum: rate of change of price over 10 days, smoothed with
   a rolling mean using a deque. Demonstrates the deque data structure
   from the course.

8. volume_ratio: today's volume divided by its 20-day average.
   Spikes in volume often accompany volatility spikes.

9. day_of_week: integer 0-4 (Monday to Friday). Some days have
   different volatility patterns.

The target variable: future_realized_volatility computes the realized
volatility over the NEXT horizon days. This is what the models try
to predict. Because it looks forward, the last horizon rows have NaN
targets and are dropped before training.

Key functions:

build_features(df, horizon): builds all 9 features plus the target.
Returns a DataFrame with everything. Does not drop NaN rows, letting
callers decide.

prepare_for_model(df): turns the feature DataFrame into X and y arrays
for scikit-learn. Drops rows with any missing value.

make_lstm_sequences(X, y, sequence_length): builds sliding windows for
the LSTM. Each sample is a (sequence_length, n_features) window. The
LSTM does not see one row at a time, it sees a window of past rows.

recursive_ema(values, alpha): exponential moving average using
recursion instead of a loop. Demonstrates the recursion concept. The
EMA of a list is the last value blended with the EMA of everything
before it. Guarded to only work on series up to 700 values.

rolling_mean_deque(values, window): rolling mean using a deque with
maxlen instead of pandas rolling. A deque automatically evicts old
values when full, which is exactly what you need for a live trading
feed.

Course concepts demonstrated: recursion (recursive_ema), generators
(not directly here but the pattern is used), deques (rolling_mean_deque).

---

### app/models/volatility_model.py

This file contains three forecasting models and the pipeline that
connects them to the rest of the app.

**RandomForestVolatilityModel:**
A scikit-learn RandomForestRegressor with 200 trees and max depth 10.
Takes the 9 engineered features and predicts future volatility. Does
not need feature scaling. Handles nonlinear relationships well. This
is the classical ML baseline.

Interface: fit(X, y) trains the model, predict(X) returns predictions.

**LSTMVolatilityModel:**
A Keras Sequential model with one LSTM layer (64 units) followed by
two Dense layers. Reads a window of 30 past rows and outputs a single
volatility number. Features and targets are scaled to 0-1 with
MinMaxScaler before training because neural networks train better on
normalized inputs. The scaler is inverted on predict so outputs are
in the original scale.

Uses Keras 3 with the JAX backend because TensorFlow has no wheels
for Python 3.14 yet. The code is identical to the classic TensorFlow
API, just the backend changes.

Interface: fit(X_seq, y) trains on sequences, predict(X_seq) returns
predictions.

**GARCHVolatilityModel:**
A GARCH(1,1) model from the arch library. This is the statistical
baseline that every serious ML paper since 2020 compares against.
GARCH models the conditional variance of returns directly using an
autoregressive structure: today's variance depends on yesterday's
squared return and yesterday's variance.

Unlike the ML models, GARCH works on raw log returns, not engineered
features. It fits a constant-mean GARCH(1,1) model on percentage-
scaled returns. The forecast method returns conditional variance,
which is converted to annualized volatility by taking the square
root and multiplying by sqrt(252).

Interface: fit(returns) trains on log returns, predict(horizon) returns
annualized volatility for the next N days, predict_next() returns a
single value for the next day.

**train_test_split_time(X, y):**
Splits data in time order, never randomly. The last 20% of rows
becomes the test set. Time series must never be shuffled before
splitting because the model learns from the past and is tested on
the future.

**run_volatility_forecast(df, horizon, model_name):**
The main pipeline function. Steps:
1. Build features from the cleaned OHLCV data
2. Prepare X and y arrays
3. Split in time order
4. Train the chosen model
5. Evaluate on the test set
6. Produce a one-step-ahead forecast for the next period
7. Return predictions, metrics, dates, and the next forecast

For GARCH, it works differently: it fits on raw returns, not features,
so it has its own branch in the function. The test set is evaluated
by computing realized volatility of the actual test returns and
comparing against GARCH's conditional variance predictions.

---

### app/models/evaluator.py

Computes three standard regression metrics:

RMSE (Root Mean Squared Error): the square root of the average
squared difference between predictions and truth. Squaring punishes
big errors extra hard, which matters when a large missed volatility
spike hurts a trading strategy.

MAE (Mean Absolute Error): the average absolute distance of
predictions from truth. Does not overweight big errors like RMSE.

R-squared: how much of the variance the model explains. A value near
1 means predictions track truth closely. A value near 0 means the
model is no better than predicting the mean.

compare_models: picks the model with the lowest RMSE from a dict of
evaluation results.

---

### app/models/explainer.py

Model explainability using SHAP (SHapley Additive exPlanations).
SHAP tells you which features pushed a prediction higher or lower,
and by how much.

explain_with_shap(model, X, feature_names, X_background):
Low-level function. Creates a TreeExplainer from the shap library,
computes SHAP values for the input data, and returns:
- shap_values: per-row contributions from each feature
- feature_importance: mean absolute SHAP value per feature, ranked
- base_value: the expected model output

TreeExplainer is fast for tree-based models and exact for random
forests. The background dataset is a random subset of the training
data (100 rows max) used to estimate the expected model output.

explain_forecast(model, X_train, X_test, feature_names, test_predictions):
High-level function called by the API. Takes a trained
RandomForestVolatilityModel, computes SHAP values for the test set,
and returns feature importance plus the top contributions for the
most recent prediction. Produces a human-readable summary like
"Top features driving predictions: realized_vol, ewma_vol, rsi".

Why it matters: in financial forecasting, you need to know the model
is not just memorizing noise. SHAP provides that trust layer.

---

### app/strategies/base_strategy.py

Abstract base class for all trading strategies. Defines the interface
that every strategy must implement.

Attributes:
- name: human readable strategy name
- params: tuning parameters as a dict

Abstract method:
- generate_signals(df): takes cleaned OHLCV data and returns a Series
  of -1, 0, or +1 for every row. +1 means long, 0 means flat, -1
  means short.

Why it matters: the backtester engine can run any strategy without
knowing what it does inside. It just calls generate_signals and
processes the result. Adding a new strategy means writing one class
that implements this interface.

---

### app/strategies/moving_average.py

Moving average crossover strategy. The classic trend following idea.

Parameters:
- fast: window for the fast moving average (default 10)
- slow: window for the slow moving average (default 50)
- allow_short: when True, go short below the slow MA (default False)

Logic:
- Compute fast and slow moving averages of closing price
- When fast is above slow, signal is +1 (long)
- When fast is below slow, signal is 0 (flat) or -1 (short)
- Before the slow MA has enough data, signal is 0 (flat)

Validation: fast must be smaller than slow, otherwise ValueError.

This is one of the simplest strategies that still works in practice.
It serves as a benchmark against the volatility breakout strategy.

---

### app/strategies/volatility_breakout.py

Breakout trading on Bollinger bands. Markets that were trading
quietly often start a strong move when they break out of their range.

Parameters:
- window: rolling window for mean and standard deviation (default 20)
- num_std: how many standard deviations wide the bands are (default 2.0)

Logic:
- Compute rolling mean and standard deviation of closing price
- Upper band = mean + num_std * std
- Lower band = mean - num_std * std
- Enter long when price closes above upper band (breakout upward)
- Exit to flat when price falls back below middle band (breakout failed)
- Before the window has enough data, stay flat

This strategy leans on volatility itself, which ties nicely into the
volatility forecasting part of the project.

---

### app/backtester/engine.py

The core simulation engine. Runs a strategy over historical data,
bar by bar.

**BacktestConfig dataclass:**
- initial_capital: starting account balance (default 100,000)
- commission: fraction of trade value charged per side (default 0.1%)
- position_pct: how much of the account to invest per trade (default 95%)
- slippage: per share price slippage (default 0, for simplicity)
- target_volatility: if > 0, enable vol-based position sizing
- forecast_volatility: the forecast to scale against

**BacktestResult dataclass:**
Holds symbol, strategy name, params, config, metrics, equity curve,
and trades. Has a to_dict method for JSON serialization.

**get_strategy(name, params):**
Factory function that creates a strategy instance from its name.
Supports "moving_average" and "volatility_breakout".

**iter_bar_events(df, strategy, config):**
The heart of the simulation. A generator that yields one event dict
per bar. On each day:
1. Check what position the strategy wants
2. If different from current position, execute a trade
3. On buy: calculate budget with vol-based sizing, buy integer shares,
   deduct commission
4. On sell: sell all shares, deduct commission, book round-trip PnL
5. Record date, price, signal, cash, shares, equity

Volatility-based position sizing: when target_volatility and
forecast_volatility are both positive, the position budget is scaled
by min(target / forecast, 1.0). When forecast volatility is high,
we trade less. When it is low, we trade the full amount. This directly
connects the forecasting and trading halves of the project.

**run_backtest(symbol, df, strategy_name, params, config):**
Runs one backtest end to end. Creates the strategy, runs the event
generator, builds the equity curve and trade list, computes metrics.
Returns a BacktestResult.

**run_multiple_backtests(...):**
Runs a grid of parameter combinations. When parallel=True, uses
ProcessPoolExecutor to run them on multiple cores. Each worker gets
one parameter set. The _run_one function must be at module level
(not a method) so it can be pickled on Windows.

---

### app/backtester/metrics.py

Computes backtest performance metrics from the equity curve and trades.

Sharpe ratio: how much return we get per unit of risk, annualized.
Calculated as mean(daily_returns) / std(daily_returns) * sqrt(252).
A Sharpe above 1 is good, above 2 is excellent.

Max drawdown: the worst peak-to-trough drop of the account. Calculated
by comparing the equity curve to its running maximum.

CAGR: compounded annual growth rate. The annualized return of the
account.

Win rate: fraction of closed trades that were profitable.

Profit factor: gross profits divided by gross losses. When there are
no losses, it returns None instead of Infinity (PostgreSQL rejects
Infinity in JSON).

Total return: final equity divided by initial equity, minus 1.

Volatility: annualized standard deviation of daily returns.

---

### app/api/routes.py

All Flask route handlers. Every endpoint is a thin layer: validate
the request, call the right part of the app, return JSON.

Endpoints:

GET /api/health: liveness check, returns {"status": "ok"}.

POST /api/data/fetch: fetches prices for one or more symbols.
Accepts {"symbols": ["AAPL", "BTC-USD"]}. Uses async fetching.

GET /api/data/{symbol}: returns stored prices for a symbol.
Optional query params start and end filter the date range.

GET /api/features/{symbol}: returns engineered features.
Optional ?horizon=5 sets the forecast horizon.

POST /api/forecast: trains a model and returns a forecast.
Accepts {"symbol": "AAPL", "model_name": "random_forest", "horizon": 5}.
Returns next forecast, metrics, test predictions and true values.

GET /api/explain/{symbol}: explains which features drive the random
forest forecast using SHAP. Returns feature importance rankings and
the top contributing features for the most recent prediction.

GET /api/forecasts/{symbol}: lists stored forecasts.
Optional ?model_name=lstm filters by model.

POST /api/backtest: runs a backtest. Accepts symbol, strategy name,
strategy params, and optional config with initial_capital, commission,
target_volatility, and forecast_volatility. Returns metrics, equity
curve, and trades.

GET /api/backtests: lists all stored backtest runs.
Optional ?symbol=AAPL filters the list.

GET /api/backtests/{run_id}: returns one backtest run with all trades.

The _jsonable helper converts numpy and pandas types into JSON-safe
values. Flask's jsonify cannot handle numpy floats or pandas
Timestamps.

---

### app/api/schemas.py

Request validation for the API endpoints. Each parser takes raw JSON,
checks it, and returns a clean dict or raises ValueError.

parse_fetch_payload: validates symbols (accepts list or comma-separated
string), start and end dates.

parse_forecast_payload: validates symbol, model_name (must be
random_forest, lstm, or garch), and horizon.

parse_backtest_payload: validates symbol, strategy name (must be
moving_average or volatility_breakout), params, and config. Config
includes initial_capital, commission, target_volatility, and
forecast_volatility.

parse_symbols: normalizes and validates a list of ticker symbols.

---

### app/utils/decorators.py

Three reusable decorators:

time_it: logs how long a function took to run. Handy for expensive
operations like model training.

retry(max_attempts, delay, backoff): retries a function when it fails.
The delay grows after each failure (exponential backoff). After the
last attempt the original exception is raised. Used for network calls
that can be flaky.

cached: remembers the result of an expensive call keyed by its
arguments. If the same arguments come in again, returns the saved
result instantly. Uses a dict to map argument tuples to values.

---

### app/utils/logger.py

Configures Python's logging system once. Sets up a console handler
with a format that includes time, level, and module name. Every module
calls get_logger(__name__) to get a ready-to-use logger.

The _configured flag ensures the root logger is only set up once even
if the module is imported multiple times.

---

### dashboard/app.py

Streamlit dashboard with four tabs.

Sidebar:
- Symbol selector (populated from database or defaults)
- Strategy selector (moving_average or volatility_breakout)
- Strategy parameter sliders (fast/slow for MA, window/std for breakout)
- "Load sample data" button for offline use

Tab 1 - Prices:
Shows the closing price line chart and the realized/EWMA volatility
lines. Uses st.line_chart for quick plotting.

Tab 2 - Volatility Forecast:
Radio buttons to pick random_forest, lstm, or garch. Number input for
horizon. "Run forecast" button trains the model and shows:
- Next predicted volatility (metric card)
- RMSE and R-squared (metric cards)
- Predicted vs actual volatility line chart

Tab 3 - Backtest:
"Run backtest" button runs the selected strategy and shows:
- Total return, Sharpe ratio, max drawdown, win rate (metric cards)
- Equity curve line chart
- Full metrics in an expander
- Trades table in an expander

Tab 4 - About:
Brief description of the project.

Uses @st.cache_data for caching loaded prices and available symbols
so the UI stays responsive.

---

### scripts/generate_sample_data.py

Generates realistic sample OHLCV data. Uses a geometric random walk
with GARCH-style volatility clustering:

    sigma[t]^2 = w + a * return[t-1]^2 + b * sigma[t-1]^2

This creates the realistic pattern where calm periods follow calm
periods and wild periods follow wild periods.

Each symbol has a profile with a starting price, daily drift, and
base volatility. Different symbols look different (AAPL starts at 180,
BTC-USD starts at 30000).

simulate_ohlcv(symbol, start, end, seed): generates one symbol's data.
The seed ensures reproducibility.

build_sample_dataset(symbols): generates CSV files for all symbols
into the dataset folder.

Course concept demonstrated: this is how the offline data source works.
The project works fully without internet because of this.

---

### scripts/fetch_and_seed.py

Helper script that generates sample data and loads it into the
database. The fastest way to get the project into a usable state.
Run with: python scripts/fetch_and_seed.py

---

### scripts/demo_backtest.py

Runs both strategies on one symbol and prints a side-by-side
comparison. Run with: python scripts/demo_backtest.py --symbol AAPL

---

### tests/ (6 test files, 64 test cases)

conftest.py: shared fixtures. clean_frame generates reproducible
synthetic OHLCV data. configured_db points the database at a temp
file. app_client provides a Flask test client.

test_data.py: tests cleaning, CSV loading, database roundtrip,
async fetching, retry and caching decorators.

test_features.py: tests log returns, realized volatility, recursive
EMA, deque rolling mean, feature building, LSTM sequence creation.

test_models.py: tests random forest predictions, time split, evaluator
metrics, LSTM training, GARCH training, SHAP explainability, and the
full forecast pipeline for all three models.

test_strategies.py: tests strategy abstract class, MA crossover logic
(uptrend, downtrend, warmup, parameter validation), volatility
breakout logic.

test_backtester.py: tests backtest results, empty frame rejection,
unknown strategy rejection, flat strategy, generator type, metrics
calculation, grid runs, and volatility-based position sizing (3 tests:
high vol reduces position, low vol gives full position, disabled when
zero).

test_api.py: tests all API endpoints through the Flask test client.
Health check, data fetch, prices, backtest flow, forecast, stored
forecasts, 404 handling.

All tests run fully offline on seeded synthetic data against an
isolated temporary database. They do not need PostgreSQL running.

---

## 4. The Three Models

### Random Forest

How it works: a random forest is an ensemble of decision trees. Each
tree is trained on a random subset of the data and features. The final
prediction is the average of all trees. This averaging reduces
overfitting compared to a single tree.

Why it fits: random forests handle nonlinear relationships well, do
not need feature scaling, and are fast to train. They work great on
tabular data like our 9-feature set.

Our implementation: 200 trees, max depth 10, trained with 80% of the
data (time-ordered split), tested on the remaining 20%.

### LSTM

How it works: Long Short-Term Memory is a type of recurrent neural
network. It reads a sequence of past rows (30 in our case) and
learns patterns in the order. The internal memory cells can remember
information over many time steps, which fits volatility's memory effect
(calm periods tend to stay calm).

Why it fits: volatility has strong temporal dependencies. The LSTM's
recurrent structure is a natural match for this kind of data.

Our implementation: one LSTM layer with 64 units, followed by Dense(16,
relu) and Dense(1). Features scaled to 0-1 with MinMaxScaler. Trained
for 25 epochs with batch size 32. Uses Keras 3 with JAX backend.

### GARCH(1,1)

How it works: GARCH models the conditional variance of returns. The
formula is:

    sigma[t]^2 = omega + alpha * r[t-1]^2 + beta * sigma[t-1]^2

Where omega is a constant, alpha controls the impact of yesterday's
squared return, and beta controls the persistence of past variance.
When alpha + beta is close to 1, volatility is highly persistent.

Why it fits: this is the gold standard statistical baseline. Every
serious ML paper since 2020 compares against GARCH. If our ML models
cannot beat GARCH, they are not adding real value.

Our implementation: uses the arch library. Fits on percentage-scaled
returns. Forecast converts conditional variance to annualized
volatility. The GARCH model does not use engineered features, it
works directly on returns.

### How they compare

Random Forest: fastest to train (seconds), handles nonlinear feature
interactions, may miss temporal patterns.

LSTM: slowest to train (minutes), captures temporal patterns in
sequences, needs feature scaling.

GARCH: fastest overall (milliseconds), purely statistical, no feature
engineering needed, but limited to linear variance dynamics.

The evaluation shows RMSE, MAE, and R-squared for all three on the
same test period, so you can see which one actually works best on
your data.

---

## 5. Feature Engineering

The 9 features in detail:

1. returns (log returns): log(close[t] / close[t-1]). Additive over
   time, symmetric around zero, and the foundation for all volatility
   calculations.

2. realized_vol: rolling std of returns over 20 days, annualized by
   sqrt(252). The most basic backward-looking volatility measure.

3. ewma_vol: exponentially weighted volatility. Recent observations
   get more weight. Reacts faster to regime changes than simple
   rolling vol. Uses recursion for short series (course concept).

4. rsi: Relative Strength Index. Measures momentum by comparing
   average gains to average losses over 14 days. Range 0-100.

5. atr: Average True Range. The biggest single-day move including
   overnight gaps. Measures absolute dollar volatility.

6. bollinger_position: where price sits inside its Bollinger bands.
   Scaled from -1 to +1. Values near +1 mean the price is stretched
   upward, near -1 means stretched downward.

7. momentum: 10-day rate of change of price, smoothed with a 20-day
   rolling mean computed using a deque (course concept).

8. volume_ratio: today's volume divided by its 20-day average. Volume
   spikes often accompany volatility spikes.

9. day_of_week: integer 0-4. Captures weekly patterns like Monday
   effect or Friday profit-taking.

The target: future realized volatility over the next horizon days.
This is what the models predict. It is the rolling standard deviation
of returns from day t+1 to t+horizon, annualized.

---

## 6. Trading Strategies

### Moving Average Crossover

The simplest trend-following strategy. Compute a fast (10-day) and
slow (50-day) moving average of closing price. When the fast average
is above the slow one, the trend is up, so go long. When it drops
below, go flat.

Why it works: trends persist in financial markets. The moving average
crossover captures the middle of the trend, missing the exact top and
bottom but staying in for the bulk of the move.

Why it sometimes fails: in sideways markets, the averages cross back
and forth, generating many small losing trades. This is called
whipsaw.

### Volatility Breakout

Markets that were quiet often start a strong move when they break out
of their range. The strategy computes Bollinger bands (mean +/- 2
standard deviations). When price closes above the upper band, it is
breaking upward with expanding volatility, so go long. When price
falls back below the middle band, the breakout failed, so go flat.

Why it works: volatility expands after compression. The breakout
strategy catches the beginning of that expansion.

Why it sometimes fails: false breakouts. Price pokes above the band
but immediately reverses. The strategy loses on the reversal.

---

## 7. Backtester

### Simulation Loop

The engine processes one bar at a time. On each day it:
1. Asks the strategy what position to hold
2. If different from the current position, executes a trade
3. Deducts commission on both buy and sell
4. Records the account value (cash + shares * price)

This is realistic because you cannot trade before the signal is
generated, and you pay costs on every trade.

### Commission Model

A flat fraction of the trade value, default 0.1% per side. So a
$10,000 trade costs $10 in commission. Both the buy and the sell
are charged. This is realistic for most brokers.

### Volatility-Based Position Sizing

When the forecast says volatility is high, the strategy sizes down.
The formula is:

    position_size = budget * min(target_vol / forecast_vol, 1.0)

If target_vol is 15% and forecast_vol is 30%, the position is half
the normal size. If forecast_vol is below target, full position is
used. This directly connects the forecasting output to the trading
decision.

### Metrics

Sharpe ratio: return per unit of risk. The single most important
number for comparing strategies.

Max drawdown: worst peak-to-trough loss. Tells you how much pain
you would have endured.

Win rate: percentage of trades that were profitable. Higher is better
but not everything (a strategy with 40% win rate but large wins can
beat one with 60% win rate and small wins).

Profit factor: gross profits divided by gross losses. Above 1 means
the strategy makes money overall.

Total return: how much the account grew or shrank.

### Multiprocessing

run_multiple_backtests runs a grid of parameter combinations. On
Windows, each worker is a separate process (ProcessPoolExecutor)
because Python's multiprocessing uses spawn on Windows. The worker
function _run_one must be at module level so it can be pickled.

---

## 8. API Endpoints

All endpoints are prefixed with /api.

### Health Check
    GET /api/health
    Returns: {"status": "ok", "service": "volatility-forecaster"}

### Fetch Data
    POST /api/data/fetch
    Body: {"symbols": ["AAPL", "BTC-USD"]}
    Fetches from Yahoo Finance (or CSV fallback) and stores in DB.

### Get Prices
    GET /api/data/AAPL
    Optional: ?start=2021-01-01&end=2024-12-31
    Returns all stored OHLCV bars for the symbol.

### Get Features
    GET /api/features/AAPL
    Optional: ?horizon=5
    Returns the engineered feature rows.

### Forecast Volatility
    POST /api/forecast
    Body: {"symbol": "AAPL", "model_name": "random_forest", "horizon": 5}
    model_name can be random_forest, lstm, or garch.
    Returns next forecast, metrics, test predictions and true values.

### Explain Model
    GET /api/explain/AAPL
    Optional: ?horizon=5
    Returns SHAP feature importance and top contributions for the
    most recent prediction. Only works for random_forest.

### List Forecasts
    GET /api/forecasts/AAPL
    Optional: ?model_name=lstm
    Returns all stored forecasts for the symbol.

### Run Backtest
    POST /api/backtest
    Body: {
      "symbol": "AAPL",
      "strategy": "moving_average",
      "params": {"fast": 10, "slow": 50},
      "config": {
        "initial_capital": 100000,
        "commission": 0.001,
        "target_volatility": 0.15,
        "forecast_volatility": 0.25
      }
    }
    Returns run_id, metrics, equity curve, and trades.

### List Backtests
    GET /api/backtests
    Optional: ?symbol=AAPL
    Returns all stored backtest runs, newest first.

### Get One Backtest
    GET /api/backtests/1
    Returns one backtest run with all its trades.

---

## 9. Dashboard

Four tabs in the Streamlit app:

### Prices Tab
Line chart of closing price. Below that, realized and EWMA volatility
lines. Shows the raw data and its volatility characteristics.

### Volatility Forecast Tab
Pick a model (random_forest, lstm, or garch) and a horizon. Click
"Run forecast". Shows three metric cards (next forecast, RMSE, R-
squared) and a line chart comparing predicted vs actual volatility
on the test period.

### Backtest Tab
Pick a strategy and parameters in the sidebar. Click "Run backtest".
Shows four metric cards (return, Sharpe, drawdown, win rate), an
equity curve chart, full metrics in an expander, and a trades table.

### About Tab
Brief project description.

The sidebar also has a "Load sample data" button for offline use.
It generates the bundled CSV data and loads it into the database.

---

## 10. Database

Four tables:

### prices
    id (PK)
    symbol (indexed)
    date (indexed)
    open, high, low, close, volume
    source
    UNIQUE(symbol, date)

### volatility_forecasts
    id (PK)
    symbol (indexed)
    model_name
    horizon_days
    forecast_date
    predicted_volatility
    actual_volatility
    created_at

### backtest_runs
    id (PK)
    symbol (indexed)
    strategy_name
    params (JSON)
    metrics (JSON)
    created_at

### backtest_trades
    id (PK)
    run_id (FK -> backtest_runs.id)
    date
    action (BUY or SELL)
    price
    size
    pnl

The relationship: one BacktestRun has many BacktestTrades. When you
load a run, trades are eagerly loaded with selectinload so the
session can be closed immediately.

---

## 11. Testing

64 test cases across 6 files:

test_data.py (9 tests): cleaning, CSV loading, database roundtrip,
async fetching, retry decorator, caching decorator.

test_features.py (8 tests): log returns, realized volatility,
recursive EMA, deque rolling mean, feature building, LSTM sequences.

test_models.py (14 tests): random forest predictions, time split,
evaluator metrics, LSTM training, GARCH fit/predict, SHAP
explainability, full pipeline for all three models.

test_strategies.py (8 tests): abstract class enforcement, MA crossover
logic, parameter validation, volatility breakout logic.

test_backtester.py (11 tests): backtest results, error handling,
flat strategy, generator type, metrics, grid runs, vol sizing (3
tests for high vol, low vol, and disabled).

test_api.py (12 tests): all API endpoints through the Flask test
client. Health, fetch, prices, backtest flow, forecast, forecasts
list, 404 handling.

All tests run offline on seeded synthetic data against an isolated
temporary SQLite database. No PostgreSQL required.

---

## 12. CI/CD

GitHub Actions workflow at .github/workflows/ci.yml.

Triggers on: push to main, pull requests to main.

Matrix: Python 3.12 and 3.14 on ubuntu-latest.

Steps:
1. Checkout code
2. Set up Python
3. Install dependencies from requirements.txt
4. Run pytest
5. Import sanity check (verify main modules import cleanly)

The CI badge in the README shows the current status (green when passing).

---

## 13. Research Context

### The Three Research Streams

1. Statistical models (GARCH family): have been the standard for
   decades. Good at capturing volatility clustering. Limitation:
   linear in nature, miss complex patterns.

2. Machine learning models (Random Forest, LSTM): treat volatility as
   a supervised learning problem. Can capture nonlinear patterns.
   Limitation: need enough data, risk of overfitting.

3. Strategy backtesting: focus on turning forecasts into trading
   decisions. Emphasize transaction costs, risk metrics, and honest
   evaluation. Limitation: assumes signals already exist.

### The Gap

Most projects focus on only one piece. A forecasting paper does not
turn the forecast into a strategy. A backtesting paper assumes you
already have good signals. This project joins all three into one
working system.

### Our Tier 1 Improvements

GARCH baseline: provides the statistical benchmark that ML models
must beat to prove value. Without it, our results are not credible
in any academic context.

SHAP explainability: shows which features drive predictions. Provides
the trust layer that regulators increasingly demand. Without it,
the model is a black box.

Volatility-based position sizing: connects the forecasting output
to the trading decision. When the model says volatility is high,
the backtester trades less. This is the natural risk management
approach that professional traders use.

---

## 14. Future Scope

### Tier 2 (do next)

Walk-forward validation: instead of a single train/test split, retrain
the model on a rolling window and retest. This avoids overfitting and
is the standard in academic ML for finance.

Regime detection: add an HMM that detects calm vs volatile markets.
Switch strategy behavior based on the detected regime. Markets in
crisis behave differently than markets in calm trends.

Transformer model: replace or supplement the LSTM with an attention-
based model. Transformers capture long-range dependencies better
than LSTMs.

Sentiment analysis: add news or social media sentiment as a feature.
Research shows sentiment has measurable predictive power for
volatility, especially during market stress.

### Tier 3 (future work)

Multi-asset portfolio: backtest across multiple assets simultaneously.
Diversification reduces drawdown.

Real-time feeds: scheduled jobs that re-fetch prices and re-run
forecasts. A "last updated" timestamp on the dashboard.

Stop-loss and take-profit: risk management rules that close positions
when losses exceed a threshold.

Alternative data: on-chain metrics for crypto, options-implied
volatility (VIX), macroeconomic indicators.

---

## 15. How to Run Everything

### Setup

    git clone https://github.com/rajit2004/FinTech-VolatilityForecasting-AlgorithmicStrategyBacktester.git
    cd FinTech-VolatilityForecasting-AlgorithmicStrategyBacktester
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

### Database

    CREATE ROLE volforecaster WITH LOGIN PASSWORD 'your_password';
    CREATE DATABASE volforecaster OWNER volforecaster;
    cp .env.example .env
    # Edit .env with your password

### Load Data (Offline)

    python scripts/fetch_and_seed.py

### Run API

    python -m app.main
    # Server starts at http://127.0.0.1:5000

### Run Dashboard

    streamlit run dashboard/app.py
    # Opens at http://localhost:8501

### Run Tests

    python -m pytest -v

### CLI Commands

    python -m app.main forecast --symbol AAPL --model garch
    python -m app.main backtest --symbol AAPL --strategy moving_average
    python -m app.main backtest --symbol AAPL --strategy volatility_breakout --target-vol 0.15 --forecast-vol 0.25
    python scripts/demo_backtest.py --symbol AAPL

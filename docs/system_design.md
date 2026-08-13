# System Design

## Problem Statement

Volatility, the size of price swings in a market, is one of the most
studied and most important quantities in finance. High volatility means
high risk and often high opportunity. Being able to forecast how volatile
a stock or crypto asset will be in the coming days is valuable to anyone
who trades, yet volatility appears unpredictable to beginners. At the
same time, testing a trading strategy against historical data is the only
honest way to know whether an idea has any value before risking real
money, but doing this by hand is slow and error prone.

This project builds one system that does both: it forecasts volatility
using machine learning and it backtests algorithmic trading strategies
against historical data, presenting everything through a web API and a
dashboard.

## Objectives

- Collect historical price data for a set of stocks and crypto assets
  from a live source, with an offline fallback.
- Clean and store that data in a database for reuse.
- Engineer features from prices and compute realized volatility.
- Forecast future volatility with a scikit-learn random forest and a
  Keras LSTM, and compare them with standard metrics.
- Implement two trading strategies, a moving average crossover and a
  volatility breakout.
- Backtest the strategies with transaction costs and report Sharpe
  ratio, maximum drawdown, win rate and total return.
- Expose all of this through a REST API.
- Show forecasts and backtest results on a simple dashboard.
- Make the whole pipeline reproducible and testable with automated tests.

## Literature Review

Three research areas informed the design. Statistical models in the
GARCH family model volatility clustering directly and remain the
baseline for short horizon forecasts. Machine learning research treats
volatility as a supervised prediction problem and shows that tree models
and recurrent networks can capture nonlinear patterns. Backtesting
research emphasizes transaction costs and risk adjusted metrics when
judging a strategy. Details are covered in research_gap.md, but the key
takeaway is that the three areas are usually studied separately.

## Research Gap

Published work tends to focus on one piece of the puzzle: improving a
forecast model, or refining a backtesting methodology, or testing a
strategy. Few projects tie all of it together into a single reproducible
tool with both classical ML and deep learning forecasting, strategies
that trade on volatility itself, honest backtesting, and a way to view
the results. This project targets exactly that gap.

## Proposed System

The system is a layered application:

- A data layer fetches, cleans and stores OHLCV prices.
- A feature layer turns prices into model inputs and computes realized
  volatility.
- A model layer trains a random forest and an LSTM to forecast future
  volatility.
- A strategy layer defines the trading rules.
- A backtest layer simulates trading and computes performance metrics.
- An API layer exposes everything over HTTP.
- A dashboard layer visualizes prices, forecasts and backtests.

## System Architecture

The architecture is a simple modular design where each package only talks
to the packages below it. The API and dashboard are the only two entry
points a user interacts with, and both read from the same database.

```
                        +--------------------------------+
                        |          Dashboard            |
                        |    (Streamlit, reads DB)      |
                        +--------------------------------+
                                       |
        +------------------------------------+------------+
        |            Flask REST API                     |
        |   /data, /features, /forecast, /backtest      |
        +------------------------------------+------------+
              |                |                |
      +-------+--------+  +---+----+   +-------+--------+
      |  Models pkg    |  |Features|   |  Strategies pkg |
      |  RF + LSTM     |  |  pkg   |   | MA + Breakout   |
      +----------------+  +---+----+   +--------+--------+
                               |                 |
      +-------------------------v-----------------v----------+
      |                  Backtester pkg                       |
      |                  engine + metrics                     |
      +-------------------------+-----------------------------+
                               |
      +-------------------------v---------------+            |
      |              Data pkg                   |            |
      |   fetcher -> cleaner -> database        |            |
      +-------------------------+---------------+            |
                               |                            |
                      +--------v---------+          +--------v---------+
                      |  PostgreSQL DB   |          |  dataset CSVs    |
                      | (prices, runs)   |          | (offline fallback|
                      +------------------+          +------------------+
```

### How data flows

- Prices come from Yahoo Finance through yfinance, or from bundled CSV
  files when offline.
- The cleaner normalizes the frame (sorting, deduplicating, filling
  gaps).
- Clean frames are stored in PostgreSQL via SQLAlchemy.
- The feature package computes returns, realized volatility, EWMA
  volatility, RSI, ATR, Bollinger position, momentum and calendar
  features.
- The models train on those features and produce next period volatility
  forecasts that are stored back in the database.
- Strategies produce position signals, the backtest engine turns them
  into trades and an equity curve, and the metrics are stored per run.

## System Flowchart

1. User requests data for a symbol (via API, CLI or dashboard).
2. The fetcher tries Yahoo Finance, and falls back to bundled CSV if the
   network call fails.
3. The cleaner validates and normalizes the frame.
4. If prices already exist in the database for that range, the stored
   copy is returned instead of downloading again (save_prices skips
   duplicates).
5. When a forecast is requested, features are built and the train/test
   split is done in time order.
6. The chosen model is trained on the training portion and evaluated on
   the test portion.
7. A one step ahead forecast is computed and stored with its metrics.
8. When a backtest is requested, the chosen strategy generates signals
   for the whole history.
9. The engine simulates the trades day by day, charging commission and
   tracking cash and shares.
10. Metrics are computed from the equity curve and trades, then stored.
11. The API returns JSON and the dashboard renders it.

## Module Design

### app/data

- fetcher.py: downloads from Yahoo, falls back to CSV, demonstrates
  asyncio for concurrent multi ticker downloads and generators for
  streaming CSV rows.
- cleaner.py: validates symbols and normalizes frames.
- database.py: SQLAlchemy models for prices, forecasts and backtests,
  plus save and load helpers.

### app/features

- engineering.py: all feature computation including realized and EWMA
  volatility, RSI, ATR, Bollinger position, momentum and the forward
  looking target. Uses deques for rolling windows and recursion for the
  EMA calculation.

### app/models

- volatility_model.py: the random forest wrapper and the Keras LSTM
  wrapper, plus the end to end training pipeline.
- evaluator.py: RMSE, MAE and R squared metrics and a model comparator.

### app/strategies

- base_strategy.py: abstract base class every strategy extends.
- moving_average.py: fast/slow crossover signals.
- volatility_breakout.py: Bollinger band breakout signals.

### app/backtester

- engine.py: the bar by bar simulation, trade execution, and a
  multiprocessing helper for parameter sweeps.
- metrics.py: total return, CAGR, Sharpe, max drawdown, win rate,
  profit factor.

### app/api

- routes.py: the Flask endpoints.
- schemas.py: request validation helpers.

### app/utils

- logger.py: centralized logging setup.
- decorators.py: timing, retry and caching decorators.

## Database Design

The database is PostgreSQL managed by SQLAlchemy. Four tables store
everything the system produces. The connection string is read from the
.env file, so pointing the app at another database is just a config
change.

### prices

| Field  | Type    | Notes                          |
|--------|---------|--------------------------------|
| id     | int PK  | auto increment                 |
| symbol | string  | indexed, e.g. AAPL             |
| date   | date    | indexed                        |
| open   | float   |                                |
| high   | float   |                                |
| low    | float   |                                |
| close  | float   |                                |
| volume | float   |                                |
| source | string  | yahoo or csv                   |

Unique constraint on (symbol, date) prevents duplicate rows.

### volatility_forecasts

| Field               | Type    | Notes                   |
|---------------------|---------|-------------------------|
| id                  | int PK  |                         |
| symbol              | string  | indexed                 |
| model_name          | string  | random_forest or lstm   |
| horizon_days        | int     | forecast horizon        |
| forecast_date       | date    | date the forecast is for|
| predicted_volatility| float   | the model output        |
| actual_volatility   | float   | filled in later, can be null |
| created_at          | datetime| when the row was saved  |

### backtest_runs

| Field         | Type    | Notes                        |
|---------------|---------|------------------------------|
| id            | int PK  |                              |
| symbol        | string  | indexed                      |
| strategy_name | string  | moving_average or breakout   |
| params        | JSON    | strategy parameters          |
| metrics       | JSON    | total return, sharpe, etc.   |
| created_at    | datetime|                              |

### backtest_trades

| Field   | Type    | Notes                            |
|---------|---------|----------------------------------|
| id      | int PK  |                                  |
| run_id  | int FK  | references backtest_runs.id      |
| date    | date    |                                  |
| action  | string  | BUY or SELL                      |
| price   | float   |                                  |
| size    | int     | shares                           |
| pnl     | float   | profit on a closed round trip    |

backtest_runs has a one to many relationship with backtest_trades,
deleting a run deletes its trades.

## Algorithms

### Volatility forecasting

Target: the annualized realized volatility over the next `horizon` days,
computed as the rolling standard deviation of daily log returns scaled by
the square root of 252.

For the random forest, the model receives one row per day with nine
features and returns the predicted volatility number. Training and test
data are split in time order (80/20) so the model is always tested on
data newer than what it trained on.

For the LSTM, the input is a sequence of the last 30 days of features at
each point in time, shaped as (samples, 30, 9). The LSTM layer reads the
whole window and a final dense layer outputs the volatility prediction.
Features and targets are scaled to 0..1 before training because neural
networks train much better on bounded inputs, and predictions are scaled
back to the original units afterwards.

Both models are compared with RMSE, MAE and R squared on the same test
period.

### Backtesting

Signals from the strategy are computed for every day. The engine holds
cash and shares. When the desired position changes from flat to long, it
buys shares worth a fixed fraction of the account at the day's close,
deducting commission. When the position returns to flat, it sells all
shares at the day's close, again paying commission. Equity on each day is
cash plus the value of held shares. The metrics module then computes the
final results from the equity curve and the list of trades.

## Technology Stack

- Python 3, pandas and NumPy for data handling.
- scikit-learn for the random forest model.
- Keras 3 with the JAX backend for the LSTM model.
- SQLAlchemy and PostgreSQL for storage.
- Flask for the REST API.
- Streamlit for the dashboard.
- yfinance for live price downloads.
- pytest for automated testing.

Note on the deep learning backend: TensorFlow does not ship wheels for
Python 3.14 yet, so the project uses Keras 3 with the JAX backend. The
Keras API is identical either way.

## Implementation notes

- All configuration goes through a .env file loaded by python-dotenv,
  no secrets or settings are hardcoded.
- Logging replaces print statements for anything that matters.
- Functions are type hinted and small, one responsibility each.
- Data fetching, modeling, strategy and backtest logic live in separate
  packages.
- The bundled CSV generator makes the project fully reproducible offline.

## Testing approach

The test suite runs entirely offline. Synthetic price data is generated
with a fixed random seed, so every run is identical. The Flask app is
tested with its built in test client against an isolated temporary
database. The LSTM test uses a single epoch and a small network to keep
the suite fast. Tests cover normal inputs, invalid inputs, missing data
and edge cases across the data, features, models, strategies, backtester
and API modules.

## Results summary

Representative results are covered in more detail in final_report.md.
In short, on the sample data the random forest usually achieved a lower
RMSE than the LSTM with far less training time, which is a common outcome
when the feature set is small. Both strategies produced positive returns
over the test range on several symbols, and the volatility breakout
strategy tended to trade less but with a higher win rate than the moving
average crossover. Because the data is synthetic, these numbers illustrate
the workflow rather than prove any real world edge.

## Future Scope

- Use longer and real market data, and add a proper GARCH baseline for
  comparison.
- Add more forecasting models, for example gradient boosting and
  attention based networks.
- Add risk management rules such as stop losses, position sizing based on
  forecast volatility, and leverage constraints.
- Add live or real time price feeds and recharging forecasts on new bars.
- Add connection pooling and read replicas when the data set grows large.
- Extend the dashboard with interactive parameter sweeps and model
  comparison charts.
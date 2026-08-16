# User Manual

## Welcome

This manual is written for someone who has never seen this codebase before. Maybe that someone is you, maybe it is a classmate, maybe it is your future self a few months from now. Either way, by the end of this file you should understand what this project does, why it exists, and how every piece of the code fits together.

You do not need to be an expert in finance or machine learning to read this. Wherever a concept could be confusing, I explain it in plain words first.

---

## Part 1: What this project is

### The short version

This is a capstone project about financial markets. It does three things:

1. It collects historical price data for stocks and cryptocurrencies.
2. It uses machine learning to predict how volatile a market will be over the next few days.
3. It tests simple trading strategies against that history to see how well they would have performed.

All of that is wrapped in a web dashboard and a REST API, so you can click around and see results without writing code.

### The slightly longer version

If you follow financial news, you will hear the word "volatility" a lot. It means, roughly, how much a price bounces around. A stock that barely moves day to day has low volatility. A stock that swings 5 percent in a single afternoon has high volatility.

Volatility matters to anyone who trades, because it is a measure of risk. If you know volatility is about to spike, you can change how much you invest, or step aside entirely. So a tool that can forecast volatility ahead of time has real value.

This project tries to do exactly that. It takes daily price history, turns it into a set of features (things like how fast price has been moving, whether the market is overbought or oversold, how much volume there is), and then trains machine learning models to predict future volatility.

Then, separately, it asks a different question: if a simple rule based strategy had been running on this history, how much money would it have made? That is called a backtest. We simulate the strategy day by day, account for costs, and measure the outcome.

### The three big ideas

To make this easy to hold in your head, think of the project as three layers sitting on top of each other:

- **Data.** Nothing works without clean prices. This layer fetches data, cleans it up, and stores it in a database.
- **Analysis.** Two kinds of analysis run on the data. One forecasts volatility using machine learning. The other simulates trading strategies.
- **Presentation.** The results are shown in a web dashboard and exposed through an API so other programs can talk to it too.

Every folder in this project belongs to one of those layers. Keep that in mind and the folder structure will feel obvious.

---

## Part 2: The technology and why it was chosen

A quick note on the tools, because a few choices were deliberate.

- **Python** is the language everything is written in. It is the standard choice for data work because of its libraries.
- **pandas** handles the tabular data. Almost every table in the project is a pandas DataFrame, which is basically a spreadsheet in memory.
- **scikit-learn** provides the random forest model. Random forests are simple, robust, and great baselines.
- **Keras 3 with the JAX backend** provides the LSTM model. An LSTM is a kind of neural network designed for sequences of data, which fits daily price history well. JAX is used because TensorFlow does not ship installers for Python 3.14 yet, and Keras 3 lets us swap the backend without changing any model code.
- **Flask** is the web framework behind the REST API.
- **Streamlit** is the dashboard framework. You write a Python script and it becomes an interactive web page.
- **PostgreSQL** is the database. All prices, forecasts, and backtest results live there.
- **SQLAlchemy** is the bridge between Python and the database. It lets us treat database rows like normal Python objects.
- **pytest** runs the test suite, which checks that nothing breaks when we change code.

---

## Part 3: How data flows through the app

Before looking at individual files, trace one full journey: a forecast for Apple.

1. We ask for prices for AAPL. The data layer tries Yahoo Finance first. If the network is unavailable, it falls back to bundled CSV files in the `dataset` folder.
2. The raw data is cleaned: duplicate dates are removed, missing values are filled, rows are sorted by date.
3. The clean prices are stored in the `prices` table in PostgreSQL, so we do not need to download again.
4. The features module turns prices into model inputs. Things like rolling volatility, RSI, and ATR are computed and put into a feature table.
5. A model (random forest or LSTM) is trained on the older part of the data and tested on the newest part. We measure how wrong the predictions are using RMSE, MAE, and R squared.
6. The trained model makes one last prediction for the days right after the last known price. That number is the forecast.
7. The forecast is saved to the `volatility_forecasts` table and returned as JSON.
8. The dashboard, or any client, reads that JSON and draws it on screen.

The backtest journey is similar but replaces steps 4 to 6 with a strategy simulation:

1. Same price loading and cleaning.
2. A strategy (moving average crossover or volatility breakout) looks at the history and says "on each day, should we be in the market or out of it?"
3. The backtest engine walks through day by day, buys when the strategy says go long, sells when it says get out, and charges a small commission each time.
4. Metrics are computed: total return, Sharpe ratio, max drawdown, win rate, and more.
5. The run and its individual trades are saved to the database and returned to the caller.

That is the whole story in one paragraph. Now let us open the code and see where each step lives.

---

## Part 4: The folder structure

```
.
|-- app/
|   |-- __init__.py          marks the folder as a Python package
|   |-- main.py              the entry point, starts the API or runs CLI commands
|   |-- config.py            all settings in one place
|   |-- api/                 the REST endpoints and input validation
|   |-- backtester/          the backtesting engine and its metrics
|   |-- data/                fetching, cleaning, and database access
|   |-- features/            turning prices into model features
|   |-- models/              the forecasting models and evaluation
|   |-- strategies/          the trading strategies
|   `-- utils/               logging and decorator helpers
|-- dashboard/app.py         the Streamlit web dashboard
|-- docs/                    research and design documents
|-- scripts/                 helper scripts for setup and demos
|-- tests/                   the pytest test suite
|-- notebooks/               an exploration notebook
|-- dataset/                 bundled CSV fallback data (gitignored)
|-- .github/workflows/       the CI pipeline that runs tests on GitHub
|-- requirements.txt         the list of Python packages
|-- .env.example             a template for your environment settings
`-- LICENSE, README.md       the legal bits and the project readme
```

In the next parts we go through each of these, starting with the boring plumbing and working up to the interesting models.

---

## Part 5: Configuration and utilities

### app/config.py: the settings hub

Every value that could change between machines lives here. Database URL, which symbols to track, date ranges, model settings, all of it.

The clever bit is how it reads a `.env` file. You copy `.env.example` to `.env`, fill in your own values, and the config reads them with a library called python-dotenv. If the `.env` file is missing, the code falls back to sensible defaults. That means a fresh clone can run without configuration, and a real deployment can override anything without touching source code.

The database URL is the one setting that actually matters for running. The default points at a local PostgreSQL database. The password in the committed files is a placeholder, so on a new machine you create the role yourself and put the real password in `.env`.

### app/utils/logger.py: one consistent logger

Instead of sprinkling `print()` everywhere, the app logs through Python's logging module. This file configures it once with a readable format that includes the time, the log level, and the module name. Every other module calls `get_logger(__name__)` and gets a ready to use logger. You see these messages in the terminal while the API or dashboard runs.

### app/utils/decorators.py: three reusable wrappers

Decorators are a Python feature that lets you wrap extra behavior around a function without editing it. This module has three, and they are used in a few places:

- `time_it` logs how long a function took. Handy for expensive things like training a model.
- `retry` retries a function a few times if it fails, waiting longer between tries. It is used on the network download, because Yahoo Finance sometimes drops a request.
- `cached` remembers the result of a call and returns it instantly if the same arguments come in again. Useful for expensive computations that repeat.

---

## Part 6: The data layer

This is the bottom of the stack. Three files live here.

### app/data/fetcher.py: where prices come from

The primary source is Yahoo Finance, through the yfinance library. The important function is `fetch_ohlcv`, which tries Yahoo first. If that fails (offline, bad ticker, rate limit), it falls back to a bundled CSV file for that symbol in the `dataset` folder. This fallback is why the whole project works on a laptop with no internet.

Two implementation details are worth noticing, because they are course concepts:

- `fetch_many_async` downloads several tickers at the same time using `asyncio`. Downloads are slow and mostly waiting on the network, so doing them concurrently is much faster than one at a time. A semaphore limits how many run at once so we do not trip rate limits.
- `stream_csv` is a generator. It yields rows one by one instead of loading a whole file into memory. For long histories that keeps memory flat.

### app/data/cleaner.py: fixing messy data

Raw market data is never perfect. A holiday might leave a missing row, a date might appear twice, a price might be zero or NaN. `clean_ohlcv` fixes all of it:

- keeps only the columns we care about, in a known order
- drops rows that are completely empty
- makes the date the index and removes duplicate dates
- sorts chronologically
- forward fills small gaps (a quiet day missing a price usually just copies the last value)
- drops whatever is still broken

It also has `validate_symbol`, which uppercases and strips ticker symbols, and rejects anything empty or containing spaces.

### app/data/database.py: talking to PostgreSQL

This is the biggest file in the data layer. It uses SQLAlchemy, and it defines four tables as Python classes:

- `Price`: one row per symbol per date, with open, high, low, close, and volume.
- `VolatilityForecast`: one row per forecast, with the model name, horizon, date, and predicted value.
- `BacktestRun`: one row per backtest, storing the strategy, its parameters, and the metrics as JSON.
- `BacktestTrade`: one row per trade in a backtest, linked to its run.

The helper functions do the obvious things: `save_prices`, `load_prices`, `save_forecast`, `load_forecasts`, `save_backtest`, `load_backtest_runs`, and `load_backtest_run`.

A few details are worth knowing:

- `configure_database(url)` lets callers point the whole module at a different database. The tests use this to run against a temporary file, and the app uses it to honor a custom `DATABASE_URL`.
- `init_db()` creates the tables if they do not exist. It is safe to call on every startup.
- `save_prices` skips dates that already exist, so running it twice never creates duplicates.
- `iter_price_chunks` is a generator that streams prices in chunks of a few hundred rows, another way the project keeps memory usage flat.

---

## Part 7: Feature engineering

### app/features/engineering.py: turning prices into numbers a model can learn from

A model cannot learn much from a raw price. A share at 30 dollars and a share at 300 dollars need different treatment, and the same dollar move means nothing on both. So we transform prices into relative features, the same way a human analyst would.

The main entry point is `build_features`, which takes clean OHLCV data and adds these columns:

- `returns`: daily log returns. Log returns are nice because they add up over time.
- `realized_vol`: rolling standard deviation of returns, scaled up to a yearly number.
- `ewma_vol`: an exponentially weighted version of volatility, which reacts faster to recent moves.
- `rsi`: the Relative Strength Index, a 0 to 100 oscillator measuring whether a market is overbought or oversold.
- `atr`: Average True Range, a common measure of price movement in dollars.
- `bollinger_position`: where the price sits inside its Bollinger bands, from -1 to 1.
- `momentum`: a rolling mean of the rate of change, computed with a deque.
- `volume_ratio`: how today's volume compares to its recent average.
- `day_of_week`: a number for the weekday, so the model can learn weekday effects.

There is also a `target` column: the realized volatility over the next `horizon` days. This is what the model is trying to predict.

The file demonstrates three course concepts:

- `recursive_ema` computes an exponential moving average using recursion. The EMA of a list is the last value blended with the EMA of the rest, which is a naturally recursive formula. There is a guard so long series switch to pandas instead of risking Python's recursion limit.
- `rolling_mean_deque` computes a rolling mean with a `deque` of fixed size, the structure you would use in a live feed where old bars drop off automatically.
- `prepare_for_model` drops rows with any missing value and returns the feature array X, the target array y, and the feature names. The rows at the start are missing because rolling windows need warm up time, and the rows at the end are missing because the target looks forward.

`make_lstm_sequences` builds the windows the LSTM needs: for each position, a block of the last `sequence_length` rows as input, with the target at the end of that block as the output.

---

## Part 8: The models

### app/models/volatility_model.py: the two forecasters

The models share a common interface: `fit(X, y)` and `predict(X)`. That lets the rest of the app treat them interchangeably.

**RandomForestVolatilityModel.** A random forest regressor from scikit-learn. Random forests combine many decision trees, each trained on a random subset of the data, and average their answers. They handle non linear relationships well and need no feature scaling, which makes them an excellent baseline.

**LSTMVolatilityModel.** A Keras LSTM. LSTMs are recurrent neural networks that keep a memory of past inputs, which suits volatility because volatility has a strong memory effect (big moves tend to follow big moves). The model reads a window of past rows, passes them through an LSTM layer, then a couple of dense layers, and outputs one number: the predicted volatility. Features and targets are scaled to 0 to 1 before training because neural networks train much better on normalized inputs, and scaled back afterward.

**run_volatility_forecast** is the pipeline function. It:

1. builds the features
2. splits the data in time order, so the model trains on the past and is tested on the future (never a random shuffle, that would leak the future into training)
3. trains the chosen model
4. evaluates it on the held out test set
5. predicts one step ahead, for the period right after the last known day

It returns a dict with the predictions, the true values, the test dates, the metrics, and the next forecast. That dict flows straight into the API and the dashboard.

One subtle thing worth knowing: the LSTM needs the test dates aligned with its predictions, and sequences end a few rows before the end of the data. The code accounts for that so the chart always has matching date, predicted, and actual values.

### app/models/evaluator.py: how good are the predictions

Three metrics compare the predictions to reality:

- **RMSE** (root mean squared error): squares the errors before averaging, so big misses are punished extra hard. That matters when one missed volatility spike can hurt a strategy.
- **MAE** (mean absolute error): the plain average distance between predictions and truth.
- **R squared**: how much of the variance the model explains. Near 1 is great, near 0 means the model is no better than guessing the average.

`compare_models` picks the model with the lowest RMSE, which is useful for reporting which forecaster did better.

---

## Part 9: The trading strategies

### app/strategies/base_strategy.py: the blueprint

Every strategy produces one thing: a signal for each day saying what position we should hold. The position is +1 for long, 0 for flat, or -1 for short. The backtest engine consumes those signals and turns them into simulated trades.

Making this an abstract base class means every strategy is guaranteed to expose a `generate_signals(df)` method with the same input and output, so the engine can run any strategy without knowing how it works inside.

### app/strategies/moving_average.py: the trend follower

This is the classic moving average crossover. Compute a fast moving average and a slow moving average of the closing price.

- When the fast average is above the slow average, the trend is turning up, so we go long.
- When it is below, the trend is turning down, so we go flat (or short, if `allow_short` is on).

It is one of the simplest strategies that still works in practice, which makes it a fair benchmark.

### app/strategies/volatility_breakout.py: the range breaker

The idea behind breakout trading is that a market trading quietly often starts a strong move when it finally breaks out of its range. This strategy uses Bollinger bands, which are a rolling mean plus and minus a few standard deviations.

- When the close punches above the upper band, we go long, betting the breakout continues.
- When the close falls back below the middle line, we go flat, betting the move is over.

This strategy leans on volatility itself, which ties nicely into the volatility forecasting half of the project.

---

## Part 10: The backtesting engine

### app/backtester/engine.py: simulating a strategy day by day

The engine walks through history one bar at a time. Each day it asks the strategy what position it wants. If that is different from the position we currently hold, it executes a trade:

- Going from flat to long means buying shares with most of our cash, paying a small commission.
- Going from long to flat means selling everything and booking the round trip profit or loss.

It tracks cash, shares, and total account value every day, and records the equity curve.

Two course concepts live in this file:

- `iter_bar_events` is a generator that yields one event per bar, so a long history can be simulated without building every intermediate object at once.
- `run_multiple_backtests` runs a grid of parameter combinations across multiple CPU cores using a process pool. Trying out many combinations serially would be far too slow, so each worker gets one combination.

The `BacktestConfig` dataclass holds the simulation settings: initial capital, commission rate, what fraction of cash to invest, and slippage. The `BacktestResult` dataclass packages up the symbol, strategy, metrics, equity curve, and trades.

### app/backtester/metrics.py: the scoreboard

After the simulation, we need numbers to judge it. `compute_metrics` takes the equity curve and the trade list and returns:

- `total_return`: how much the account grew overall.
- `cagr`: the compound annual growth rate, a normalized yearly figure.
- `sharpe_ratio`: return per unit of risk, annualized. Higher is better.
- `max_drawdown`: the worst peak to trough drop the account suffered.
- `win_rate`: what fraction of closed trades were winners.
- `profit_factor`: gross profits divided by gross losses. When there are no losses at all, it is stored as null, because an infinite number cannot go into a JSON column.
- `num_trades`: how many round trips happened.
- `final_equity` and `volatility`: the ending value and the annualized volatility of returns.

---

## Part 11: The API

### app/api/schemas.py: checking what comes in

Before any request is processed, its payload is validated. This module has small focused functions that take the raw JSON, check it, and either return a clean dict or raise a `ValueError` with a human readable message. The route handlers turn those errors into proper 400 responses.

It checks things like: is the symbol a non empty string with no spaces? Is the model one of the two we support? Is the horizon a positive integer? Is the strategy name known? Are the backtest config numbers sane?

### app/api/routes.py: answering requests

Every endpoint here is a thin layer. It validates the request, calls the right part of the app, and returns JSON. No business logic lives here, it stays in the data, models, and backtester packages.

The endpoints are:

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/api/health` | Liveness check, returns ok when the server is up. |
| POST | `/api/data/fetch` | Downloads prices for a list of symbols and stores them. |
| GET | `/api/data/<symbol>` | Returns stored prices, fetching them on first use. |
| GET | `/api/features/<symbol>` | Returns the engineered features for a symbol. |
| POST | `/api/forecast` | Trains a model and returns a volatility forecast. |
| GET | `/api/forecasts/<symbol>` | Returns stored forecasts for a symbol. |
| POST | `/api/backtest` | Runs a backtest and stores the result. |
| GET | `/api/backtests` | Lists stored backtests, newest first. |
| GET | `/api/backtests/<id>` | Returns one backtest with all of its trades. |

Two helpers are used throughout. `_jsonable` converts numpy and pandas values (like numpy floats and pandas timestamps) into plain JSON friendly values, because Flask cannot serialize them directly. `_frame_to_records` turns a DataFrame into a list of dicts with a plain `date` field, so consumers do not have to parse a separate index.

### app/main.py: the front door

Running `python -m app.main` starts the Flask server on `http://127.0.0.1:5000`. The `create_app` function wires everything together: it configures the database, enables CORS so the dashboard can call the API from a browser, registers the API blueprint under the `/api` prefix, and adds a root page plus friendly error handlers.

The same file doubles as a command line tool. Without the server you can run:

```
python -m app.main fetch --symbols AAPL,MSFT
python -m app.main forecast --symbol AAPL --model random_forest --horizon 5
python -m app.main backtest --symbol AAPL --strategy moving_average
```

These are handy for testing the pipeline quickly without starting a web server.

---

## Part 12: The dashboard

### dashboard/app.py: everything in one page

The Streamlit app is the most user friendly way to interact with the project. You launch it and a browser page opens with a sidebar and four tabs.

The sidebar lets you pick a symbol (from whatever is in the database), a strategy, and its parameters. There is also a button to load sample data, which generates realistic offline data and saves it to the database, a fully offline path.

The four tabs:

- **Prices**: charts the closing price history and the realized and EWMA volatility.
- **Volatility Forecast**: pick a model (random forest or LSTM) and a horizon, hit run, and it trains the model, shows the predicted volatility, the RMSE and R squared, and draws predicted versus actual volatility on the test period.
- **Backtest**: run a strategy and see total return, Sharpe, max drawdown, win rate, the equity curve, the full metrics table, and every trade.
- **About**: a short summary of what the project does.

Because training a model can take a while, the app caches loaded prices with `st.cache_data`, so switching tabs does not re-download or re-query the database on every click.

---

## Part 13: The scripts

Three scripts live in `scripts/`, and they are the fastest way to get up and running without a browser.

- **generate_sample_data.py**: simulates realistic OHLCV data and writes CSV files to the `dataset` folder. The simulation is a random walk with GARCH style volatility clustering, meaning calm periods follow calm periods and wild periods follow wild periods, which is how real markets behave. The seed makes it reproducible: the same seed always produces the same data.
- **fetch_and_seed.py**: the one stop setup script. It generates the sample data and loads it into the database. Run this once and the dashboard and API both have data to show.
- **demo_backtest.py**: runs both strategies on one symbol and prints a summary table comparing them. Great for a quick sanity check that everything works.

---

## Part 14: The tests

### tests/: making sure nothing breaks

The suite uses pytest, and it is split by area: `test_data`, `test_features`, `test_models`, `test_strategies`, `test_backtester`, and `test_api`.

The fixtures in `conftest.py` do the important isolation work:

- `clean_frame` generates deterministic synthetic OHLCV data using the same simulator the app uses offline. Tests never hit the network.
- `configured_db` points the database module at a fresh file in a temp folder, so every test starts with a clean database and never touches your real PostgreSQL data.
- `app_client` gives each API test a Flask test client bound to that isolated database.

The tests cover a lot of ground: cleaning produces a proper index, recursion and deque helpers return the expected values, model outputs have the right shapes, strategies produce the expected signals, backtests produce sensible metrics, and the API handles happy paths plus errors like missing fields and unknown routes. The whole suite is expected to be green.

---

## Part 15: Continuous integration

### .github/workflows/ci.yml: the safety net on GitHub

Every time you push to the main branch, and every time someone opens a pull request, GitHub Actions runs the test suite on two Python versions (3.12 and 3.14). If any test fails, the run turns red and you know immediately.

The workflow is simple: check out the code, set up Python, install the packages from `requirements.txt`, run pytest, and verify the app imports. Because the tests use an isolated database, the CI machine does not need PostgreSQL at all.

There is a badge in the README showing the latest status. Note that it only works when the repository is public, because GitHub hides unauthenticated requests to private repos.

---

## Part 16: Setting up and running the project

Here is the full sequence on a fresh machine.

**1. Get the code and create a virtual environment.**

```
python -m venv .venv
```

Activate it. On Windows:

```
.venv\Scripts\activate
```

On macOS or Linux:

```
source .venv/bin/activate
```

**2. Install the dependencies.**

```
pip install -r requirements.txt
```

**3. Set up the database.**

PostgreSQL needs a user and a database for the project. As the postgres superuser:

```sql
CREATE ROLE volforecaster WITH LOGIN PASSWORD 'CHANGE_ME';
CREATE DATABASE volforecaster OWNER volforecaster;
```

Then copy the environment template and put the real password in it:

```
cp .env.example .env
```

Edit the `DATABASE_URL` line in `.env` to match the role you created. No secrets live in the source code, everything lives in `.env`, which is gitignored.

**4. Load some data.**

The fastest path is the seed script, which works offline:

```
python scripts/fetch_and_seed.py
```

Or fetch live data from Yahoo Finance:

```
python -m app.main fetch --symbols AAPL,MSFT,BTC-USD
```

**5. Run the API.**

```
python -m app.main
```

The API listens on `http://127.0.0.1:5000`. Check it with `http://127.0.0.1:5000/api/health`.

**6. Run the dashboard.**

In a second terminal:

```
streamlit run dashboard/app.py
```

It opens at `http://localhost:8501`.

**7. Run the tests.**

```
python -m pytest -q
```

**8. Try a demo backtest.**

```
python scripts/demo_backtest.py --symbol AAPL
```

---

## Part 17: Example API calls

Here is what each interesting endpoint looks like with curl.

Check the service:

```
curl http://127.0.0.1:5000/api/health
```

Fetch and store prices:

```
curl -X POST http://127.0.0.1:5000/api/data/fetch \
  -H "Content-Type: application/json" \
  -d "{\"symbols\": [\"AAPL\", \"BTC-USD\"]}"
```

Get stored prices:

```
curl http://127.0.0.1:5000/api/data/AAPL
```

Run a forecast with the random forest:

```
curl -X POST http://127.0.0.1:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d "{\"symbol\": \"AAPL\", \"model_name\": \"random_forest\", \"horizon\": 5}"
```

Run a backtest:

```
curl -X POST http://127.0.0.1:5000/api/backtest \
  -H "Content-Type: application/json" \
  -d "{\"symbol\": \"AAPL\", \"strategy\": \"moving_average\", \"params\": {\"fast\": 10, \"slow\": 50}}"
```

List backtests and fetch one:

```
curl "http://127.0.0.1:5000/api/backtests?symbol=AAPL"
curl http://127.0.0.1:5000/api/backtests/1
```

---

## Part 18: Common questions

**Why do the LSTM and the random forest give different numbers?** They are completely different model families. The forest averages many decision trees over the feature table. The LSTM reads sequential windows and keeps a memory. It is normal for them to disagree, and the point of the dashboard is to compare them.

**Why is the LSTM slower?** Neural networks need to see the data many times over (epochs) and every pass is real math. The forest trains in a couple of seconds. The LSTM can take tens of seconds depending on how many epochs you ask for.

**What does a negative R squared mean?** It means the model is doing worse than simply predicting the average. It is not a bug, it just reflects how hard volatility is to predict. The RMSE and MAE tell you the size of the error, which is usually more intuitive.

**Why does the database get a null profit factor?** When a strategy never has a losing round trip, profit factor would be division by zero, an infinite number. PostgreSQL JSON cannot store Infinity, so the code stores null instead.

**Why is the dashboard slow the first time I run a forecast?** Because it is actually training a model on your data. The random forest is quick, the LSTM is not. Streamlit caches prices, so at least the data loading is fast on repeat visits.

**I deleted my database, how do I recover?** Just rerun `python scripts/fetch_and_seed.py`. The tables are recreated on startup and the sample data is regenerated deterministically.

---

## Part 19: Where to look next

If you want to go deeper, the `docs` folder has three design documents: `research_gap.md` explains the academic motivation, `system_design.md` describes the architecture, and `final_report.md` is the overall writeup. The `notebooks` folder has an exploration notebook where you can poke at the data interactively.

The best way to understand this project is to run it. Start the dashboard, click Load sample data, try a forecast with both models, run a backtest with both strategies, and compare what you see. Everything in this manual is easier to remember once you have clicked it yourself.
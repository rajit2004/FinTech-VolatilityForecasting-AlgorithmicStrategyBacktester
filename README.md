<p align="center">
  <span style="font-size: 75px;">📈</span>
</p>

<h1 align="center">FinTech Volatility Forecasting & Algorithmic Strategy Backtester</h1>

<p align="center">
  <strong>Forecast market volatility with machine learning, then backtest algorithmic strategies against real historical data. No PhD required.</strong>
</p>

<p align="center">
  <a href="#-what-is-this-project"><img src="https://img.shields.io/badge/What_It_Is-6DB33F?style=flat-square" alt="What It Is" /></a>
  <a href="#-features"><img src="https://img.shields.io/badge/Features-FF6F61?style=flat-square" alt="Features" /></a>
  <a href="#-how-it-works"><img src="https://img.shields.io/badge/How_It_Works-4FC3F7?style=flat-square" alt="How It Works" /></a>
  <a href="#-tech-stack"><img src="https://img.shields.io/badge/Tech_Stack-FFB74D?style=flat-square" alt="Tech Stack" /></a>
  <a href="#-project-structure"><img src="https://img.shields.io/badge/Structure-81C784?style=flat-square" alt="Structure" /></a>
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-AB47BC?style=flat-square" alt="Quick Start" /></a>
  <a href="#-api-documentation"><img src="https://img.shields.io/badge/API-4DD0E1?style=flat-square" alt="API" /></a>
  <a href="#-contributing"><img src="https://img.shields.io/badge/Contributing-F06292?style=flat-square" alt="Contributing" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/rajit2004/FinTech-VolatilityForecasting-AlgorithmicStrategyBacktester/ci.yml?style=flat-square&label=CI" alt="CI" />
  <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/language-Python_3.14-3776AB?logo=python&style=flat-square" alt="Python" />
  <img src="https://img.shields.io/badge/ml-scikit_learn-F7931E?logo=scikit-learn&style=flat-square" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/dl-Keras_%2B_JAX-D00000?logo=keras&style=flat-square" alt="Keras" />
  <img src="https://img.shields.io/badge/api-Flask-000000?logo=flask&style=flat-square" alt="Flask" />
  <img src="https://img.shields.io/badge/database-PostgreSQL-4169E1?logo=postgresql&style=flat-square" alt="PostgreSQL" />
</p>

---

## 💡 What is this project?

This is a college capstone project that puts the whole quant workflow into one system. Most tutorials stop at fetching data, and many papers stop at forecasting alone. This one goes all the way from raw prices to a tested trading strategy, and it shows the results on a dashboard.

The project does three big things:

1. **Forecasts volatility.** It measures how much prices move, then uses machine learning to predict how much they will move in the coming days.
2. **Backtests strategies.** It runs algorithmic trading rules over historical data with transaction costs and tells you the Sharpe ratio, max drawdown, win rate and total return.
3. **Exposes everything.** Prices, forecasts and backtest results are available through a REST API and a Streamlit dashboard.

### The forecasting models

| Model                 | Type                          | Why it is here                              |
| --------------------- | ------------------------------ | ------------------------------------------- |
| 🌲 **Random Forest**  | scikit-learn regressor         | A solid classical ML baseline that handles nonlinear patterns |
| 🧠 **LSTM**           | Keras 3 (deep learning)        | A recurrent network built for time series   |
| 📊 **GARCH(1,1)**     | arch library (statistical)     | The classic volatility baseline every ML model should beat |

### The trading strategies

| Strategy                     | Logic                                                    |
| ---------------------------- | ------------------------------------------------------- |
| 📈 **Moving Average Crossover** | Buy when the fast average crosses above the slow one    |
| ⚡ **Volatility Breakout**   | Buy when price punches through a rolling Bollinger band |

---

## ✨ Features

* 🧮 **Realized & EWMA Volatility**
  Rolling and exponentially weighted volatility, annualized and ready for the models.

* 🤖 **Three Forecasting Models**
  A random forest, an LSTM and a GARCH baseline, all trained on the same data and compared on the same test period.

* 🔍 **Model Explainability with SHAP**
  SHAP feature importance shows which inputs drive the random forest predictions, so you know the model is not just memorizing noise.

* 📉 **Volatility-Based Position Sizing**
  When the forecast says volatility is high, the backtester automatically reduces the position size to protect capital.

* 🧪 **Backtesting Engine**
  Simulates trades bar by bar with commission, and reports Sharpe ratio, max drawdown, win rate and profit factor.

* 🛢️ **Database Backed**
  Prices, forecasts and backtest runs are all stored in PostgreSQL through SQLAlchemy.

* 🌐 **REST API**
  Every piece of functionality is reachable over HTTP with clean JSON.

* 📊 **Interactive Dashboard**
  Streamlit app to browse prices, run forecasts and inspect backtests without touching code.

* 📴 **Fully Offline Mode**
  Runs on bundled generated sample data when you have no internet. Results are reproducible every time.

* ✅ **Automated Tests**
  A pytest suite that covers data, features, models, strategies, backtesting and the API.

---

## 🧠 How It Works

```text
Data is fetched (Yahoo Finance or bundled CSV)
        |
        v
Cleaner sorts, deduplicates and fills gaps
        |
        v
Prices are stored in PostgreSQL
        |
        v
Feature engineering: returns, realized & EWMA vol, RSI, ATR...
        |
        v
Train / test split in time order (no peeking into the future)
        |
        v
Random forest, LSTM or GARCH forecasts next period volatility
        |
        v
SHAP explains which features drove the forecast
        |
        v
Strategy generates buy / sell signals
        |
        v
Backtester simulates trades with commission and vol-based sizing
        |
        v
Metrics (Sharpe, drawdown, win rate) are stored and shown
        |
        v
Flask API serves JSON, dashboard renders the charts
```

---

## 🛠️ Tech Stack

| Layer                  | Technology                                    |
| ---------------------- | --------------------------------------------- |
| **Language**           | Python 3.14                                   |
| **Data handling**      | pandas, NumPy                                 |
| **Classical ML**       | scikit-learn (RandomForestRegressor)          |
| **Statistical ML**     | arch (GARCH volatility models)                |
| **Model explainability** | shap (SHAP feature importance)              |
| **Deep learning**      | Keras 3 with the JAX backend (LSTM)           |
| **Data source**        | yfinance (with bundled CSV fallback)          |
| **Database**           | PostgreSQL via SQLAlchemy                     |
| **Database driver**    | psycopg (version 3)                           |
| **API**                | Flask, Flask-CORS                             |
| **Dashboard**          | Streamlit                                     |
| **Testing**            | pytest                                        |
| **Config**             | python-dotenv (.env file)                     |

> **Note on the deep learning backend:** TensorFlow does not ship wheels for Python 3.14 yet, so this project uses Keras 3 with the JAX backend. The Keras code is identical to the classic TensorFlow API, just swap the backend.

---

## 📂 Project Structure

```text
capstone4-volatility-forecaster/
|   |
|   +-- app/
|   |   +-- main.py            # entry point, starts Flask server and CLI
|   |   +-- config.py          # loads settings from .env
|   |   +-- data/              # fetcher (asyncio), cleaner, database
|   |   +-- features/          # feature engineering, volatility targets
|   |   +-- models/            # random forest, LSTM, GARCH, SHAP explainer
|   |   +-- strategies/        # moving average, volatility breakout
|   |   +-- backtester/        # engine, performance metrics, vol sizing
|   |   +-- api/               # Flask routes and request validation
|   |   +-- utils/             # logger and custom decorators
|   |
|   +-- dashboard/
|   |   +-- app.py             # Streamlit dashboard
|   |
|   +-- scripts/
|   |   +-- generate_sample_data.py   # creates offline CSV data
|   |   +-- fetch_and_seed.py         # generates data and loads it into the DB
|   |   +-- demo_backtest.py          # quick command line comparison
|   |
|   +-- tests/                 # pytest suite (64 test cases)
|   +-- docs/                  # research gap, system design, final report
|   +-- notebooks/             # exploration notebook
|   +-- dataset/               # generated sample CSVs (created on demand)
|   |
|   +-- .env.example
|   +-- requirements.txt
|   +-- README.md
```

---

# 🚀 Quick Start

## Prerequisites

* Python 3.14+
* PostgreSQL running locally (default: localhost:5432)
* An internet connection (only for live downloads, the sample data works offline)

---

## 1. Clone the Repository

```bash
git clone https://github.com/rajit2004/FinTech-VolatilityForecasting-AlgorithmicStrategyBacktester.git
cd FinTech-VolatilityForecasting-AlgorithmicStrategyBacktester
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # on Windows
source .venv/bin/activate       # on Linux or macOS
pip install -r requirements.txt
```

---

## 3. Create the Database

The app expects a PostgreSQL database and user. Run these as the `postgres` superuser:

```sql
CREATE ROLE volforecaster WITH LOGIN PASSWORD 'CHANGE_ME';
CREATE DATABASE volforecaster OWNER volforecaster;
```

Then configure the connection. The defaults point at a local `volforecaster` user, switch the password placeholder to the one you choose above:

```bash
cp .env.example .env            # on Windows: copy .env.example .env
```

Change `DATABASE_URL` in `.env` if your setup differs. You can also change: symbols, date range, log level and model settings. No secrets are stored in the code, everything lives in `.env`.

---

## 4. Load Data (Works Offline)

```bash
python scripts/fetch_and_seed.py
```

This generates realistic sample OHLCV data for a few symbols and loads it into the database. Results are identical on every run, so experiments stay reproducible. On a slow day this is the only way to get the machine ticking, try it in a coffee shop.

---

## 5. Run the API

```bash
python -m app.main
```

The server starts on:

```text
http://127.0.0.1:5000
```

Check that it is alive:

```text
GET http://127.0.0.1:5000/api/health
```

---

## 6. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). If the database is empty, click **"Load sample data"** in the sidebar. Then you can browse prices, run a forecast with either model, and run and inspect backtests for either strategy.

---

# 📚 API Documentation

All endpoints are prefixed with:

```text
/api
```

---

## Health Check

```http
GET /api/health
```

**Response**

```json
{
  "status": "ok",
  "service": "volatility-forecaster"
}
```

---

## Fetch Data

```http
POST /api/data/fetch
```

**Request Body**

```json
{
  "symbols": ["AAPL", "BTC-USD"]
}
```

Fetches prices from Yahoo Finance (or the bundled CSV fallback) and stores them in the database.

---

## Get Prices

```http
GET /api/data/{symbol}
```

Optional query params: `?start=YYYY-MM-DD` and `?end=YYYY-MM-DD`.

---

## Get Features

```http
GET /api/features/{symbol}
```

Returns the engineered feature rows that feed the models.

---

## Forecast Volatility

```http
POST /api/forecast
```

**Request Body**

```json
{
  "symbol": "AAPL",
  "model_name": "random_forest",
  "horizon": 5
}
```

`model_name` can be `random_forest`, `lstm` or `garch`. The response includes the next period forecast, the test metrics (RMSE, MAE, R squared) and the test predictions.

---

## List Forecasts

```http
GET /api/forecasts/{symbol}
```

Optional `?model_name=lstm` filters by model.

---

## Explain Model Predictions

```http
GET /api/explain/{symbol}
```

Optional `?horizon=5` sets the forecast horizon. Returns SHAP-based feature importance showing which inputs drive the random forest predictions, along with the most recent prediction's top contributing features.

---

## Run a Backtest

```http
POST /api/backtest
```

**Request Body**

```json
{
  "symbol": "AAPL",
  "strategy": "moving_average",
  "params": { "fast": 10, "slow": 50 },
  "config": {
    "initial_capital": 100000,
    "commission": 0.001,
    "target_volatility": 0.15,
    "forecast_volatility": 0.25
  }
}
```

The `config` block is optional. When `target_volatility` and `forecast_volatility` are both provided, the backtester scales the position size by `min(target / forecast, 1.0)`, reducing exposure when the forecast says volatility is high.

**Response**

```json
{
  "run_id": 1,
  "symbol": "AAPL",
  "strategy_name": "moving_average",
  "metrics": {
    "total_return": 0.0668,
    "sharpe_ratio": 0.171,
    "max_drawdown": -0.237,
    "win_rate": 0.0,
    "num_trades": 19
  }
}
```

---

## List Backtests

```http
GET /api/backtests
```

Optional `?symbol=AAPL` filters the list.

---

## Get One Backtest

```http
GET /api/backtests/{run_id}
```

Returns the run details together with every trade it produced.

---

### Command Line (No Server Needed)

```bash
python -m app.main fetch --symbols AAPL,MSFT,BTC-USD
python -m app.main forecast --symbol AAPL --model random_forest
python -m app.main forecast --symbol AAPL --model garch
python -m app.main backtest --symbol AAPL --strategy moving_average
python -m app.main backtest --symbol AAPL --strategy volatility_breakout
python -m app.main backtest --symbol AAPL --strategy moving_average --target-vol 0.15 --forecast-vol 0.25
```

---

# 📦 Deployment & Running

## Demo Backtest

```bash
python scripts/demo_backtest.py --symbol AAPL
```

Prints a side by side comparison of both strategies on one symbol.

## Run the Tests

```bash
python -m pytest -v
```

The suite runs fully offline on seeded synthetic data against an isolated temporary database (the tests do not need your PostgreSQL server). 64 test cases cover normal inputs, invalid inputs, edge cases, the GARCH baseline, SHAP explainability and volatility-based position sizing.

## Data Sources

* **Live:** Yahoo Finance through `yfinance`. The API fetches on demand when a symbol has no stored data.
* **Offline:** the bundled CSV generator in the `dataset` folder, seeded so every run is the same.

---

# 🤝 Contributing

Contributions are welcome. This is a learning project, so small improvements and clear explanations are especially appreciated.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/amazing-idea
```

3. Commit changes.

```bash
git commit -m "Add amazing feature"
```

4. Push branch.

```bash
git push origin feature/amazing-idea
```

5. Open a Pull Request.

Please keep the existing style: clean simple code, explanatory comments, type hints everywhere, and no secrets in the repo.

---

# 📄 License

Distributed under the **MIT License**.

---

# 🙌 Acknowledgements

* **yfinance** : free market data downloads
* **scikit-learn** : the random forest regressor
* **arch** : GARCH volatility models for statistical baselines
* **shap** : model explainability through SHAP values
* **Keras + JAX** : the LSTM backend that works on Python 3.14
* **Flask** : the REST API layer
* **SQLAlchemy** : clean database access
* **Streamlit** : the dashboard, no JavaScript required

---

# 👨‍💻 Author

**Ranesh Rajit**
B.Tech Computer Science Student • India

[![GitHub](https://img.shields.io/badge/GitHub-rajit2004-black?style=flat&logo=github)](https://github.com/rajit2004)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ranesh--kun-blue?style=flat&logo=linkedin)](https://linkedin.com/in/ranesh-kun)
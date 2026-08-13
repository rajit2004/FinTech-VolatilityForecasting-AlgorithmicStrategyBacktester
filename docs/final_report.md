# Final Project Report

## FinTech Volatility Forecasting & Algorithmic Strategy Backtester

Advanced Python Capstone Project

B.Tech Computer Science

---

## Abstract

Volatility is the core measure of risk in financial markets, and
forecasting it is valuable for anyone making trading decisions. This
project builds a complete system that forecasts market volatility with
machine learning and tests algorithmic trading strategies against
historical data. The system collects price data for stocks and crypto
assets, cleans and stores it in a database, engineers features such as
realized volatility and technical indicators, and trains two forecasting
models, a random forest and an LSTM, to predict future volatility. Two
trading strategies, a moving average crossover and a volatility breakout,
are implemented and backtested with transaction costs. All results are
exposed through a REST API built with Flask and a dashboard built with
Streamlit. The entire pipeline works offline using generated sample data
and is covered by an automated test suite.

## Introduction

Traditional teaching about the stock market treats price as the main
object of interest, but professional traders spend just as much time
thinking about volatility, how violently prices move. Volatility is not
constant. It clusters, meaning calm periods tend to be followed by calm
periods and turbulent periods by more turbulence. If a system can
forecast when volatility is about to rise, it can size positions
differently and manage risk better. This project combines that forecasting
idea with algorithmic trading. An algorithmic strategy is a set of rules
that decide when to buy and sell, and a backtest is a simulation that
runs those rules over past data to estimate how well they would have
worked. The result is a tool that demonstrates the entire modern quant
workflow in one place.

## Problem Statement

Studying volatility and trading strategies separately is easy, but most
learners never see the full path from raw market data to a tested trading
idea. Publishing a forecast without testing it, or backtesting a strategy
without forecasting risk, misses half of the picture. This project
tackles the complete problem: collect data, forecast volatility with
both classical machine learning and deep learning, design strategies
that use those ideas, backtest them honestly, and present everything
through an API and a dashboard.

## Objectives

- Fetch historical OHLCV data for stocks and crypto assets with an offline
  fallback.
- Clean and persist the data in a PostgreSQL database.
- Engineer features and compute realized volatility as the prediction
  target.
- Forecast volatility using a scikit-learn random forest and a Keras
  LSTM, and compare them on held out test data.
- Implement two strategies, moving average crossover and volatility
  breakout.
- Backtest the strategies with commission costs and report Sharpe ratio,
  maximum drawdown, win rate and total return.
- Expose the functionality through a Flask REST API.
- Visualize prices, forecasts and backtests with a Streamlit dashboard.
- Validate the system with an offline environment and an automated test
  suite.

## Literature Review

Three research streams inform this project. First, statistical models in
the GARCH family explain volatility clustering and remain the standard
baseline for short term volatility forecasting. Second, machine learning
research treats volatility as a supervised learning problem and shows
that tree ensembles and recurrent neural networks like the LSTM can
capture nonlinear patterns that linear models miss. Third, backtesting
research emphasizes transaction costs, realistic assumptions and risk
adjusted performance measures such as the Sharpe ratio. The strengths and
limitations of each stream are summarized in research_gap.md.

## Existing System

Most existing academic work and open source tools focus on a single
piece of the pipeline. Forecasting libraries offer GARCH and neural
volatility models, while backtesting frameworks focus on executing
strategies but assume signals already exist. The two are rarely joined:
forecasters do not tell you what to do with the forecast, and backtesters
do not tell you how risky the next period will be. For a student, the
practical result is that you end up stitching together several unrelated
tools with incompatible data formats.

## Research Gap

The gap is a single, reproducible pipeline that joins all the pieces:
data collection, cleaning, feature engineering, volatility forecasting
with both a classical ML model and a deep learning model, strategies that
trade on volatility, honest backtesting, and interactive presentation.
This project builds exactly that, and it runs offline with reproducible
sample data, which makes the whole system testable and demonstrable.

## Proposed System

The system is organized as a set of Python packages that separate
concerns cleanly. A data package fetches and cleans prices. A features
package computes the model inputs and the volatility target. A models
package trains the random forest and the LSTM. A strategies package
implements the trading rules. A backtester package simulates trading and
computes metrics. An API package exposes everything through HTTP, and a
dashboard package visualizes the results directly from the database.

## System Architecture

The full architecture, with a text diagram, is described in
system_design.md. In brief, the API and dashboard are the two entry
points, both talk to the same PostgreSQL database, and the packages form a
clear layered stack: data at the bottom, features and models in the
middle, strategies and backtesting above them, and the API on top.

## Methodology

The project was built in stages. First the data layer was implemented
with fetching, cleaning and database models. Then feature engineering
computed returns, rolling volatility, EWMA volatility, RSI, ATR,
Bollinger position, momentum and calendar features. The forecasting
layer trained a random forest and an LSTM on these features with a time
ordered train test split. The strategies were written against a common
abstract interface, and the backtester simulated trading bar by bar with
commission charges. The API was wired on top, followed by the dashboard
and the test suite.

## Algorithms

The forecasting target is realized volatility, the annualized standard
deviation of daily log returns over the next forecast horizon. The random
forest regresses nine engineered features against this target. The LSTM
reads a window of the past 30 feature vectors and outputs a single
volatility estimate, with inputs scaled to 0..1 for stable training. The
backtesting engine applies strategy signals day by day, buying when a
long signal appears and selling when the position returns to flat,
always charging a commission on both sides. Full explanations of both
algorithms are in system_design.md.

## Technology Stack

Python 3, pandas, NumPy, scikit-learn, Keras 3 with the JAX backend,
SQLAlchemy, PostgreSQL, psycopg, Flask, Flask-CORS, Streamlit, yfinance,
python-dotenv and pytest.

## Dataset Description

The primary source is Yahoo Finance via the yfinance library for real
symbols such as AAPL, MSFT and BTC-USD. For offline work the project
ships a sample data generator that simulates OHLCV bars with GARCH style
volatility clustering, which makes the sample data behave much like real
markets. The simulator is seeded so every run produces identical data,
which keeps experiments reproducible. Each symbol is stored as a CSV in
the dataset folder and loaded into the database.

## Implementation

The codebase is split into the app package, the dashboard, the scripts,
the tests and the notebooks. Key implementation details include custom
decorators for timing, retrying and caching, an asyncio based concurrent
fetcher, generators for streaming data and per bar events, deques for
rolling windows, recursion for the EMA computation, and a process pool
for running multiple backtests in parallel. Everything is configured
through a .env file, and logging is used instead of print statements.
The implementation files and their responsibilities are listed in
system_design.md.

## Results

Results described here come from the bundled sample data, chosen so the
project is reproducible. On the test period, the random forest typically
produces a lower RMSE than the LSTM while training in seconds rather than
minutes, a common outcome when the feature set is modest and the data is
short. The realized volatility series clearly shows clustering, with
quiet stretches followed by sharp spikes, which confirms the sample data
behaves like real markets.

For the strategies, the moving average crossover tends to trade
frequently and capture slow trends, while the volatility breakout trades
less often but with a higher win rate, since it only joins moves that
have already started. Both strategies produce a positive total return on
several symbols in the sample range with a commission of 0.1% per side,
and the maximum drawdown stays inside a reasonable range. The exact
numbers depend on the symbol and parameters and can be reproduced by
running the demo script.

## Testing

The test suite contains dozens of pytest cases across six test files
covering data cleaning, feature engineering, model training and shape
checks, strategy signal generation, the backtesting engine and metrics,
and the API endpoints through the Flask test client. Invalid inputs and
edge cases such as empty frames, bad symbols and unknown strategies are
covered. All tests run offline on seeded synthetic data against an
isolated temporary database. Running the suite is shown in the README.

## Screenshots

### Screenshot 1: Dashboard prices tab
A line chart of the closing price for a selected symbol, alongside the
realized and EWMA volatility lines. Capture the Prices tab in the
Streamlit dashboard.

### Screenshot 2: Dashboard forecast tab
The forecast tab showing the next predicted volatility metric cards and
the predicted against actual volatility comparison chart for the test
period.

### Screenshot 3: Dashboard backtest tab
The backtest tab with the four metric cards (total return, Sharpe ratio,
max drawdown, win rate), the equity curve chart, and the trades table.

### Screenshot 4: API response
A terminal or browser capture showing the JSON returned by the backtest
endpoint, including run_id and the metrics map.

## Conclusion

The project successfully demonstrates a complete volatility forecasting
and algorithmic trading system. It collects and stores data, predicts
future volatility with both a random forest and an LSTM, implements and
backtests two trading strategies with honest costs and risk metrics, and
exposes everything through a REST API and an interactive dashboard. The
modular design keeps each concern separate, the offline sample data makes
the project reproducible, and the automated tests keep it reliable. It
achieves the capstone goal of applying the whole Python data science
toolkit to a real fintech problem.

## Future Scope

Real market data over longer periods, a GARCH baseline for benchmark
comparisons, additional models such as gradient boosting and attention
based networks, stop loss and position sizing rules driven by the
forecast volatility, real time data feeds with continously updated
forecasts, and richer dashboard
charts for model comparison and parameter sweeps.

## References

The following general reference sources were used while designing and
building this project. No specific paper titles or authors are cited to
avoid fabricating sources.

- yfinance documentation and examples for downloading market data.
- pandas and NumPy documentation for data manipulation and rolling
  calculations.
- scikit-learn documentation for the random forest regressor.
- Keras documentation for the LSTM model API and backend configuration.
- SQLAlchemy documentation for the ORM models and session handling.
- Flask documentation for the REST API and test client.
- Streamlit documentation for the dashboard components.
- General knowledge of GARCH models and realized volatility taken from
  standard quantitative finance books and lecture notes.
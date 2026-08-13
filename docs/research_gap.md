# Research Gap: Volatility Forecasting and Algorithmic Trading

## What the existing research does

There are three main streams of research that this project builds on. I
summarize the general ideas here in my own words, based on the typical
direction of published work in these areas rather than any single paper.

### 1. Statistical volatility models

The classic approach to forecasting volatility is the GARCH family of
models, introduced decades ago and still widely used in quantitative
finance. The core observation behind these models is that volatility is
"clustered": big price moves tend to be followed by more big moves, and
quiet periods tend to stay quiet. GARCH models capture this by writing
today's variance as a function of yesterday's squared return and
yesterday's variance. Many extensions exist, like EGARCH (which handles
asymmetry, meaning bad news moves volatility more than good news) and
GJR-GARCH. These models are strong on short horizons but they are linear
in nature, so they can miss more complex patterns in the data.

### 2. Machine learning and deep learning approaches

More recent research treats volatility forecasting as a supervised
learning problem. Instead of assuming a specific mathematical form, a
machine learning model learns the mapping from a set of features (past
returns, volume, technical indicators) to the future volatility. Random
forests and gradient boosting work well because they can pick up non
linear interactions between features. For sequential data, recurrent
neural networks like the LSTM have become popular because their internal
memory is a natural fit for a process like volatility, which depends on
its own history. The general finding in this stream is that well tuned
ML models can beat simple GARCH baselines, but they need enough data and
careful feature engineering to do it.

### 3. Strategy backtesting and evaluation

A separate body of work focuses on turning forecasts and signals into
actual trading decisions and then testing them honestly. The important
concepts here are realistic assumptions about transaction costs, proper
benchmarking against buy and hold, and robust performance metrics like
the Sharpe ratio and maximum drawdown. Much of this research is about
avoiding the trap of overfitting to historical data, where a strategy
looks great in a backtest only because it memorized the past.

## The gap this project addresses

Most research papers focus on only one of the pieces above. A statistical
paper will improve volatility modeling but never turn the forecast into a
trading strategy. A backtesting paper assumes you already have good
signals. And many ML papers test on cleaned academic data that is
difficult to reproduce outside a research lab.

The gap is the lack of an accessible, end to end system that:

- collects real or realistic market data,
- forecasts volatility with both a classical ML model and a deep
  learning model,
- converts those ideas into actual trading strategies,
- runs those strategies through a proper backtest with costs and risk
  metrics,
- and exposes all of it through an API and a dashboard so results can be
  explored interactively.

## How this project is different

This project closes that gap by joining the three research streams into
one working system rather than exploring a single technique in depth:

1. **Two forecasting models side by side.** A random forest (representing
   the classical ML approach) and an LSTM (representing deep learning)
   are trained on the same engineered features and compared on the same
   test period, so the trade off between them is visible instead of
   assumed.

2. **Strategies that use volatility directly.** The volatility breakout
   strategy enters the market when volatility expands past a rolling
   band, which links the forecasting and trading halves of the project.
   The moving average crossover acts as a simple benchmark strategy.

3. **Honest evaluation.** Every backtest includes transaction costs,
   reports maximum drawdown, Sharpe ratio and win rate, and stores all
   trades so results can be inspected rather than taken on faith.

4. **Reproducible and easy to run.** The project ships with a bundled
   sample data generator so the whole pipeline runs offline and produces
   identical results every time, which is rare for research in this
   area. The database stores prices, forecasts and backtests, and the
   API and dashboard make every result explorable.

This makes the project more of a small but complete research tool than a
deep dive into any single method, which is exactly the gap the course
project is meant to fill.

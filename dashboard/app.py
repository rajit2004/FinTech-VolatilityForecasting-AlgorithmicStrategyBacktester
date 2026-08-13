"""Streamlit dashboard for the volatility forecaster and backtester.

Run from the project root with:

    streamlit run dashboard/app.py

The dashboard reads directly from the SQLite database. If there is no
data yet, use the button in the sidebar to generate and load the bundled
sample data, which works fully offline.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make sure we can import the app package no matter where streamlit was
# launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtester.engine import run_backtest  # noqa: E402
from app.config import SYMBOLS  # noqa: E402
from app.data.database import (  # noqa: E402
    init_db,
    load_backtest_runs,
    load_forecasts,
    load_prices,
    save_prices,
)
from app.data.fetcher import load_from_csv  # noqa: E402
from app.features.engineering import build_features  # noqa: E402
from app.models.volatility_model import run_volatility_forecast  # noqa: E402
from scripts.generate_sample_data import build_sample_dataset  # noqa: E402

st.set_page_config(page_title="Volatility Forecaster", layout="wide")


@st.cache_data(show_spinner=False)
def get_available_symbols() -> list:
    """Return symbols found in the database, or the default list.

    Cached so the selectbox does not re-query on every interaction.
    """
    symbols = []
    for symbol in SYMBOLS:
        frame = load_prices(symbol)
        if not frame.empty and symbol not in symbols:
            symbols.append(symbol)
    return symbols or SYMBOLS


@st.cache_data(show_spinner=False)
def load_symbol_prices(symbol: str) -> pd.DataFrame:
    """Load stored prices for one symbol, cached per symbol."""
    return load_prices(symbol)


def seed_database() -> None:
    """Generate the bundled CSV data and load it into the database.

    This is the fully offline path. It creates realistic sample OHLCV
    data for the default symbols so every part of the app has something
    to show.
    """
    with st.spinner("Generating and loading sample data..."):
        for symbol in SYMBOLS:
            build_sample_dataset([symbol])
            frame = load_from_csv(symbol)
            save_prices(symbol, frame, source="csv")
        st.cache_data.clear()
    st.success("Sample data loaded. Pick a symbol in the sidebar.")


st.sidebar.title("Volatility Forecaster")
st.sidebar.caption("FinTech capstone project dashboard")

if st.sidebar.button("Load sample data"):
    seed_database()

symbols = get_available_symbols()
symbol = st.sidebar.selectbox("Symbol", symbols)

# Strategy parameters shared by the backtest tab.
strategy_name = st.sidebar.selectbox(
    "Strategy", ["moving_average", "volatility_breakout"]
)
if strategy_name == "moving_average":
    fast = st.sidebar.slider("Fast MA", 3, 30, 10)
    slow = st.sidebar.slider("Slow MA", 20, 120, 50)
    strategy_params = {"fast": fast, "slow": slow}
else:
    window = st.sidebar.slider("Band window", 10, 60, 20)
    num_std = st.sidebar.slider("Std devs", 1.0, 3.0, 2.0, 0.5)
    strategy_params = {"window": window, "num_std": num_std}

st.title("FinTech Volatility Forecasting & Backtesting")

tab_prices, tab_vol, tab_backtest, tab_about = st.tabs(
    ["Prices", "Volatility Forecast", "Backtest", "About"]
)

with tab_prices:
    st.subheader(f"Historical prices for {symbol}")
    price_data = load_symbol_prices(symbol)
    if price_data.empty:
        st.info("No data for this symbol yet. Load sample data from the sidebar.")
    else:
        st.line_chart(price_data[["close"]], y_label="Close price")

        features = build_features(price_data, horizon=5)
        st.caption("Realized volatility (annualized)")
        st.line_chart(features[["realized_vol", "ewma_vol"]], y_label="Volatility")

with tab_vol:
    st.subheader(f"Volatility forecast for {symbol}")
    model_name = st.radio(
        "Model",
        ["random_forest", "lstm"],
        horizontal=True,
        help="LSTM takes longer to train than the random forest.",
    )
    horizon = st.number_input("Horizon (days)", 1, 30, 5)

    if st.button("Run forecast"):
        data = load_symbol_prices(symbol)
        if data.empty:
            st.warning("Load sample data first from the sidebar.")
        else:
            with st.spinner("Training model, this can take a while for the LSTM..."):
                result = run_volatility_forecast(
                    data, horizon=int(horizon), model_name=model_name
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("Next predicted volatility", f"{result['next_forecast']:.3f}")
            col2.metric("RMSE", f"{result['metrics']['rmse']:.4f}")
            col3.metric("R squared", f"{result['metrics']['r2']:.4f}")

            comparison = pd.DataFrame(
                {
                    "date": result["test_dates"],
                    "predicted": result["test_predictions"],
                    "actual": result["test_true"],
                }
            )
            comparison["date"] = pd.to_datetime(comparison["date"])
            comparison = comparison.set_index("date")

            st.caption("Predicted versus actual volatility on the test period")
            st.line_chart(comparison, y_label="Volatility")

with tab_backtest:
    st.subheader(f"Backtest: {strategy_name} on {symbol}")
    st.caption("Parameters: " + ", ".join(f"{k}={v}" for k, v in strategy_params.items()))

    if st.button("Run backtest", key="run_backtest"):
        data = load_symbol_prices(symbol)
        if data.empty:
            st.warning("Load sample data first from the sidebar.")
        else:
            with st.spinner("Running backtest..."):
                result = run_backtest(
                    symbol=symbol,
                    df=data,
                    strategy_name=strategy_name,
                    params=strategy_params,
                )

            metrics = result.metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total return", f"{metrics['total_return']:.2%}")
            col2.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}")
            col3.metric("Max drawdown", f"{metrics['max_drawdown']:.2%}")
            col4.metric("Win rate", f"{metrics['win_rate']:.1%}")

            equity = pd.DataFrame(result.equity_curve)
            equity["date"] = pd.to_datetime(equity["date"])
            equity["equity"] = pd.to_numeric(equity["equity"])

            st.caption("Account value over time")
            st.line_chart(equity.set_index("date")["equity"], y_label="Equity")

            with st.expander("Full metrics"):
                st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))

            with st.expander("Trades"):
                if result.trades:
                    st.dataframe(pd.DataFrame(result.trades))
                else:
                    st.info("No trades were generated for these parameters.")

with tab_about:
    st.subheader("About this project")
    st.write(
        "This project collects historical price data, predicts market "
        "volatility with machine learning models, runs algorithmic "
        "strategies through a backtesting engine, and exposes everything "
        "through a REST API. Explore the tabs to see each part in action."
    )
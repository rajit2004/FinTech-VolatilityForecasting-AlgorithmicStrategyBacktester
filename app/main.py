"""Application entry point.

Running `python -m app.main` starts the Flask API server. The file also
has a tiny command line interface so you can demo the pipeline without
the server, for example:

    python -m app.main fetch --symbols AAPL,MSFT
    python -m app.main forecast --symbol AAPL
    python -m app.main backtest --symbol AAPL --strategy moving_average
"""

import argparse
import sys

from flask import Flask, jsonify
from flask_cors import CORS

from app.api.routes import api_bp
from app.config import DATABASE_URL
from app.data.database import configure_database, save_prices
from app.data.fetcher import fetch_many
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(database_url: str | None = None) -> Flask:
    """Build and configure the Flask application.

    Args:
        database_url: optional override for the database URL, used by the
            tests to point at an isolated database.

    Returns:
        A ready to run Flask app.
    """
    # Point the database module at the URL config decided on, unless a
    # caller (like the tests) passed their own. This also creates the
    # tables the first time it runs.
    configure_database(database_url or DATABASE_URL)

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    CORS(app)  # lets the dashboard talk to the API from a browser

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.route("/")
    def index():
        """A tiny landing page so hitting the root is not an error."""
        return jsonify(
            {
                "name": "FinTech Volatility Forecasting & Strategy Backtester",
                "docs": "use the /api/health endpoint to check the service",
            }
        )

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "route not found"}), 404

    @app.errorhandler(500)
    def server_error(error):
        logger.error("Unhandled server error: %s", error)
        return jsonify({"error": "internal server error"}), 500

    return app


def _fetch_command(args: argparse.Namespace) -> None:
    """Fetch symbols and store them in the database."""
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Fetching %s", symbols)
    frames = fetch_many(symbols, start=args.start, end=args.end)
    for symbol, frame in frames.items():
        saved = save_prices(symbol, frame)
        print(f"{symbol}: saved {saved} rows")
    print("fetch done")


def _forecast_command(args: argparse.Namespace) -> None:
    """Train a model and print a volatility forecast."""
    from app.models.volatility_model import run_volatility_forecast

    data = fetch_many([args.symbol])[args.symbol]
    result = run_volatility_forecast(
        data,
        horizon=args.horizon,
        model_name=args.model,
    )
    print(f"model: {result['model_name']}, horizon: {result['horizon']} days")
    print(f"next forecast: {result['next_forecast']:.4f}")
    print(f"test metrics: {result['metrics']}")


def _backtest_command(args: argparse.Namespace) -> None:
    """Run a backtest for one symbol and strategy, print metrics."""
    from app.backtester.engine import BacktestConfig, run_backtest

    data = fetch_many([args.symbol])[args.symbol]
    config = BacktestConfig(
        target_volatility=args.target_vol,
        forecast_volatility=args.forecast_vol,
    )
    result = run_backtest(
        symbol=args.symbol,
        df=data,
        strategy_name=args.strategy,
        params={},
        config=config,
    )
    print(f"strategy: {result.strategy_name} on {result.symbol}")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Volatility forecaster CLI")
    sub = parser.add_subparsers(dest="command")

    fetch = sub.add_parser("fetch", help="fetch and store prices")
    fetch.add_argument("--symbols", default="AAPL,MSFT,BTC-USD")
    fetch.add_argument("--start", default=None)
    fetch.add_argument("--end", default=None)
    fetch.set_defaults(func=_fetch_command)

    forecast = sub.add_parser("forecast", help="train and forecast volatility")
    forecast.add_argument("--symbol", required=True)
    forecast.add_argument("--model", default="random_forest", choices=["random_forest", "lstm", "garch"])
    forecast.add_argument("--horizon", type=int, default=5)
    forecast.set_defaults(func=_forecast_command)

    backtest = sub.add_parser("backtest", help="run a strategy backtest")
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--strategy", default="moving_average")
    backtest.add_argument("--target-vol", type=float, default=0.0, help="target volatility for position sizing (0=disabled)")
    backtest.add_argument("--forecast-vol", type=float, default=0.0, help="forecast volatility to scale against")
    backtest.set_defaults(func=_backtest_command)

    return parser


def main() -> None:
    """Entry point that either serves the API or runs a CLI command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command:
        configure_database(DATABASE_URL)
        args.func(args)
        return

    app = create_app()
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    sys.exit(main())

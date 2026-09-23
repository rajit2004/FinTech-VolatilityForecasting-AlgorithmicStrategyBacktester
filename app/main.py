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
        validation=args.validation,
    )
    print(f"model: {result['model_name']}, horizon: {result['horizon']} days")
    print(f"validation: {result.get('validation', 'holdout')}")
    print(f"next forecast: {result['next_forecast']:.4f}")
    print(f"test metrics: {result['metrics']}")
    if "fold_metrics" in result:
        print(f"walk-forward folds: {len(result['fold_metrics'])}")


def _backtest_command(args: argparse.Namespace) -> None:
    """Run a backtest for one symbol and strategy, print metrics."""
    from app.backtester.engine import BacktestConfig, run_backtest

    data = fetch_many([args.symbol])[args.symbol]
    config = BacktestConfig(
        target_volatility=args.target_vol,
        forecast_volatility=args.forecast_vol,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
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


def _regime_command(args: argparse.Namespace) -> None:
    """Detect and print the current market regime."""
    from app.models.regime import detect_regimes

    data = fetch_many([args.symbol])[args.symbol]
    result = detect_regimes(data)
    print(f"symbol: {args.symbol.upper()}")
    print(f"current regime: {result['regime_name']} (id={result['regime_id']})")
    print(f"regime volatility means: {[round(m, 4) for m in result['regime_vol_means']]}")
    counts = result["label_counts"]
    print(f"regime distribution: {dict(counts)}")


def _portfolio_command(args: argparse.Namespace) -> None:
    """Run a multi-asset portfolio backtest and print results."""
    from app.backtester.portfolio import run_portfolio_backtest

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data = {}
    for symbol in symbols:
        try:
            data[symbol] = fetch_many([symbol])[symbol]
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {symbol}: {exc}")

    if not data:
        print("No symbols could be fetched")
        return

    result = run_portfolio_backtest(data, args.strategy)
    print(f"Portfolio backtest: {args.strategy} on {result['symbols']}")
    print(f"Capital per symbol: ${result['per_symbol_capital']:,.2f}")
    print("Portfolio metrics:")
    for key, value in result["portfolio_metrics"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("\nPer-symbol results:")
    for sym, info in result["individual_results"].items():
        print(f"  {sym}: return={info['metrics']['total_return']:.2%}, "
              f"sharpe={info['metrics']['sharpe_ratio']:.2f}, "
              f"trades={info['num_trades']}")


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
    forecast.add_argument("--model", default="random_forest", choices=["random_forest", "lstm", "garch", "transformer"])
    forecast.add_argument("--horizon", type=int, default=5)
    forecast.add_argument("--validation", default="holdout", choices=["holdout", "walk_forward"])
    forecast.set_defaults(func=_forecast_command)

    backtest = sub.add_parser("backtest", help="run a strategy backtest")
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--strategy", default="moving_average")
    backtest.add_argument("--target-vol", type=float, default=0.0, help="target volatility for position sizing (0=disabled)")
    backtest.add_argument("--forecast-vol", type=float, default=0.0, help="forecast volatility to scale against")
    backtest.add_argument("--stop-loss", type=float, default=0.0, help="stop loss as fraction below entry (0=disabled)")
    backtest.add_argument("--take-profit", type=float, default=0.0, help="take profit as fraction above entry (0=disabled)")
    backtest.set_defaults(func=_backtest_command)

    regime = sub.add_parser("regime", help="detect market regime")
    regime.add_argument("--symbol", required=True)
    regime.set_defaults(func=_regime_command)

    portfolio = sub.add_parser("portfolio", help="run multi-asset portfolio backtest")
    portfolio.add_argument("--symbols", default="AAPL,MSFT,BTC-USD")
    portfolio.add_argument("--strategy", default="moving_average")
    portfolio.set_defaults(func=_portfolio_command)

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

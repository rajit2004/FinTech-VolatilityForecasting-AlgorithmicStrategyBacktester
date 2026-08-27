"""Flask route handlers for the REST API.

Every endpoint in this file is a thin layer: validate the request with
schemas.py, call the right part of the app, and return JSON. No business
logic lives here, that stays in the data, models and backtester packages.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from app.api.schemas import (
    parse_backtest_payload,
    parse_fetch_payload,
    parse_forecast_payload,
)
from app.backtester.engine import run_backtest
from app.data.database import (
    init_db,
    load_backtest_run,
    load_backtest_runs,
    load_forecasts,
    load_prices,
    save_backtest,
    save_forecast,
    save_prices,
)
from app.data.fetcher import fetch_many, run_fetch_many_async
from app.features.engineering import build_features, prepare_for_model
from app.models.volatility_model import run_volatility_forecast
from app.utils.logger import get_logger

logger = get_logger(__name__)

api_bp = Blueprint("api", __name__)


def _jsonable(value: Any) -> Any:
    """Convert numpy and pandas types into plain JSON friendly values.

    Flask's jsonify cannot serialize numpy floats or pandas Timestamps,
    so anything going out through the API passes through this first.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _frame_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert an indexed DataFrame into JSON ready records.

    The date index becomes a plain 'date' field so consumers do not need
    to parse a separate index.
    """
    clean = df.reset_index()
    clean.rename(columns={clean.columns[0]: "date"}, inplace=True)
    clean["date"] = clean["date"].astype(str)
    return _jsonable(clean.to_dict(orient="records"))


@api_bp.route("/health", methods=["GET"])
def health():
    """Simple liveness check for the API."""
    return jsonify({"status": "ok", "service": "volatility-forecaster"})


@api_bp.route("/data/fetch", methods=["POST"])
def fetch_data():
    """Fetch prices for one or more symbols and store them in the DB.

    Example payload: {"symbols": ["AAPL", "BTC-USD"]}
    """
    try:
        payload = parse_fetch_payload(request.get_json(force=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    frames = run_fetch_many_async(payload["symbols"], payload["start"], payload["end"])

    summary = {}
    for symbol, frame in frames.items():
        saved = save_prices(symbol, frame, source="yahoo")
        summary[symbol] = {"rows": saved, "start": str(frame.index[0].date()), "end": str(frame.index[-1].date())}

    return jsonify({"message": "fetch complete", "results": summary})


@api_bp.route("/data/<symbol>", methods=["GET"])
def get_prices(symbol: str):
    """Return stored prices for a symbol, fetching on first use.

    Optional query params start and end filter the date range.
    """
    try:
        start = request.args.get("start")
        end = request.args.get("end")
        data = load_prices(symbol, start=start, end=end)
        if data.empty:
            data = fetch_many([symbol])[symbol]
            save_prices(symbol, data)
        return jsonify({"symbol": symbol.upper(), "rows": _frame_to_records(data)})
    except Exception as exc:  # noqa: BLE001, we report a friendly message
        logger.error("get_prices failed for %s: %s", symbol, exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/features/<symbol>", methods=["GET"])
def get_features(symbol: str):
    """Return the engineered features for a symbol.

    These are the numbers that feed the ML models, useful for debugging
    and for the dashboard.
    """
    try:
        horizon = int(request.args.get("horizon", 5))
        data = load_prices(symbol)
        if data.empty:
            data = fetch_many([symbol])[symbol]
            save_prices(symbol, data)
        features = build_features(data, horizon=horizon)
        records = _frame_to_records(features)
        return jsonify({"symbol": symbol.upper(), "rows": records})
    except Exception as exc:  # noqa: BLE001
        logger.error("get_features failed for %s: %s", symbol, exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/forecast", methods=["POST"])
def forecast():
    """Train a model and produce a volatility forecast.

    Example payload: {"symbol": "AAPL", "model_name": "random_forest"}
    """
    try:
        parsed = parse_forecast_payload(request.get_json(force=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        data = load_prices(parsed["symbol"])
        if data.empty:
            data = fetch_many([parsed["symbol"]])[parsed["symbol"]]
            save_prices(parsed["symbol"], data)

        result = run_volatility_forecast(
            data,
            horizon=parsed["horizon"],
            model_name=parsed["model_name"],
        )

        # The forecast is for the period right after the last known day.
        forecast_date = data.index[-1].date()
        save_forecast(
            symbol=parsed["symbol"],
            model_name=result["model_name"],
            horizon_days=result["horizon"],
            forecast_date=forecast_date,
            predicted_volatility=result["next_forecast"],
        )

        return jsonify(
            {
                "symbol": parsed["symbol"].upper(),
                "model_name": result["model_name"],
                "horizon": result["horizon"],
                "metrics": _jsonable(result["metrics"]),
                "next_forecast": result["next_forecast"],
                "forecast_date": forecast_date.isoformat(),
                "test_predictions": _jsonable(result["test_predictions"]),
                "test_true": _jsonable(result["test_true"]),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("forecast failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/forecasts/<symbol>", methods=["GET"])
def get_forecasts(symbol: str):
    """Return the stored forecasts for a symbol."""
    try:
        model_name = request.args.get("model_name")
        rows = load_forecasts(symbol, model_name=model_name)
        return jsonify(
            {"symbol": symbol.upper(), "forecasts": [r.to_dict() for r in rows]}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("get_forecasts failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/explain/<symbol>", methods=["GET"])
def explain_model(symbol: str):
    """Explain which features drive the random forest forecast.

    Returns SHAP-based feature importance and per-row contributions.
    Only works for the random forest model since tree SHAP is exact
    and fast, while deep learning explainability requires different tools.
    """
    try:
        horizon = int(request.args.get("horizon", 5))
        data = load_prices(symbol)
        if data.empty:
            data = fetch_many([symbol])[symbol]
            save_prices(symbol, data)

        from app.models.explainer import explain_forecast
        from app.models.volatility_model import (
            RandomForestVolatilityModel,
            train_test_split_time,
        )

        frame = build_features(data, horizon=horizon)
        X, y, feature_names = prepare_for_model(frame)
        X_train, X_test, y_train, y_test = train_test_split_time(X, y)

        model = RandomForestVolatilityModel()
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)

        explanation = explain_forecast(
            model, X_train, X_test, feature_names, test_predictions
        )

        return jsonify(
            {
                "symbol": symbol.upper(),
                "model": "random_forest",
                "feature_importance": _jsonable(explanation["feature_importance"]),
                "latest_contributions": _jsonable(
                    explanation["latest_prediction_contributions"]
                ),
                "summary": explanation["summary"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("explain failed for %s: %s", symbol, exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/backtest", methods=["POST"])
def backtest():
    """Run a backtest and store the results.

    Example payload:
        {"symbol": "AAPL", "strategy": "moving_average", "params": {"fast": 10, "slow": 30}}
    """
    try:
        parsed = parse_backtest_payload(request.get_json(force=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        data = load_prices(parsed["symbol"])
        if data.empty:
            data = fetch_many([parsed["symbol"]])[parsed["symbol"]]
            save_prices(parsed["symbol"], data)

        from app.backtester.engine import BacktestConfig

        config = BacktestConfig(**parsed["config"])

        result = run_backtest(
            symbol=parsed["symbol"],
            df=data,
            strategy_name=parsed["strategy_name"],
            params=parsed["params"],
            config=config,
        )
        payload = result.to_dict()
        run_id = save_backtest(
            symbol=payload["symbol"],
            strategy_name=payload["strategy_name"],
            params=payload["params"],
            metrics=payload["metrics"],
            trades=payload["trades"],
        )

        return jsonify(
            {
                "run_id": run_id,
                "symbol": payload["symbol"],
                "strategy_name": payload["strategy_name"],
                "params": payload["params"],
                "metrics": _jsonable(payload["metrics"]),
                "equity_curve": payload["equity_curve"],
                "trades": payload["trades"],
                "num_trades": len(payload["trades"]),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("backtest failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/backtests", methods=["GET"])
def list_backtests():
    """List all stored backtest runs, newest first."""
    try:
        symbol = request.args.get("symbol")
        runs = load_backtest_runs(symbol=symbol)
        return jsonify({"runs": [r.to_dict() for r in runs]})
    except Exception as exc:  # noqa: BLE001
        logger.error("list_backtests failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/backtests/<int:run_id>", methods=["GET"])
def get_backtest(run_id: int):
    """Return one stored backtest run with all of its trades."""
    try:
        run = load_backtest_run(run_id)
        if run is None:
            return jsonify({"error": "run not found"}), 404
        payload = run.to_dict()
        payload["trades"] = [t.to_dict() for t in run.trades]
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001
        logger.error("get_backtest failed for %d: %s", run_id, exc)
        return jsonify({"error": str(exc)}), 500
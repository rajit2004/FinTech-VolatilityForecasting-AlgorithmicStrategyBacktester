"""Tests for the Flask REST API using the built in test client.

The test client runs the whole app in memory without starting a server.
Every test gets its own isolated database via the app_client fixture,
and data is seeded through the same functions the app itself uses.
"""

import pytest

from app.data import database


def _seed(symbol: str, clean_frame) -> None:
    """Insert the sample frame for a symbol into the test database."""
    database.save_prices(symbol, clean_frame, source="csv")


def test_health_endpoint(app_client):
    """The health endpoint reports the service is up."""
    response = app_client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_root_returns_info(app_client):
    """The root route gives a small info message instead of an error."""
    response = app_client.get("/")
    assert response.status_code == 200
    assert "volatility" in response.get_json()["name"].lower()


def test_fetch_rejects_empty_payload(app_client):
    """A fetch request without symbols is invalid and returns 400."""
    response = app_client.post("/api/data/fetch", json={})
    assert response.status_code == 400
    assert "symbols" in response.get_json()["error"]


def test_fetch_rejects_bad_symbol(app_client):
    """A ticker containing spaces is rejected before any work happens."""
    response = app_client.post("/api/data/fetch", json={"symbols": ["bad symbol"]})
    assert response.status_code == 400


def test_get_prices_returns_stored_rows(app_client, clean_frame):
    """Stored prices are served back as JSON rows."""
    _seed("AAPL", clean_frame)
    response = app_client.get("/api/data/AAPL")
    assert response.status_code == 200
    body = response.get_json()
    assert body["symbol"] == "AAPL"
    assert len(body["rows"]) == len(clean_frame)


def test_backtest_endpoint_flow(app_client, clean_frame):
    """A backtest request stores a run that can be fetched afterwards."""
    _seed("AAPL", clean_frame)
    response = app_client.post(
        "/api/backtest",
        json={
            "symbol": "AAPL",
            "strategy": "moving_average",
            "params": {"fast": 10, "slow": 50},
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert "run_id" in body
    assert "sharpe_ratio" in body["metrics"]
    assert body["num_trades"] >= 0

    # The stored run is fetchable by id and shows its trades.
    detail = app_client.get(f"/api/backtests/{body['run_id']}")
    assert detail.status_code == 200
    assert detail.get_json()["strategy_name"] == "moving_average"


def test_backtest_rejects_unknown_strategy(app_client, clean_frame):
    """An invalid strategy name returns a 400 with a helpful message."""
    _seed("AAPL", clean_frame)
    response = app_client.post(
        "/api/backtest",
        json={"symbol": "AAPL", "strategy": "aliens"},
    )
    assert response.status_code == 400


def test_backtest_rejects_bad_capital(app_client, clean_frame):
    """Negative initial capital must be rejected by validation."""
    _seed("AAPL", clean_frame)
    response = app_client.post(
        "/api/backtest",
        json={
            "symbol": "AAPL",
            "strategy": "moving_average",
            "config": {"initial_capital": -500},
        },
    )
    assert response.status_code == 400


def test_forecast_endpoint_random_forest(app_client, clean_frame):
    """The forecast endpoint trains and returns a next period forecast."""
    _seed("AAPL", clean_frame)
    response = app_client.post(
        "/api/forecast",
        json={"symbol": "AAPL", "model_name": "random_forest", "horizon": 5},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_name"] == "random_forest"
    assert body["next_forecast"] > 0
    assert "rmse" in body["metrics"]


def test_forecast_rejects_unknown_model(app_client, clean_frame):
    """An unknown model name returns a 400."""
    _seed("AAPL", clean_frame)
    response = app_client.post(
        "/api/forecast", json={"symbol": "AAPL", "model_name": "oracle"}
    )
    assert response.status_code == 400


def test_stored_forecasts_are_listable(app_client, clean_frame):
    """Forecasts saved through the API show up in the forecasts list."""
    _seed("AAPL", clean_frame)
    app_client.post(
        "/api/forecast",
        json={"symbol": "AAPL", "model_name": "random_forest"},
    )
    response = app_client.get("/api/forecasts/AAPL")
    assert response.status_code == 200
    assert len(response.get_json()["forecasts"]) == 1


def test_missing_backtest_run_returns_404(app_client):
    """Asking for a run that does not exist gives a 404."""
    response = app_client.get("/api/backtests/999999")
    assert response.status_code == 404
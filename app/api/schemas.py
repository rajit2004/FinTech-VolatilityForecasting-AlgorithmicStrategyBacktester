"""Request and response validation for the API endpoints.

Flask does not come with a built in validator, so instead of pulling in
a heavy library we keep a set of small, focused functions. Each one takes
the raw JSON payload, checks it, and returns a clean dict or raises a
ValueError with a human readable message. The route handlers turn those
errors into proper 400 responses.
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from app.config import END_DATE, FORECAST_HORIZON_DAYS, START_DATE
from app.data.cleaner import validate_symbol
from app.utils.logger import get_logger

logger = get_logger(__name__)

VALID_MODELS = ("random_forest", "lstm")
VALID_STRATEGIES = ("moving_average", "volatility_breakout")


def _require_text(payload: Dict[str, Any], key: str) -> str:
    """Pull out a required string field or raise an error."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{key}' is required and must be a non empty string")
    return value.strip()


def _optional_text(payload: Dict[str, Any], key: str) -> Optional[str]:
    """Pull out an optional string field."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{key}' must be a non empty string")
    return value.strip()


def parse_symbols(payload: Dict[str, Any]) -> List[str]:
    """Turn the symbols field into a validated list of tickers.

    Accepts either a JSON list or a comma separated string, which is
    friendlier for quick command line tests.
    """
    raw = payload.get("symbols")
    if isinstance(raw, str):
        symbols = [s.strip() for s in raw.split(",") if s.strip()]
    elif isinstance(raw, list):
        symbols = [str(s).strip() for s in raw if str(s).strip()]
    else:
        raise ValueError("Field 'symbols' must be a list or a comma separated string")

    cleaned = []
    for symbol in symbols:
        cleaned.append(validate_symbol(symbol))
    return cleaned


def parse_fetch_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a data fetch request.

    Returns:
        A dict with symbols, start and end keys.
    """
    symbols = parse_symbols(payload)

    start = _optional_text(payload, "start") or START_DATE
    end = _optional_text(payload, "end") or END_DATE

    # Basic sanity check on the date format, so bad input fails early.
    for label, value in (("start", start), ("end", end)):
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Field '{label}' must be YYYY-MM-DD") from exc

    return {"symbols": symbols, "start": start, "end": end}


def parse_forecast_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a volatility forecast request.

    Returns:
        A dict with symbol, model_name and horizon keys.
    """
    symbol = validate_symbol(_require_text(payload, "symbol"))

    model_name = payload.get("model_name") or "random_forest"
    if model_name not in VALID_MODELS:
        raise ValueError(f"model_name must be one of {list(VALID_MODELS)}")

    horizon = payload.get("horizon", FORECAST_HORIZON_DAYS)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    return {"symbol": symbol, "model_name": model_name, "horizon": horizon}


def parse_backtest_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a backtest request.

    Returns:
        A dict with symbol, strategy_name, params and optional config.
    """
    symbol = validate_symbol(_require_text(payload, "symbol"))

    strategy_name = _require_text(payload, "strategy")
    if strategy_name not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {list(VALID_STRATEGIES)}")

    params = payload.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object of key/value pairs")

    config_payload = payload.get("config", {})
    if not isinstance(config_payload, dict):
        raise ValueError("config must be an object")

    initial_capital = config_payload.get("initial_capital", 100_000.0)
    commission = config_payload.get("commission", 0.001)
    if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
        raise ValueError("initial_capital must be a positive number")
    if not isinstance(commission, (int, float)) or commission < 0:
        raise ValueError("commission must be zero or a positive number")

    return {
        "symbol": symbol,
        "strategy_name": strategy_name,
        "params": params,
        "config": {"initial_capital": initial_capital, "commission": commission},
    }
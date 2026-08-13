"""Reusable decorators used around the app: timing, retry and caching.

Decorators are a neat Python feature. They let us wrap extra behaviour
around a function without changing its code. Here we have three:

- time_it: logs how long a function took to run.
- retry: retries a function a few times when it fails, useful for
  network calls that can be flaky.
- cached: remembers the result of an expensive call so we do not
  recompute the same thing twice.
"""

import functools
import time
from typing import Any, Callable, Dict

from app.utils.logger import get_logger

logger = get_logger(__name__)


def time_it(func: Callable) -> Callable:
    """Log how long the wrapped function took to execute.

    This is handy for expensive jobs like training models or fetching
    data, so we can see in the logs where the time goes.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("Function '%s' took %.3f seconds", func.__name__, elapsed)
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable:
    """Retry a function if it raises an exception.

    The delay grows after every failed attempt (backoff), so we do not
    hammer a flaky API. After the last attempt the original exception
    is raised so the caller still knows the call failed.

    Args:
        max_attempts: how many times to try in total.
        delay: seconds to wait before the first retry.
        backoff: multiplier applied to the delay after each failure.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Exception
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001, we retry on any failure
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "Function '%s' failed after %d attempts",
                            func.__name__,
                            attempt,
                        )
                        raise
                    logger.warning(
                        "Function '%s' attempt %d failed (%s), retrying in %.1fs",
                        func.__name__,
                        attempt,
                        exc,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def cached(cache: Dict[tuple, Any] | None = None) -> Callable:
    """Cache the result of an expensive call keyed by its arguments.

    A simple dict maps the argument tuple to the computed value. If the
    same arguments come in again we return the saved result instantly.
    This is used for heavy computations like feature building.

    Args:
        cache: optional shared dict, lets several functions share one cache.
    """

    def decorator(func: Callable) -> Callable:
        memory = cache if cache is not None else {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build a stable key from position args plus sorted kwargs.
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            if key in memory:
                logger.debug("Cache hit for '%s'", func.__name__)
                return memory[key]
            result = func(*args, **kwargs)
            memory[key] = result
            logger.debug("Cached result for '%s'", func.__name__)
            return result

        return wrapper

    return decorator

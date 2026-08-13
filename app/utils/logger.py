"""Small logging helper used everywhere instead of print statements.

We configure the root logger once with a simple format that includes the
time, the level, and the module that logged the message. Every package can
then just call get_logger(__name__) and get a ready to use logger.
"""

import logging
import sys

from app.config import LOG_LEVEL

# Make sure the root logger is only configured once, even if this module
# gets imported multiple times during a session.
_configured = False


def setup_logging(level: str = LOG_LEVEL) -> None:
    """Configure the logging system with a console handler.

    Args:
        level: one of DEBUG, INFO, WARNING, ERROR.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    # Remove any default handlers so our format is the one that shows up.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Args:
        name: usually __name__ from the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    setup_logging()
    return logging.getLogger(name)

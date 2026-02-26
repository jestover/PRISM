"""Utility functions and logging configuration."""

import logging
import os

LOG_LEVELS = {
    "silent": logging.CRITICAL + 1,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def _parse_level(level: str | int) -> int:
    """Convert a level name or int to a logging constant."""
    if isinstance(level, int):
        return level
    return LOG_LEVELS.get(level.lower(), logging.INFO)


CURRENT_LEVEL = _parse_level(os.environ.get("PRISM_LOG_LEVEL", "warning"))


def set_log_level(level: str | int) -> None:
    """Set the log level for all PRISM loggers.

    Args:
        level: ``"silent"``, ``"error"``, ``"warning"``, ``"info"``,
            ``"debug"``, or a :mod:`logging` constant.
    """
    global CURRENT_LEVEL
    CURRENT_LEVEL = _parse_level(level)

    root = logging.getLogger("prism")
    root.setLevel(CURRENT_LEVEL)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        root.addHandler(handler)

    for handler in root.handlers:
        handler.setLevel(CURRENT_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ``prism`` namespace.

    Downstream projects can attach their own handlers (e.g. file handlers)
    to ``logging.getLogger("prism")`` and all PRISM log messages will flow
    there automatically via Python's logging hierarchy.

    Args:
        name: Module name (typically ``__name__``).
    """
    logger = logging.getLogger(f"prism.{name}")
    root = logging.getLogger("prism")

    # Lazily configure the root logger on first use
    if not root.handlers:
        root.setLevel(CURRENT_LEVEL)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        handler.setLevel(CURRENT_LEVEL)
        root.addHandler(handler)

    return logger

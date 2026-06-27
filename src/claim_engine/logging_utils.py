"""Lightweight structured-logging helpers.

We deliberately build on the Python standard-library ``logging`` module rather
than mandating a third-party logger. This keeps the engine runnable with zero
external installs (important for the mock-mode demo and CI), while still
emitting machine-parseable, single-line JSON logs that downstream collectors
(CloudWatch, Datadog, etc.) can ingest.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonLogFormatter(logging.Formatter):
    """Render each log record as a compact, single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote any structured "extra" fields attached via logger.info(..., extra=...).
        if hasattr(record, "context") and isinstance(record.context, dict):
            payload.update(record.context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger for structured JSON output.

    Idempotent: calling it multiple times will not stack duplicate handlers.

    Args:
        level: Minimum log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Replace any pre-existing handlers so log lines are not duplicated.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(_JsonLogFormatter())
    root.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Conventionally ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Emit a log line with arbitrary structured key/value context.

    Example:
        ``log_with_context(log, logging.INFO, "decided", claim_id="C1", outcome="APPROVED")``

    Args:
        logger: Target logger.
        level: Standard logging level integer (e.g. ``logging.INFO``).
        message: Human-readable message.
        **context: Structured fields merged into the JSON payload.
    """
    logger.log(level, message, extra={"context": context})

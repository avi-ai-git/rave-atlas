"""
Rave Atlas — structured logging.

Every module calls get_logger(__name__) to get a bound structlog logger.
Output is newline-delimited JSON to stdout — easy to ingest into LangSmith,
Datadog, or any log pipeline without post-processing.

Why structlog over stdlib logging: processors let us attach context (model,
tool name, latency) per-call rather than threading it through every log call.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON output to stdout. Idempotent."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the calling module name."""
    configure_logging()
    return structlog.get_logger(name)

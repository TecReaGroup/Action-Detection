"""Logging configuration for the application."""

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one-line JSON objects."""

    _standard_attributes = frozenset(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            (key, value)
            for key, value in record.__dict__.items()
            if key not in self._standard_attributes and key not in {"message", "asctime"}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root logger and return the application logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    return logging.getLogger("python_template")

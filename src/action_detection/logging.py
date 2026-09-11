"""Daily logging in UTC+08:00."""

import logging
import sys
from datetime import datetime, timedelta, timezone

LOCAL_TZ = timezone(timedelta(hours=8))


class LocalFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S +08:00")


def configure_logging() -> None:
    """Configure UTF-8 logging to the terminal."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    formatter = LocalFormatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    sink = logging.StreamHandler(sys.stdout)
    sink.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[sink], force=True)

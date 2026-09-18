"""Daily logging in UTC+08:00."""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOCAL_TZ = timezone(timedelta(hours=8))
LOG_DIR = Path(__file__).resolve().parents[2] / "log"


class DailyFileHandler(logging.FileHandler):
    """Switch log files at the UTC+08:00 calendar boundary."""

    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.date = datetime.now(LOCAL_TZ).date()
        super().__init__(LOG_DIR / f"log_{self.date.isoformat()}.log", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        date = datetime.fromtimestamp(record.created, LOCAL_TZ).date()
        if date != self.date:
            self.close()
            self.date = date
            self.baseFilename = str(LOG_DIR / f"log_{date.isoformat()}.log")
            self.stream = self._open()
        super().emit(record)


class LocalFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S +08:00")


def configure_logging() -> None:
    """Configure UTF-8 terminal and daily persistent logging."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    formatter = LocalFormatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    sink = logging.StreamHandler(sys.stdout)
    sink.setFormatter(formatter)
    daily = DailyFileHandler()
    daily.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[sink, daily], force=True)

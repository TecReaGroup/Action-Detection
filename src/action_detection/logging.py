"""Daily logging in UTC+08:00."""

import logging
from datetime import datetime, timedelta, timezone

from .setting import ROOT

LOCAL_TZ = timezone(timedelta(hours=8))


class LocalFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S +08:00")


class DailyFile(logging.Handler):
    """Choose the log file using each record's local date."""

    def emit(self, record: logging.LogRecord) -> None:
        day = datetime.fromtimestamp(record.created, LOCAL_TZ).strftime("%Y-%m-%d")
        try:
            with (ROOT / "log" / f"log_{day}.log").open("a", encoding="utf-8") as stream:
                stream.write(self.format(record) + "\n")
        except OSError:
            self.handleError(record)


def configure_logging() -> None:
    """Attach console and daily persistent logging."""
    (ROOT / "log").mkdir(exist_ok=True)
    formatter = LocalFormatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    sinks = [logging.StreamHandler(), DailyFile()]
    for sink in sinks:
        sink.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=sinks, force=True)

"""Validated video annotations with half-open millisecond intervals."""

import json
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
VIDEO_DIR = APP_ROOT / "data" / "video"
MODEL_DIR = APP_ROOT / "data" / "model"
FEATURE_DIR = APP_ROOT / "data" / "feature"
TEMP_DIR = APP_ROOT.parent / "temp"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass
class Annotation:
    """Own the positive intervals for one explicitly reviewed video."""

    video: Path
    duration_ms: int
    intervals: list[tuple[int, int]] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.video.with_suffix(self.video.suffix + ".json")

    @classmethod
    def load(cls, video: Path) -> "Annotation":
        """Reject malformed or stale annotations before editing or training."""
        with video.with_suffix(video.suffix + ".json").open(encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict) or document.get("version") != 1:
            raise ValueError(f"Unsupported annotation: {video.name}")
        stat = video.stat()
        if (document.get("video") != video.name
                or document.get("video_size") != stat.st_size
                or document.get("video_mtime_ns") != stat.st_mtime_ns):
            raise ValueError(f"Video changed since annotation: {video.name}")
        duration = document.get("duration_ms")
        if type(duration) is not int or duration <= 0:
            raise ValueError(f"Invalid duration: {video.name}")
        intervals = document.get("positive_intervals_ms")
        if not isinstance(intervals, list):
            raise ValueError(f"Invalid intervals: {video.name}")
        previous_end = 0
        for interval in intervals:
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(value) is not int for value in interval)
                    or not previous_end <= interval[0] < interval[1] <= duration):
                raise ValueError(f"Invalid or overlapping interval: {video.name}")
            previous_end = interval[1]
        return cls(video, duration, [tuple(interval) for interval in intervals])

    def add(self, start: int, end: int) -> None:
        """Merge touching positive intervals into their union."""
        start, end = sorted((max(0, min(start, self.duration_ms)),
                             max(0, min(end, self.duration_ms))))
        if start == end:
            return
        merged: list[tuple[int, int]] = []
        for left, right in sorted([*self.intervals, (start, end)]):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        self.intervals = merged

    def save(self) -> None:
        """Atomically persist explicit review, including all-negative videos."""
        stat = self.video.stat()
        document = {
            "version": 1, "video": self.video.name,
            "video_size": stat.st_size, "video_mtime_ns": stat.st_mtime_ns,
            "duration_ms": self.duration_ms,
            "positive_intervals_ms": self.intervals,
        }
        TEMP_DIR.mkdir(exist_ok=True)
        temporary = TEMP_DIR / f"annotation_{self.video.name}.json"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(self.path)

"""Adaptive One Euro smoothing of observed hand landmarks."""

import numpy as np

from .setting import (
    FEATURE_JOINT_COUNT, KEYPOINT_THRESHOLD, ONE_EURO_BETA,
    ONE_EURO_DERIVATIVE_CUTOFF, ONE_EURO_MIN_CUTOFF, ONE_EURO_RESET_SECONDS,
)


class OneEuroFilter:
    """Reduce stationary jitter while increasing responsiveness with pixel velocity."""

    def __init__(self) -> None:
        self.filtered = np.zeros((FEATURE_JOINT_COUNT, 2), dtype=np.float32)
        self.velocity = np.zeros_like(self.filtered)
        self.active = np.zeros(FEATURE_JOINT_COUNT, dtype=bool)
        self.previous_timestamp: float | None = None

    def update(self, points: np.ndarray, scores: np.ndarray, timestamp: float) -> np.ndarray:
        """Filter finite observations at strictly increasing source timestamps in seconds."""
        visible = scores >= KEYPOINT_THRESHOLD
        dt = 0.0 if self.previous_timestamp is None else timestamp - self.previous_timestamp
        continuing = visible & self.active
        if dt > ONE_EURO_RESET_SECONDS:
            continuing.fill(False)
        fresh = visible & ~continuing
        self.filtered[fresh] = points[fresh]
        self.velocity[fresh] = 0
        if continuing.any():
            derivative_alpha = 2 * np.pi * ONE_EURO_DERIVATIVE_CUTOFF * dt
            derivative_alpha /= 1 + derivative_alpha
            derivative = (points[continuing] - self.filtered[continuing]) / dt
            self.velocity[continuing] += derivative_alpha * (derivative - self.velocity[continuing])
            cutoff = ONE_EURO_MIN_CUTOFF + ONE_EURO_BETA * np.abs(self.velocity[continuing])
            alpha = 2 * np.pi * cutoff * dt
            alpha /= 1 + alpha
            self.filtered[continuing] += alpha * (points[continuing] - self.filtered[continuing])
        self.active = visible
        self.previous_timestamp = timestamp
        smoothed = points.copy()
        smoothed[visible] = self.filtered[visible]
        return smoothed

"""Project paths and shared recognition parameters."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
NEGATIVE_DIR = TRAIN_DIR / "other"
MODEL_DIR = ROOT / "data" / "model"
FEATURE_DIR = ROOT / "data" / "feature"
CONFIG_PATH = ROOT / "config" / "config.toml"
SAMPLE_FPS = 10
MIN_WINDOW_FRAMES = 13


def load_window_frames() -> int:
    """Read the shared window, long enough for the causal model's receptive field."""
    with CONFIG_PATH.open("rb") as stream:
        section = tomllib.load(stream).get("train")
    if not isinstance(section, dict):
        raise ValueError(f"Missing [train] section in {CONFIG_PATH}")
    window_frames = section.get("window_frames")
    if type(window_frames) is not int or window_frames < MIN_WINDOW_FRAMES:
        raise ValueError(f"train.window_frames must be an integer >= {MIN_WINDOW_FRAMES}")
    return window_frames


CLIP_LENGTH = load_window_frames()
KEYPOINT_THRESHOLD = 0.35
ACTION_THRESHOLD = 0.75
HAND_JOINT_COUNT = 21
with CONFIG_PATH.open("rb") as stream:
    HAND_SELECTION = tomllib.load(stream).get("pose", {}).get("hand")
if HAND_SELECTION not in ("left", "right", "both"):
    raise ValueError("pose.hand must be left, right, or both")
HAND_COUNT = 2 if HAND_SELECTION == "both" else 1
FEATURE_JOINT_COUNT = HAND_JOINT_COUNT * HAND_COUNT
LEFT_HAND_START = 91
RIGHT_HAND_START = 112
HAND_EDGES = tuple(
    edge
    for finger in range(5)
    for edge in (
        (0, finger * 4 + 1),
        (finger * 4 + 1, finger * 4 + 2),
        (finger * 4 + 2, finger * 4 + 3),
        (finger * 4 + 3, finger * 4 + 4),
    )
)

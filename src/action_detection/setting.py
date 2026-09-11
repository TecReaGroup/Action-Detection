"""Project paths and shared recognition parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
NEGATIVE_DIR = TRAIN_DIR / "other"
MODEL_DIR = ROOT / "data" / "model"
FEATURE_DIR = ROOT / "data" / "feature"
CHECKPOINT = MODEL_DIR / "thumb_sway.pt"
SAMPLE_FPS = 10
CLIP_LENGTH = 16
KEYPOINT_THRESHOLD = 0.35
ACTION_THRESHOLD = 0.75
HAND_JOINT_COUNT = 21
LEFT_HAND_START = 91
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

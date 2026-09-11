"""RTMLib single-person hand extraction and shared normalization."""

import logging
import shutil
import urllib.request
import zipfile

import numpy as np
from rtmlib import RTMPose

from .setting import HAND_JOINT_COUNT, KEYPOINT_THRESHOLD, LEFT_HAND_START, MODEL_DIR, ROOT

POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"
)
LOGGER = logging.getLogger(__name__)


def prepare_pose_model() -> str:
    """Download the official RTMW whole-body model into the project."""
    destination = MODEL_DIR / "rtmw.onnx"
    if destination.exists():
        return str(destination)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "temp"
    temporary.mkdir(exist_ok=True)
    archive = temporary / "rtmw.zip"
    LOGGER.info("Downloading RTMW model: %s", POSE_URL)
    urllib.request.urlretrieve(POSE_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(".onnx"))
        partial = temporary / "rtmw.onnx"
        with bundle.open(member) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        partial.replace(destination)
    archive.unlink()
    return str(destination)


class HandPose:
    """Extract only the person's anatomical left hand from RTMW landmarks."""

    def __init__(self) -> None:
        self.estimator = RTMPose(
            prepare_pose_model(), model_input_size=(192, 256),
            to_openpose=False, backend="onnxruntime", device="cpu",
        )

    def extract(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return left-hand pixels, scores and normalized (3, 21) features."""
        coordinates, confidence = self.estimator(frame)
        left_hand = slice(LEFT_HAND_START, LEFT_HAND_START + HAND_JOINT_COUNT)
        points = coordinates[0, left_hand].astype(np.float32)
        scores = np.clip(confidence[0, left_hand], 0, 1).astype(np.float32)
        features = np.zeros((3, HAND_JOINT_COUNT), dtype=np.float32)
        visible = scores >= KEYPOINT_THRESHOLD
        scale = float(np.linalg.norm(points[9] - points[0]))
        # A visible wrist, palm anchor and thumb are required for thumb motion.
        if visible[0] and visible[9] and visible[1:5].all() and scale >= 5 and visible.sum() >= 12:
            normalized = np.clip((points - points[0]) / scale, -5, 5)
            normalized[~visible] = 0
            features[:2] = normalized.T
            features[2] = scores * visible
        return points, scores, features

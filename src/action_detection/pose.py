"""RTMLib single-person hand extraction and shared normalization."""

import logging
import shutil
import tomllib
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose, Wholebody

from .setting import (
    CONFIG_PATH,
    HAND_JOINT_COUNT,
    KEYPOINT_THRESHOLD,
    LEFT_HAND_START,
    MODEL_DIR,
    ROOT,
)

LOGGER = logging.getLogger(__name__)


def prepare_pose_model(model_url: str) -> str:
    """Download the official RTMW whole-body model into the project."""
    model_name = Path(urllib.parse.urlparse(model_url).path).stem
    destination = MODEL_DIR / f"{model_name}.onnx"
    if destination.exists():
        return str(destination)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "temp"
    temporary.mkdir(exist_ok=True)
    archive = temporary / f"{model_name}.zip"
    LOGGER.info("Downloading RTMW model: %s", model_url)
    urllib.request.urlretrieve(model_url, archive)
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(".onnx"))
        partial = temporary / f"{model_name}.onnx"
        with bundle.open(member) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        partial.replace(destination)
    archive.unlink()
    return str(destination)


class HandPose:
    """Extract only the person's anatomical left hand from RTMW landmarks."""

    def __init__(self) -> None:
        with CONFIG_PATH.open("rb") as stream:
            section = tomllib.load(stream).get("pose")
        if not isinstance(section, dict):
            raise ValueError(f"Missing [pose] section in {CONFIG_PATH}")
        model_size = section.get("model_size")
        device = section.get("device")
        if not isinstance(model_size, str) or model_size not in Wholebody.MODE:
            raise ValueError(f"pose.model_size must be one of: {', '.join(Wholebody.MODE)}")
        if device not in ("cpu", "cuda"):
            raise ValueError("pose.device must be cpu or cuda")
        if device == "cuda":
            ort.preload_dlls(directory="")
        preset = Wholebody.MODE[model_size]
        input_size = preset["pose_input_size"]
        model_path = prepare_pose_model(preset["pose"])
        self.estimator = RTMPose(
            model_path, model_input_size=tuple(input_size),
            to_openpose=False, backend="onnxruntime", device=device,
        )
        providers = self.estimator.session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in providers:
            LOGGER.error("RTMW CUDA initialization failed; active providers=%s", providers)
            raise RuntimeError("RTMW requires CUDA but CUDAExecutionProvider did not initialize")
        LOGGER.info(
            "RTMW size=%s model=%s input_size=%s device=%s providers=%s",
            model_size, model_path, input_size, device, providers,
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

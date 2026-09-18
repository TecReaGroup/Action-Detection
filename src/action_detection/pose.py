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
from rtmlib import RTMPose, Wholebody, YOLOX

from .setting import (
    CONFIG_PATH,
    FEATURE_JOINT_COUNT,
    HAND_JOINT_COUNT,
    HAND_SELECTION,
    KEYPOINT_THRESHOLD,
    LEFT_HAND_START,
    RIGHT_HAND_START,
    MODEL_DIR,
    ROOT,
)

LOGGER = logging.getLogger(__name__)


def prepare_pose_model(model_url: str) -> str:
    """Download an OpenMMLab detection or pose model into the project."""
    model_name = Path(urllib.parse.urlparse(model_url).path).stem
    destination = MODEL_DIR / f"{model_name}.onnx"
    if destination.exists():
        return str(destination)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "temp"
    temporary.mkdir(exist_ok=True)
    archive = temporary / f"{model_name}.zip"
    LOGGER.info("Downloading detection/pose model: %s", model_url)
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
    """Extract the configured anatomical hands from RTMW landmarks."""

    def __init__(self) -> None:
        with CONFIG_PATH.open("rb") as stream:
            section = tomllib.load(stream).get("pose")
        if not isinstance(section, dict):
            raise ValueError(f"Missing [pose] section in {CONFIG_PATH}")
        model_size = section.get("model_size")
        hand = HAND_SELECTION
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
        detector_path = prepare_pose_model(preset["det"])
        self.detector = YOLOX(
            detector_path, model_input_size=tuple(preset["det_input_size"]),
            backend="onnxruntime", device=device,
        )
        detector_providers = self.detector.session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in detector_providers:
            raise RuntimeError("Human detector CUDAExecutionProvider did not initialize")
        LOGGER.info(
            "Human detector model=%s input_size=%s device=%s providers=%s selection=largest",
            detector_path, preset["det_input_size"], device, detector_providers,
        )
        self.estimator = RTMPose(
            model_path, model_input_size=tuple(input_size),
            to_openpose=False, backend="onnxruntime", device=device,
        )
        self.hand = hand
        LOGGER.info("Pose hand=%s", hand)
        providers = self.estimator.session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in providers:
            LOGGER.error("RTMW CUDA initialization failed; active providers=%s", providers)
            raise RuntimeError("RTMW requires CUDA but CUDAExecutionProvider did not initialize")
        LOGGER.info(
            "RTMW size=%s model=%s input_size=%s device=%s providers=%s",
            model_size, model_path, input_size, device, providers,
        )

    def extract(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect the largest person and return hand pixels, scores and features."""
        boxes = np.asarray(self.detector(frame), dtype=np.float32)
        if len(boxes) == 0:
            return (
                np.zeros((FEATURE_JOINT_COUNT, 2), dtype=np.float32),
                np.zeros(FEATURE_JOINT_COUNT, dtype=np.float32),
                np.zeros((3, FEATURE_JOINT_COUNT), dtype=np.float32),
            )
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        person_box = boxes[int(np.argmax(areas))]
        coordinates, confidence = self.estimator(frame, bboxes=[person_box])
        starts = {"left": (LEFT_HAND_START,), "right": (RIGHT_HAND_START,),
                  "both": (LEFT_HAND_START, RIGHT_HAND_START)}[self.hand]
        point_sets = [coordinates[0, start:start + HAND_JOINT_COUNT].astype(np.float32)
                      for start in starts]
        score_sets = [np.clip(confidence[0, start:start + HAND_JOINT_COUNT], 0, 1).astype(np.float32)
                      for start in starts]
        points = np.concatenate(point_sets)
        scores = np.concatenate(score_sets)
        features = np.zeros((3, HAND_JOINT_COUNT * len(starts)), dtype=np.float32)
        for hand_index, (hand_points, hand_scores) in enumerate(zip(point_sets, score_sets)):
            offset = hand_index * HAND_JOINT_COUNT
            hand_features = features[:, offset:offset + HAND_JOINT_COUNT]
            visible = hand_scores >= KEYPOINT_THRESHOLD
            scale = float(np.linalg.norm(hand_points[9] - hand_points[0]))
            if visible[0] and visible[9] and visible[1:5].all() and scale >= 5 and visible.sum() >= 12:
                normalized = np.clip((hand_points - hand_points[0]) / scale, -5, 5)
                normalized[~visible] = 0
                hand_features[:2] = normalized.T
                hand_features[2] = hand_scores * visible
        if any(np.count_nonzero(features[2, offset:offset + HAND_JOINT_COUNT]) < 12
               for offset in range(0, features.shape[1], HAND_JOINT_COUNT)):
            features.fill(0)
        return points, scores, features

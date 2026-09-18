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

from .smoothing import OneEuroFilter
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
    ONE_EURO_MIN_CUTOFF,
    ONE_EURO_BETA,
    ONE_EURO_DERIVATIVE_CUTOFF,
)

LOGGER = logging.getLogger(__name__)
POSE_DIAGNOSTIC_INTERVAL = 100


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
            "Human detector model=%s input_size=%s device=%s providers=%s selection=hand_confidence",
            detector_path, preset["det_input_size"], device, detector_providers,
        )
        self.estimator = RTMPose(
            model_path, model_input_size=tuple(input_size),
            to_openpose=False, backend="onnxruntime", device=device,
        )
        self.hand = hand
        self.reset()
        LOGGER.info("Pose hand=%s", hand)
        LOGGER.info(
            "Hand One Euro min_cutoff=%.2f beta=%.3f derivative_cutoff=%.2f velocity_unit=pixel/s",
            ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_DERIVATIVE_CUTOFF,
        )
        providers = self.estimator.session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in providers:
            LOGGER.error("RTMW CUDA initialization failed; active providers=%s", providers)
            raise RuntimeError("RTMW requires CUDA but CUDAExecutionProvider did not initialize")
        LOGGER.info(
            "RTMW size=%s model=%s input_size=%s device=%s providers=%s",
            model_size, model_path, input_size, device, providers,
        )

    def reset(self) -> None:
        """Start an independent video or seek without carrying previous landmarks."""
        self.previous_timestamp: float | None = None
        self.smoother = OneEuroFilter()
        self.observed_at: float | None = None
        self.sample_count = 0
        self.full_frame_count = 0
        self.person_count = 0
        self.observation_count = 0

    def extract(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return smoothed hand pixels and features at an increasing source time in seconds."""
        if not np.isfinite(timestamp) or (
            self.previous_timestamp is not None and timestamp <= self.previous_timestamp
        ):
            raise ValueError("Pose timestamps must be finite and strictly increasing; reset after seeking")
        self.previous_timestamp = timestamp
        boxes = np.asarray(self.detector(frame), dtype=np.float32)
        self.full_frame_count += int(len(boxes) == 0)
        starts = {"left": (LEFT_HAND_START,), "right": (RIGHT_HAND_START,),
                  "both": (LEFT_HAND_START, RIGHT_HAND_START)}[self.hand]
        points = np.zeros((FEATURE_JOINT_COUNT, 2), dtype=np.float32)
        scores = np.zeros(FEATURE_JOINT_COUNT, dtype=np.float32)
        # RTMLib falls back to the full frame when the detector returns no boxes.
        coordinates, confidence = self.estimator(frame, bboxes=boxes)
        candidates = []
        for index, (person_points, person_scores) in enumerate(zip(coordinates, confidence)):
            selected_points = np.concatenate([person_points[start:start + HAND_JOINT_COUNT] for start in starts])
            selected_scores = np.concatenate([person_scores[start:start + HAND_JOINT_COUNT] for start in starts])
            valid = np.isfinite(selected_points).all(axis=1) & np.isfinite(selected_scores)
            candidates.append((float(np.where(valid, np.clip(selected_scores, 0, 1), 0).sum()), index))
        if candidates:
            person_index = max(candidates)[1]
            points = np.concatenate([coordinates[person_index, start:start + HAND_JOINT_COUNT]
                                     for start in starts]).astype(np.float32)
            scores = np.clip(np.concatenate([confidence[person_index, start:start + HAND_JOINT_COUNT]
                                              for start in starts]), 0, 1).astype(np.float32)
            self.person_count += 1
            valid = np.isfinite(scores) & np.isfinite(points).all(axis=1)
            scores[~valid] = 0
            points[~valid] = 0
        points = self.smoother.update(points, scores, timestamp)
        features = self.normalize(points, scores)
        if np.any(features[2]):
            self.observed_at = timestamp
            self.observation_count += 1
        self.sample_count += 1
        if self.sample_count % POSE_DIAGNOSTIC_INTERVAL == 0:
            LOGGER.info(
                "Pose quality samples=%d full_frame_fallback=%d person_selected=%d clear_hand=%d unavailable=%d",
                self.sample_count, self.full_frame_count, self.person_count,
                self.observation_count, self.sample_count - self.observation_count,
            )
        return points, scores, features

    @staticmethod
    def normalize(points: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Normalize available joints independently without rejecting incomplete hands."""
        hand_count = len(points) // HAND_JOINT_COUNT
        point_sets = np.split(points, hand_count)
        score_sets = np.split(scores, hand_count)
        features = np.zeros((3, len(points)), dtype=np.float32)
        for hand_index, (hand_points, hand_scores) in enumerate(zip(point_sets, score_sets)):
            offset = hand_index * HAND_JOINT_COUNT
            hand_features = features[:, offset:offset + HAND_JOINT_COUNT]
            visible = (hand_scores >= KEYPOINT_THRESHOLD) & np.isfinite(hand_scores) & np.isfinite(hand_points).all(axis=1)
            if visible.any():
                origin = hand_points[0] if visible[0] else hand_points[visible].mean(axis=0)
                scale = (
                    float(np.linalg.norm(hand_points[9] - hand_points[0]))
                    if visible[0] and visible[9]
                    else float(np.linalg.norm(np.ptp(hand_points[visible], axis=0)))
                )
                normalized = np.clip((hand_points - origin) / max(scale, 1.0), -5, 5)
                normalized[~visible] = 0
                hand_features[:2] = normalized.T
                hand_features[2] = np.where(visible, hand_scores, 0)
        return features

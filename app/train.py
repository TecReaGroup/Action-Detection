"""Train shared action architectures using only app-local annotated videos."""

import hashlib
import logging
import math
import tomllib

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from action_detection.logging import configure_logging
from action_detection.pose import HandPose
from action_detection.setting import CLIP_LENGTH, CONFIG_PATH, POSE_FEATURE_VERSION, SAMPLE_FPS
from action_detection.temporal import load_temporal_model
from app.annotation import FEATURE_DIR, MODEL_DIR, TEMP_DIR, VIDEO_DIR, VIDEO_SUFFIXES, Annotation

LOGGER = logging.getLogger(__name__)
BATCH_SIZE = 16
FEATURE_VERSION = f"annotation-{POSE_FEATURE_VERSION}"


def extract_features(annotation: Annotation, pose: HandPose) -> tuple[np.ndarray, np.ndarray]:
    """Cache sampled landmarks with their original timestamps and pose identity."""
    stat = annotation.video.stat()
    signature = (
        f"{FEATURE_VERSION}:{annotation.video}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{pose.estimator.onnx_model}:{SAMPLE_FPS}"
    )
    cache = FEATURE_DIR / f"{hashlib.sha256(signature.encode()).hexdigest()}.npz"
    if cache.exists():
        with np.load(cache) as stored:
            return stored["feature"], stored["timestamp_ms"]
    capture = cv2.VideoCapture(str(annotation.video))
    pose.reset()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {annotation.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS: {annotation.video}")
        frames = []
        timestamps = []
        frame_index = 0
        next_sample = 0.0
        previous_ms = -1.0
        while True:
            available, image = capture.read()
            if not available:
                break
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(timestamp_ms) or timestamp_ms <= previous_ms:
                timestamp_ms = frame_index * 1000 / fps
            if timestamp_ms <= previous_ms:
                raise ValueError(f"Non-monotonic video timestamps: {annotation.video}")
            previous_ms = timestamp_ms
            frame_index += 1
            if timestamp_ms + 1e-6 < next_sample:
                continue
            _, _, feature = pose.extract(image, timestamp_ms / 1000)
            frames.append(feature)
            timestamps.append(timestamp_ms)
            next_sample = (math.floor(timestamp_ms * SAMPLE_FPS / 1000) + 1) * 1000 / SAMPLE_FPS
        if not frames:
            raise ValueError(f"No decoded frames: {annotation.video}")
        if abs(previous_ms + 1000 / fps - annotation.duration_ms) > max(1000, 2000 / fps):
            raise ValueError(f"Decoded duration differs from annotation: {annotation.video}")
        sequence = np.stack(frames, axis=1)
        timestamp_array = np.asarray(timestamps)
        partial = TEMP_DIR / cache.name
        np.savez_compressed(partial, feature=sequence, timestamp_ms=timestamp_array)
        partial.replace(cache)
        LOGGER.info("Extracted %s: %d frames", annotation.video.name, len(frames))
        return sequence, timestamp_array
    finally:
        capture.release()


def annotated_clips(
    annotation: Annotation, sequence: np.ndarray, timestamps: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Keep temporally continuous windows within annotation boundaries."""
    clips = []
    labels = []
    for offset in range(0, len(timestamps) - CLIP_LENGTH + 1, max(1, CLIP_LENGTH // 4)):
        stop = offset + CLIP_LENGTH
        start_ms, end_ms = timestamps[offset], timestamps[stop - 1] + 1000 / SAMPLE_FPS
        if end_ms > annotation.duration_ms + 1:
            continue
        if np.any(np.diff(timestamps[offset:stop]) > 2000 / SAMPLE_FPS):
            continue
        positive = any(left <= start_ms and end_ms <= right
                       for left, right in annotation.intervals)
        overlaps = any(left < end_ms and right > start_ms for left, right in annotation.intervals)
        if overlaps and not positive:
            continue
        clip = sequence[:, offset:stop]
        clips.append(clip.copy())
        labels.append(float(positive))
    return clips, labels


def train_annotations() -> None:
    """Fit a binary action classifier and save it exclusively under app/data/model."""
    with CONFIG_PATH.open("rb") as stream:
        configuration = tomllib.load(stream)
    epochs = configuration.get("train", {}).get("epochs")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("train.epochs must be a positive integer")
    annotations = [Annotation.load(video) for video in sorted(VIDEO_DIR.iterdir())
                   if video.suffix.lower() in VIDEO_SUFFIXES
                   and video.with_suffix(video.suffix + ".json").exists()]
    if not annotations:
        raise ValueError("No reviewed videos in app/data/video")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    for folder in (MODEL_DIR, FEATURE_DIR, TEMP_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    temporal = load_temporal_model()
    pose = HandPose()
    clips = []
    labels = []
    for annotation in annotations:
        sequence, timestamps = extract_features(annotation, pose)
        video_clips, video_labels = annotated_clips(annotation, sequence, timestamps)
        clips.extend(video_clips)
        labels.extend(video_labels)
        LOGGER.info("%s: positive=%d negative=%d", annotation.video.name,
                    sum(video_labels), len(video_labels) - sum(video_labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Both classes need complete visible-hand windows ({CLIP_LENGTH} frames): "
            f"positive={int(positives)}, negative={int(negatives)}"
        )
    device = temporal.device
    del pose
    network = temporal.create_training_network().to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(np.stack(clips)), torch.tensor(labels)),
                        batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    LOGGER.info("Training model=%s epochs=%d device=%s samples=%d; final-epoch checkpoint",
                temporal.name, epochs, device, len(labels))
    for epoch in range(epochs):
        network.train()
        total_loss = 0.0
        for clip, target in loader:
            clip, target = clip.to(device), target.to(device)
            optimizer.zero_grad()
            logits = network.training_logits(clip)
            loss = criterion(logits, target[:, None].expand_as(logits))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            total_loss += loss.item() * len(target)
        LOGGER.info("Epoch %d/%d train_loss=%.6f", epoch + 1, epochs, total_loss / len(labels))
    checkpoint = {
        "version": 1, "state_dict": network.cpu().state_dict(),
        "temporal_model": temporal.name, "clip_length": CLIP_LENGTH,
        "sample_fps": SAMPLE_FPS, "feature_version": FEATURE_VERSION,
        "pose": configuration["pose"], "epochs": epochs,
        "annotation": [str(annotation.path) for annotation in annotations],
        "positive_intervals_ms": {annotation.video.name: annotation.intervals
                                  for annotation in annotations},
        "checkpoint_selection": "final_epoch", "action": "annotated_action",
    }
    destination = MODEL_DIR / f"action_{temporal.name}.pt"
    partial = TEMP_DIR / f"app_action_{temporal.name}.partial"
    torch.save(checkpoint, partial)
    partial.replace(destination)
    LOGGER.info("Saved model: %s", destination)


if __name__ == "__main__":
    configure_logging()
    try:
        train_annotations()
    except Exception:
        LOGGER.exception("Annotation training failed")
        raise SystemExit(1) from None

"""Extract video features and train the binary hand action classifier."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .pose import HandPose
from .setting import CLIP_LENGTH, NEGATIVE_DIR, POSE_FEATURE_VERSION, TRAIN_DIR
from .temporal import load_temporal_model

LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
FEATURE_VERSION = POSE_FEATURE_VERSION
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42


def extract_video(video: Path, pose: HandPose) -> np.ndarray:
    """Extract pose features from every decoded video frame."""
    LOGGER.info("Extracting pose from every video frame: %s", video)
    capture = cv2.VideoCapture(str(video))
    pose.reset()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open training video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video frame rate: {video}")
        frames = []
        index = 0
        while True:
            available, image = capture.read()
            if not available:
                break
            timestamp = index / fps
            index += 1
            _, _, feature = pose.extract(image, timestamp)
            frames.append(feature)
        if not frames:
            raise ValueError(f"Training video has no frames: {video}")
        sequence = np.stack(frames, axis=1)
        LOGGER.info("Extracted %s: %d source frames source_fps=%.3f", video, sequence.shape[1], fps)
        return sequence
    finally:
        capture.release()


def video_clips(sequence: np.ndarray) -> list[np.ndarray]:
    """Keep non-overlapping complete windows, including missing hand observations."""
    clips = []
    for start in range(0, sequence.shape[1] - CLIP_LENGTH + 1, CLIP_LENGTH):
        clip = sequence[:, start:start + CLIP_LENGTH].copy()
        clips.append(clip)
    return clips


def build_examples(
    positive_clips: list[np.ndarray], negative_clips: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label target-action clips as one and real negative clips as zero."""
    examples = np.stack(positive_clips + negative_clips)
    labels = torch.cat((torch.ones(len(positive_clips)), torch.zeros(len(negative_clips))))
    return torch.from_numpy(examples), labels


def find_videos(folder: Path) -> list[Path]:
    """Find at least one source video in the class directory."""
    if not folder.is_dir():
        raise ValueError(f"Training folder does not exist: {folder}")
    videos = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No training videos found: {folder}")
    return videos


def extract_class_clips(
    videos: list[Path], pose: HandPose,
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    """Pool all clips in a class and randomly split them into train and holdout."""
    class_clips = []
    for video in videos:
        clips = video_clips(extract_video(video, pose))
        LOGGER.info("%s: %d non-overlapping clips window_frames=%d stride_frames=%d", video, len(clips), CLIP_LENGTH, CLIP_LENGTH)
        class_clips.extend((video, clip) for clip in clips)
        if not clips:
            LOGGER.warning("Skipping video without a complete clip: %s", video)
    if not class_clips:
        raise ValueError(f"No usable hand clips found: {videos[0].parent}")
    shuffled_indices = np.random.default_rng(SPLIT_SEED).permutation(len(class_clips))
    training_count = max(1, int(len(class_clips) * TRAIN_FRACTION))
    training_clips = [class_clips[index][1] for index in shuffled_indices[:training_count]]
    holdout_indices = shuffled_indices[training_count:]
    holdout_clips = [class_clips[index][1] for index in holdout_indices]
    holdout_videos = sorted({str(class_clips[index][0]) for index in holdout_indices})
    LOGGER.info(
        "Class clip split %s: seed=%d total_clips=%d "
        "train_clips=%d holdout_clips=%d target_train_fraction=%.2f",
        videos[0].parent, SPLIT_SEED, len(class_clips),
        len(training_clips), len(holdout_clips), TRAIN_FRACTION,
    )
    if not holdout_clips:
        LOGGER.warning("Only one clip available; no holdout for class: %s", videos[0].parent)
    return training_clips, holdout_clips, "; ".join(holdout_videos) or "unavailable"


def train_model(action: str, epochs: int, batch_size: int) -> None:
    """Train with available holdouts and persist the selected checkpoint."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    temporal = load_temporal_model()
    positive_videos = find_videos(TRAIN_DIR / action)
    negative_videos = find_videos(NEGATIVE_DIR)
    LOGGER.info("Training sources: positive_videos=%d negative_videos=%d", len(positive_videos), len(negative_videos))
    pose = HandPose()
    positive_train, positive_holdout, positive_video = extract_class_clips(positive_videos, pose)
    negative_train, negative_holdout, negative_video = extract_class_clips(negative_videos, pose)
    train_x, train_y = build_examples(positive_train, negative_train)
    complete_holdout = bool(positive_holdout and negative_holdout)
    validation_x, validation_y = (
        build_examples(positive_holdout, negative_holdout)
        if positive_holdout or negative_holdout
        else (train_x[:0], train_y[:0])
    )
    if not complete_holdout:
        LOGGER.warning("Holdout lacks a class; saving the final epoch without validation-based selection")
    LOGGER.info("Real training clips: positive=%d negative=%d", len(positive_train), len(negative_train))
    LOGGER.info("Holdout: positive=%s (%d clips), negative=%s (%d clips)", positive_video, len(positive_holdout), negative_video, len(negative_holdout))
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    network = temporal.create_training_network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(len(negative_train) / len(positive_train), device=temporal.device),
    )
    validation_criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_loss = float("inf")
    for epoch in range(epochs):
        network.train()
        total_loss = 0.0
        for clip, target in loader:
            clip, target = clip.to(temporal.device), target.to(temporal.device)
            optimizer.zero_grad()
            logits = network.training_logits(clip)
            loss = criterion(logits, target[:, None].expand_as(logits))
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            total_loss += loss.item() * len(target)
        network.eval()
        validation_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for start in range(0, len(validation_y), batch_size):
                target = validation_y[start:start + batch_size].to(temporal.device)
                logits = network.training_logits(validation_x[start:start + batch_size].to(temporal.device))
                clip_loss = validation_criterion(logits, target[:, None].expand_as(logits)).mean(1)
                class_weight = torch.where(
                    target.bool(), 1 / max(len(positive_holdout), 1),
                    1 / max(len(negative_holdout), 1),
                )
                class_weight /= max(bool(positive_holdout) + bool(negative_holdout), 1)
                validation_loss += (clip_loss * class_weight).sum().item()
                correct += ((logits.sigmoid().mean(1) >= 0.5) == target.bool()).sum().item()
        LOGGER.info("Epoch %d/%d train_loss=%.4f holdout_loss=%.4f holdout_accuracy=%.3f", epoch + 1, epochs, total_loss / len(train_y), validation_loss if len(validation_y) else float("nan"), correct / len(validation_y) if len(validation_y) else float("nan"))
        if (complete_holdout and validation_loss < best_loss) or (not complete_holdout and epoch == epochs - 1):
            best_loss = validation_loss
            checkpoint = {
                "version": 2, "state_dict": network.state_dict(), "action": action,
                "temporal_model": temporal.name, "clip_length": CLIP_LENGTH,
                "frame_sampling": "source_frames", "feature_version": FEATURE_VERSION,
                "synthetic_negative": False,
                "negative_directory": str(NEGATIVE_DIR),
                "holdout_video": [str(positive_video), str(negative_video)],
                "split_strategy": "stratified_random_clip",
                "split_seed": SPLIT_SEED,
                "train_fraction": TRAIN_FRACTION,
                "holdout_loss": best_loss if len(validation_y) else None,
                "complete_holdout": complete_holdout,
                "checkpoint_selection": "holdout_loss" if complete_holdout else "final_epoch",
            }
            partial = temporal.checkpoint_path.with_suffix(".partial")
            torch.save(checkpoint, partial)
            partial.replace(temporal.checkpoint_path)
    LOGGER.info("Saved model: %s", temporal.checkpoint_path)

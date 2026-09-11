"""Video-only inference preview for the annotation application."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from action_detection.pose import HandPose
from action_detection.setting import ACTION_THRESHOLD, CLIP_LENGTH, FEATURE_JOINT_COUNT, MODEL_DIR
from action_detection.temporal import load_temporal_model
from app.annotation import VIDEO_DIR, VIDEO_SUFFIXES

LOGGER = logging.getLogger(__name__)


class ConfidenceVideo(QWidget):
    """Render the video and a color-coded confidence label."""

    def __init__(self) -> None:
        super().__init__()
        self.confidence = 0.0
        self.caption = "Confidence: --"
        self.video = QVideoWidget(self)
        self.video.setGeometry(self.rect())
        self.video.setStyleSheet("background: #16181c;")

    def resizeEvent(self, event) -> None:
        self.video.setGeometry(self.rect())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        color = QColor("#48d597") if self.confidence > 0.8 else QColor("#f0c94b") if self.confidence > 0.6 else QColor("white")
        painter.setPen(QPen(color, 2))
        painter.drawText(20, 38, self.caption)
        painter.end()


class PreviewWindow(QWidget):
    """Play a selected app video and infer its trailing action window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Action Preview")
        self.resize(1000, 700)
        self.capture: cv2.VideoCapture | None = None
        self.pose: HandPose | None = None
        self.network = None
        self.window: list[np.ndarray] = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.infer_frame)
        self.player = QMediaPlayer(self)
        self.player.positionChanged.connect(self.sync_capture)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.video_choice = QComboBox()
        controls.addWidget(self.video_choice, 1)
        self.play = QPushButton("Start")
        self.play.clicked.connect(self.start)
        controls.addWidget(self.play)
        layout.addLayout(controls)
        self.display = ConfidenceVideo()
        layout.addWidget(self.display, 1)
        self.refresh()

    def refresh(self) -> None:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        for path in sorted(VIDEO_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                self.video_choice.addItem(path.name, path)

    def start(self) -> None:
        video = self.video_choice.currentData()
        if video is None:
            return
        self.player.setVideoOutput(self.display.video)
        self.player.setSource(QUrl.fromLocalFile(str(video)))
        self.player.play()
        self.pose = HandPose()
        temporal = load_temporal_model()
        checkpoint = MODEL_DIR / f"action_{temporal.name}.pt"
        if not checkpoint.exists():
            checkpoint = temporal.checkpoint_path
        if not checkpoint.exists():
            self.display.caption = "Confidence: model unavailable"
            self.display.update()
            return
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("clip_length") != CLIP_LENGTH or state.get("feature_version", "").split("-")[1:2] != ["both"] and FEATURE_JOINT_COUNT == 42:
            raise ValueError("Checkpoint does not match current hand configuration")
        self.inference_device = temporal.device
        self.network = temporal.network_type().to(self.inference_device).eval()
        self.network.load_state_dict(state["state_dict"])
        self.capture = cv2.VideoCapture(str(video))
        self.window.clear()
        self.timer.start(100)

    def sync_capture(self, position: int) -> None:
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, position)

    def infer_frame(self) -> None:
        if self.capture is None or self.pose is None or self.network is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.timer.stop()
            return
        _, _, features = self.pose.extract(frame)
        if np.count_nonzero(features[2]) < 12 * (features.shape[1] // 21):
            self.window.clear()
            self.display.confidence = 0.0
            self.display.caption = "Confidence: hand unavailable"
        else:
            self.window.append(features)
            self.window = self.window[-CLIP_LENGTH:]
            if len(self.window) == CLIP_LENGTH:
                with torch.inference_mode():
                    window = torch.from_numpy(np.stack(self.window, axis=1)[None]).to(self.inference_device)
                    confidence = self.network.training_logits(window).sigmoid().mean().item()
                self.display.confidence = confidence
                self.display.caption = f"Confidence: {confidence:.1%}"
        self.display.update()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.player.stop()
        if self.capture is not None:
            self.capture.release()
        event.accept()

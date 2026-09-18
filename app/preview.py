"""Synchronized video recognition on a single painted canvas."""

import logging
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QRect, QRectF, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from action_detection.pose import HandPose
from action_detection.overlay import draw_confidence_card
from action_detection.render import draw_hand_skeleton
from action_detection.setting import (
    ACTION_THRESHOLD, CLIP_LENGTH,
    HAND_SELECTION, POSE_FEATURE_VERSION, SAMPLE_FPS,
    STREAM_RETENTION_SECONDS,
)
from action_detection.temporal import load_temporal_model
from app.annotation import MODEL_DIR

LOGGER = logging.getLogger(__name__)
DISPLAY_INTERVAL_MS = 16


class VideoRecognition(QThread):
    """Own decoding, inference and a bounded mailbox of synchronized frames."""

    status = Signal(str)
    failed = Signal(str)
    duration = Signal(int)

    def __init__(self, video: Path, position: int, parent: QWidget) -> None:
        super().__init__(parent)
        self.video = video
        self.lock = threading.Lock()
        self.latest: tuple[QImage, str, float | None, float] | None = None
        self.stopped = threading.Event()
        self.seek_position: int | None = position
        self.generation = 0
        self.completed = False

    def seek(self, position: int) -> None:
        """Discard queued frames and request a decoder seek on the worker thread."""
        with self.lock:
            self.seek_position = position
            self.generation += 1
            self.latest = None

    def stop(self) -> None:
        """Cancel playback, including any wait for the next presentation time."""
        self.stopped.set()

    def take_frame(self) -> tuple[QImage, str, float | None, float] | None:
        """Consume the latest complete frame without queuing image signals."""
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def run(self) -> None:
        """Validate app weights before opening the video and loading pose inference."""
        capture = None
        try:
            self.status.emit("正在加载动作模型…")
            temporal = load_temporal_model()
            checkpoint_path = MODEL_DIR / f"action_{temporal.name}.pt"
            if not checkpoint_path.is_file():
                raise ValueError(f"未找到标注训练模型：{checkpoint_path}。请先点击 Train。")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            expected = {
                "version": 1, "temporal_model": temporal.name,
                "clip_length": CLIP_LENGTH, "sample_fps": SAMPLE_FPS,
                "feature_version": f"annotation-{POSE_FEATURE_VERSION}",
            }
            if not isinstance(checkpoint, dict):
                raise ValueError(f"无效模型文件：{checkpoint_path}")
            mismatch = [f"{key}: 模型={checkpoint.get(key)!r}, 当前={value!r}"
                        for key, value in expected.items() if checkpoint.get(key) != value]
            if mismatch:
                raise ValueError("模型与当前训练配置不匹配，请重新 Train。" + "；".join(mismatch))
            network = temporal.network_type().to(temporal.device).eval()
            network.load_state_dict(checkpoint["state_dict"])
            LOGGER.info("Loaded preview checkpoint=%s hand=%s", checkpoint_path, HAND_SELECTION)
            if self.stopped.is_set():
                return
            self.status.emit("正在加载手部骨架模型…")
            pose = HandPose()
            if self.stopped.is_set():
                return
            capture = cv2.VideoCapture(str(self.video))
            if not capture.isOpened():
                raise ValueError(f"无法打开视频：{self.video}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"视频帧率无效：{fps}")
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if math.isfinite(frame_count) and frame_count > 0:
                self.duration.emit(round(frame_count * 1000 / fps))
            LOGGER.info("Preview started video=%s fps=%.3f sample_fps=%d", self.video, fps, SAMPLE_FPS)
            self.status.emit("正在播放并识别；推理较慢时自动减速")
            window: deque[np.ndarray] = deque(maxlen=CLIP_LENGTH)
            last_valid = None
            frame_index = 0
            previous_ms = -1.0
            next_sample_ms = 0.0
            previous_sample_ms = None
            presented_at = time.monotonic()
            with torch.inference_mode():
                while not self.stopped.is_set():
                    with self.lock:
                        seek_position, self.seek_position = self.seek_position, None
                        generation = self.generation
                    if seek_position is not None:
                        if not capture.set(cv2.CAP_PROP_POS_MSEC, seek_position):
                            raise RuntimeError(f"无法跳转到 {seek_position} ms")
                        frame_index = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
                        previous_ms = -1.0
                        next_sample_ms = 0.0
                        previous_sample_ms = None
                        pose.reset()
                        last_valid = None
                        window.clear()
                        presented_at = time.monotonic()
                        LOGGER.info("Preview seek video=%s position_ms=%d", self.video, seek_position)
                    available, frame = capture.read()
                    if not available:
                        if frame_index == 0:
                            raise ValueError(f"视频没有可解码画面：{self.video}")
                        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                        if math.isfinite(frame_count) and frame_count - frame_index > 1:
                            raise ValueError(f"视频解码提前中断：{frame_index}/{frame_count:.0f} 帧")
                        self.status.emit("播放完成")
                        self.completed = True
                        LOGGER.info("Preview completed video=%s frames=%d", self.video, frame_index)
                        break
                    timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    if not math.isfinite(timestamp_ms) or timestamp_ms <= previous_ms:
                        timestamp_ms = frame_index * 1000 / fps
                    if timestamp_ms <= previous_ms:
                        raise ValueError("视频时间戳不连续，无法同步识别")
                    previous_ms = timestamp_ms
                    frame_index += 1
                    if timestamp_ms + 1e-6 < next_sample_ms:
                        continue
                    next_sample_ms = (math.floor(timestamp_ms * SAMPLE_FPS / 1000) + 1) * 1000 / SAMPLE_FPS
                    timestamp = timestamp_ms / 1000
                    points, scores, features = pose.extract(frame, timestamp)
                    if last_valid is not None and timestamp - last_valid > STREAM_RETENTION_SECONDS:
                        window.clear()
                        last_valid = None
                        LOGGER.info("Action history expired after missing hand observations")
                    if np.any(features[2]):
                        last_valid = pose.observed_at
                    # Repeat the first observation to satisfy the fixed model input window.
                    if not window:
                        window.extend([features] * (CLIP_LENGTH - 1))
                    window.append(features)
                    clip = torch.from_numpy(np.stack(window, axis=1)[None]).to(temporal.device)
                    confidence = network.training_logits(clip).sigmoid().mean().item()
                    if not math.isfinite(confidence):
                        raise RuntimeError("动作模型输出了无效置信度")
                    caption = "识别到目标动作" if confidence >= ACTION_THRESHOLD else "未识别到目标动作"
                    draw_hand_skeleton(frame, points, scores)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                   QImage.Format.Format_RGB888).copy()
                    # Slow playback instead of dropping the temporal samples used in training.
                    interval = 0 if previous_sample_ms is None else (timestamp_ms - previous_sample_ms) / 1000
                    delay = max(0.0, presented_at + interval - time.monotonic())
                    if self.stopped.wait(delay):
                        break
                    with self.lock:
                        if generation == self.generation:
                            self.latest = image, caption, confidence, timestamp_ms
                    presented_at = time.monotonic()
                    previous_sample_ms = timestamp_ms
        except Exception as exc:
            LOGGER.exception("Video preview failed: %s", self.video)
            self.failed.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            LOGGER.info("Preview resources released: %s", self.video)


class VideoCanvas(QWidget):
    """Paint the recognized frame and its caption in one surface."""

    def __init__(self) -> None:
        super().__init__()
        self.image: QImage | None = None
        self.caption = "选择视频后点击 Start"
        self.confidence: float | None = None
        self.setMinimumSize(480, 270)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#16181c"))
        if self.image is not None:
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect((self.width() - size.width()) // 2,
                           (self.height() - size.height()) // 2, size.width(), size.height())
            painter.drawImage(target, self.image)
        draw_confidence_card(painter, QRectF(self.rect()), self.confidence)
        if self.confidence is None:
            painter.setPen(QColor("#d1d5db"))
            painter.drawText(32, 160, self.caption)
        painter.end()


class PreviewPanel(QWidget):
    """Present seekable recognition inside the annotation workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video: Path | None = None
        self.worker: VideoRecognition | None = None
        self.closing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.display = VideoCanvas()
        layout.addWidget(self.display, 1)
        playback = QHBoxLayout()
        self.progress = QSlider(Qt.Orientation.Horizontal)
        self.progress.setRange(0, 0)
        self.progress.sliderReleased.connect(self.seek)
        playback.addWidget(self.progress, 1)
        self.clock = QLabel("00:00.000")
        playback.addWidget(self.clock)
        layout.addLayout(playback)
        controls = QHBoxLayout()
        self.play = QPushButton("Start")
        self.play.clicked.connect(self.start)
        controls.addWidget(self.play)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_frame)
        self.play.setEnabled(False)

    def open_video(self, video: Path | None) -> None:
        """Select the header's video and retire any previous recognition session."""
        if video == self.video:
            return
        self.stop()
        self.video = video
        self.progress.setRange(0, 0)
        self.clock.setText("00:00.000")
        self.display.image = None
        self.display.confidence = None
        self.show_status("点击 Start 开始识别" if video else "请选择视频")
        self.play.setEnabled(video is not None and self.worker is None)

    def show_status(self, caption: str) -> None:
        """Show session messages on the video surface."""
        self.display.caption = caption
        self.display.setToolTip(caption)
        self.display.update()

    def set_duration(self, duration: int) -> None:
        """Enable seeking once the decoder knows the video duration."""
        if self.worker is not None and self.worker.video == self.video:
            self.progress.setMaximum(max(0, duration - round(1000 / SAMPLE_FPS)))

    def seek(self) -> None:
        """Seek an active session or retain the position for the next Start."""
        if self.worker is not None:
            self.worker.seek(self.progress.value())
        self.display.confidence = None
        self.show_status("正在跳转…" if self.worker else "点击 Start 从此处识别")

    def start(self) -> None:
        """Start a fresh session with no retained frames or model state."""
        video = self.video
        if video is None or self.worker is not None:
            return
        self.display.image = None
        self.display.confidence = None
        self.display.caption = "正在启动…"
        self.display.update()
        self.play.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = VideoRecognition(video, self.progress.value(), self)
        self.worker.status.connect(self.show_status)
        self.worker.duration.connect(self.set_duration)
        self.worker.failed.connect(self.show_failure)
        self.worker.finished.connect(self.session_finished)
        self.timer.start(DISPLAY_INTERVAL_MS)
        self.worker.start()

    def refresh_frame(self) -> None:
        """Present pixels, confidence and timestamp from the same inference sample."""
        if self.worker is None:
            return
        latest = self.worker.take_frame()
        if latest is not None and self.worker.video == self.video and not self.progress.isSliderDown():
            self.display.image, self.display.caption, self.display.confidence, timestamp = latest
            minutes, milliseconds = divmod(round(timestamp), 60000)
            self.clock.setText(f"{minutes:02d}:{milliseconds / 1000:06.3f}")
            self.progress.setValue(round(timestamp))
            self.display.update()

    def show_failure(self, message: str) -> None:
        """Keep failures visible without leaving a video playing behind them."""
        self.refresh_frame()
        self.show_status(f"识别失败：{message}")
        self.display.confidence = None
        self.display.update()

    def stop(self) -> None:
        """Request cancellation without blocking the GUI during model loading."""
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.show_status("正在停止并释放资源…")

    def session_finished(self) -> None:
        """Release the finished thread before allowing another session."""
        self.refresh_frame()
        self.timer.stop()
        worker, self.worker = self.worker, None
        assert worker is not None
        worker.deleteLater()
        self.play.setEnabled(self.video is not None)
        self.stop_button.setEnabled(False)
        if worker.stopped.is_set():
            self.show_status("已停止")
        elif worker.completed:
            self.progress.setValue(0)
            self.show_status("播放完成，点击 Start 重新播放")
        if self.closing:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Defer closing until the worker has released decoding and GPU resources."""
        if self.worker is not None:
            self.closing = True
            self.stop()
            event.ignore()
        else:
            self.closing = False
            event.accept()

"""Qt preview with a bounded latest-frame mailbox and streaming inference."""

import importlib.util
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPainter, QColor, QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from .pose import HandPose
from .render import draw_hand_skeleton
from .setting import (
    ACTION_THRESHOLD, CLIP_LENGTH, ROOT,
    SAMPLE_FPS, STREAM_RETENTION_SECONDS,
)
from .temporal import load_temporal_model
from .train import FEATURE_VERSION

LOGGER = logging.getLogger(__name__)


class RecognitionThread(QThread):
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.lock = threading.Lock()
        self.latest: tuple[QImage, str] | None = None

    def take_frame(self) -> tuple[QImage, str] | None:
        """Consume the newest frame without accumulating Qt events."""
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def run(self) -> None:
        camera = None
        try:
            torch.set_num_threads(4)
            temporal = load_temporal_model()
            network = None
            action = ""
            if temporal.checkpoint_path.exists():
                checkpoint = torch.load(temporal.checkpoint_path, map_location="cpu", weights_only=True)
                expected = {
                    "version": 2, "feature_version": FEATURE_VERSION,
                    "sample_fps": SAMPLE_FPS, "clip_length": CLIP_LENGTH,
                    "temporal_model": temporal.name,
                }
                if any(checkpoint.get(key) != value for key, value in expected.items()):
                    raise ValueError("Checkpoint model or preprocessing changed; run training again")
                network = temporal.network_type().to(temporal.device).eval()
                network.load_state_dict(checkpoint["state_dict"])
                action = checkpoint["action"]
                LOGGER.info("Loaded action model: %s", temporal.checkpoint_path)
            else:
                LOGGER.warning("No weights for %s; run make train. Skeleton preview only.", temporal.name)
            self.status.emit("正在加载手部骨架模型…")
            pose = HandPose()
            if self.isInterruptionRequested():
                return
            driver_path = ROOT / "device" / "LogiCamera.py"
            specification = importlib.util.spec_from_file_location("logi_camera", driver_path)
            if specification is None or specification.loader is None:
                raise RuntimeError(f"Cannot load camera driver: {driver_path}")
            driver = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(driver)
            camera = driver.LogiCamera({
                "deviceId": self.camera_index, "colorImageSizeX": 1280,
                "colorImageSizeY": 720, "fps": 30, "frameInterval": 0.01,
            })
            camera.open()
            LOGGER.info("Camera %d opened using %s", self.camera_index, camera.backend)
            next_sample = 0.0
            last_frame = time.monotonic()
            window: deque[np.ndarray] = deque(maxlen=CLIP_LENGTH)
            probability = 0.0
            last_valid = None
            label = "模型未训练"
            with torch.inference_mode():
                while not self.isInterruptionRequested():
                    timestamp, frame = camera.getFrame(timeout=0.1)
                    if frame is None:
                        if time.monotonic() - last_frame > 5:
                            raise RuntimeError("Camera did not provide frames for 5 seconds")
                        continue
                    last_frame = time.monotonic()
                    if timestamp < next_sample:
                        continue
                    if last_valid is not None and timestamp - last_valid > STREAM_RETENTION_SECONDS and network is not None:
                        window.clear()
                        probability = 0.0
                        last_valid = None
                        LOGGER.info("Action history expired after missing hand observations")
                    if not next_sample or timestamp - next_sample > 0.5:
                        next_sample = timestamp + 1 / SAMPLE_FPS
                    else:
                        next_sample += (int((timestamp - next_sample) * SAMPLE_FPS) + 1) / SAMPLE_FPS
                    points, scores, features = pose.extract(frame, timestamp)
                    if network is not None:
                        if np.any(features[2]):
                            last_valid = pose.observed_at
                        if not window:
                            window.extend([features] * (CLIP_LENGTH - 1))
                        window.append(features)
                        clip = torch.from_numpy(np.stack(window, axis=1)[None]).to(temporal.device)
                        probability = network.training_logits(clip).sigmoid().mean().item()
                        if not np.isfinite(probability):
                            raise RuntimeError("Action model returned a non-finite probability")
                        label = action if probability >= ACTION_THRESHOLD else "未识别到目标动作"
                    draw_hand_skeleton(frame, points, scores)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
                    caption = label
                    if network is not None:
                        caption += f"\n{action}置信度：{probability:.1%}"
                    with self.lock:
                        self.latest = (image, caption)
        except Exception as exc:
            LOGGER.exception("Preview failed")
            self.failed.emit(str(exc))
        finally:
            if camera is not None:
                camera.stopThread()
                LOGGER.info("Camera released")


class CameraView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.image: QImage | None = None
        self.caption = "正在启动…"
        self.setMinimumSize(640, 360)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101214"))
        origin_x, origin_y = 0, 0
        if self.image is not None:
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            origin_x = (self.width() - size.width()) // 2
            origin_y = (self.height() - size.height()) // 2
            painter.drawImage(self.rect().adjusted(origin_x, origin_y, -origin_x, -origin_y), self.image)
        painter.setFont(QFont("Microsoft YaHei", 15))
        metrics = painter.fontMetrics()
        lines = self.caption.splitlines()
        width = min(self.width() - 24, max(metrics.horizontalAdvance(line) for line in lines) + 24)
        height = metrics.height() * len(lines) + 16
        painter.fillRect(origin_x + 12, origin_y + 12, width, height, QColor(0, 0, 0, 175))
        painter.setPen(QColor("#f2f4f5"))
        for index, line in enumerate(lines):
            painter.drawText(origin_x + 24, origin_y + 20 + metrics.ascent() + index * metrics.height(), metrics.elidedText(line, Qt.TextElideMode.ElideRight, width - 24))


class PreviewWindow(QMainWindow):
    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self.setWindowTitle("单人动作识别 · 手部骨架")
        self.resize(1100, 700)
        self.view = CameraView()
        self.setCentralWidget(self.view)
        self.worker = RecognitionThread(camera_index)
        self.worker.status.connect(self.show_status)
        self.worker.failed.connect(self.show_status)
        self.worker.finished.connect(self.finish_close)
        self.closing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_frame)
        self.timer.start(33)
        self.worker.start()

    def show_status(self, caption: str) -> None:
        self.view.caption = caption
        self.view.update()
        self.statusBar().showMessage(caption)

    def refresh_frame(self) -> None:
        latest = self.worker.take_frame()
        if latest is not None:
            self.view.image, self.view.caption = latest
            self.view.update()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker.isRunning():
            self.closing = True
            self.timer.stop()
            self.worker.requestInterruption()
            self.show_status("正在释放相机，请稍候…")
            event.ignore()
        else:
            self.timer.stop()
            event.accept()

    def finish_close(self) -> None:
        if self.closing:
            self.close()


def run_preview(camera_index: int) -> int:
    """Run the desktop preview until its window is closed."""
    application = QApplication.instance() or QApplication([])
    window = PreviewWindow(camera_index)
    window.show()
    return application.exec()

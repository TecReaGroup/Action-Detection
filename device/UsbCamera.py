# -*- coding: utf-8 -*-
"""OpenCV driver for standard USB cameras."""

import queue
import threading
import time

import cv2


class _CameraHandleValidator:
    """Provide the handle validation API used by the sampling controller."""

    @staticmethod
    def isValidHandle(handle):
        try:
            return handle is not None and handle.isOpened()
        except Exception:
            return False


class UsbCamera:
    """Camera adapter for standard USB cameras accessed through OpenCV."""

    def __init__(self, cameraConfig):
        self.cameraConfig = cameraConfig
        self.deviceId = cameraConfig.get("deviceId", 0)
        try:
            camera_index = int(self.deviceId)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "USB camera deviceId must be a camera index such as 0 or 1"
            ) from exc

        self.cameraId = str(camera_index)
        self.runStatus = 0
        self.frameQueue = queue.Queue(maxsize=3)

        backend_name = str(cameraConfig.get("backend", "dshow")).lower()
        backend_candidates = []
        if backend_name in ("dshow", "directshow") and hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(("DSHOW", cv2.CAP_DSHOW))
        if backend_name in ("msmf", "microsoft") and hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(("MSMF", cv2.CAP_MSMF))
        backend_candidates.append(("default", None))

        self.cap = None
        opened_backend = None
        for name, backend in backend_candidates:
            cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                self.cap = cap
                opened_backend = name
                break
            cap.release()

        if self.cap is None:
            raise RuntimeError(
                f"无法打开USB摄像头索引: {camera_index}，尝试后端: "
                f"{', '.join(name for name, _ in backend_candidates)}"
            )

        self._configureCapture()
        self.handle = self.cap
        self.cl = _CameraHandleValidator()
        self.backend = opened_backend

        self.thread = threading.Thread(target=self._videoThread, daemon=True)
        self.thread.start()

    def _configureCapture(self):
        width = self.cameraConfig.get("colorImageSizeX")
        height = self.cameraConfig.get("colorImageSizeY")
        fps = self.cameraConfig.get("fps")

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

        fourcc = str(self.cameraConfig.get("fourcc", "MJPG"))[:4]
        if len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    def open(self):
        self.runStatus = 10

    def close(self):
        self.runStatus = 1

    def getFrame(self, timeout=0.01):
        try:
            return self.frameQueue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def stopThread(self):
        self.runStatus = -1
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.handle = None

    def _videoThread(self):
        interval = float(self.cameraConfig.get("frameInterval", 0.033))
        while self.runStatus >= 0:
            if self.runStatus != 10:
                time.sleep(0.05)
                continue

            started = time.perf_counter()
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if self.frameQueue.full():
                    try:
                        self.frameQueue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.frameQueue.put_nowait((time.time(), frame))
                except queue.Full:
                    pass
            else:
                time.sleep(0.02)

            time.sleep(max(0.0, interval - (time.perf_counter() - started)))

"""Launch the independent PySide video annotation workspace."""

import codecs
import logging
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSlider, QStyle,
    QToolButton, QVBoxLayout, QWidget,
)

from action_detection.logging import configure_logging
from app.annotation import APP_ROOT, VIDEO_DIR, VIDEO_SUFFIXES, Annotation
from app.timeline import Timeline

LOGGER = logging.getLogger(__name__)


class AnnotationWindow(QMainWindow):
    """Edit reviewed videos and run isolated training in a child process."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Action Annotation")
        self.resize(1120, 850)
        self.setMinimumSize(760, 650)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #f6f7f9; color: #222831; }"
            "QComboBox, QDoubleSpinBox, QListWidget, QPlainTextEdit {"
            " background: white; border: 1px solid #d4d8df; padding: 5px; }"
            "QPushButton, QToolButton { background: #ffffff; border: 1px solid #cbd1d9;"
            " border-radius: 4px; padding: 6px; }"
            "QPushButton:hover, QToolButton:hover { background: #e7edf4; }"
            "QPushButton:disabled, QToolButton:disabled { color: #9ba1aa; }"
            "QPushButton#train { background: #266ac0; color: white; border: none; }"
            "QPushButton#train:disabled { background: #9ba8b9; }"
        )
        self.annotation: Annotation | None = None
        self.dirty = False
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.training = QProcess(self)
        self.training.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.training.readyReadStandardOutput.connect(self.read_training_log)
        self.training.finished.connect(self.training_finished)
        self.training.errorOccurred.connect(self.training_error)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.preview = QVideoWidget()
        self.preview.setStyleSheet("background: #16181c;")
        self.preview.setMinimumSize(320, 220)
        self.player.setVideoOutput(self.preview)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.positionChanged.connect(self.position_changed)
        self.player.errorOccurred.connect(lambda *_: self.show_error(self.player.errorString()))
        self.player.playbackStateChanged.connect(self.playback_changed)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 14)
        header = QHBoxLayout()
        self.video_choice = QComboBox()
        self.video_choice.setMinimumWidth(280)
        self.video_choice.currentIndexChanged.connect(self.open_video)
        header.addWidget(self.video_choice, 1)
        refresh = self.tool(QStyle.StandardPixmap.SP_BrowserReload, "Refresh video list")
        refresh.clicked.connect(self.refresh_videos)
        header.addWidget(refresh)
        folder = self.tool(QStyle.StandardPixmap.SP_DirOpenIcon, "Open video folder")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(VIDEO_DIR))))
        header.addWidget(folder)
        self.save_button = self.tool(QStyle.StandardPixmap.SP_DialogSaveButton, "Save annotation")
        self.save_button.clicked.connect(self.save_annotation)
        header.addWidget(self.save_button)
        preview_button = QPushButton("Preview")
        preview_button.setToolTip("Open video inference preview")
        preview_button.clicked.connect(self.open_preview)
        header.addWidget(preview_button)
        layout.addLayout(header)
        layout.addWidget(self.preview, 5)

        playback = QHBoxLayout()
        self.play_button = self.tool(QStyle.StandardPixmap.SP_MediaPlay, "Play / pause")
        self.play_button.clicked.connect(self.toggle_playback)
        playback.addWidget(self.play_button)
        self.progress = QSlider(Qt.Orientation.Horizontal)
        self.progress.setRange(0, 0)
        self.progress.sliderPressed.connect(self.player.pause)
        self.progress.sliderMoved.connect(self.player.setPosition)
        self.progress.sliderReleased.connect(lambda: self.player.setPosition(self.progress.value()))
        playback.addWidget(self.progress, 1)
        self.clock = QLabel("00:00.000 / 00:00.000")
        self.clock.setMinimumWidth(190)
        playback.addWidget(self.clock)
        layout.addLayout(playback)
        layout.addWidget(QLabel("Action intervals"))
        self.timeline = Timeline()
        self.timeline.interval_drawn.connect(self.add_interval)
        self.timeline.selected.connect(self.select_interval)
        self.timeline.seek.connect(self.player.setPosition)
        self.timeline.drawing_started.connect(self.player.pause)
        layout.addWidget(self.timeline)

        editing = QHBoxLayout()
        self.interval_list = QListWidget()
        self.interval_list.setMaximumHeight(110)
        self.interval_list.currentRowChanged.connect(self.select_interval)
        editing.addWidget(self.interval_list, 1)
        self.start_time = QDoubleSpinBox()
        self.end_time = QDoubleSpinBox()
        for label, spin in (("Start", self.start_time), ("End", self.end_time)):
            spin.setDecimals(3)
            spin.setSuffix(" s")
            spin.setSingleStep(0.1)
            editing.addWidget(QLabel(label))
            editing.addWidget(spin)
        apply = QPushButton("Apply")
        apply.clicked.connect(self.apply_interval)
        editing.addWidget(apply)
        delete = self.tool(QStyle.StandardPixmap.SP_TrashIcon, "Delete selected interval")
        delete.clicked.connect(self.delete_interval)
        editing.addWidget(delete)
        layout.addLayout(editing)
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumBlockCount(2000)
        self.training_log.setMaximumHeight(120)
        layout.addWidget(self.training_log)
        footer = QHBoxLayout()
        self.status = QLabel("No video selected")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.stop_button = self.tool(QStyle.StandardPixmap.SP_MediaStop, "Stop training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.training.kill)
        footer.addWidget(self.stop_button)
        self.train_button = QPushButton("Train")
        self.train_button.setObjectName("train")
        self.train_button.setMinimumWidth(110)
        self.train_button.clicked.connect(self.start_training)
        footer.addWidget(self.train_button)
        layout.addLayout(footer)
        self.refresh_videos()

    def open_preview(self) -> None:
        """Open the standalone inference window while keeping annotation state."""
        from app.preview import PreviewWindow

        self.preview_window = PreviewWindow(self)
        self.preview_window.show()

    def tool(self, icon: QStyle.StandardPixmap, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(tooltip)
        button.setFixedSize(34, 32)
        return button

    def show_error(self, message: str) -> None:
        LOGGER.error(message)
        self.status.setText(message)
        QMessageBox.warning(self, "Action Annotation", message)

    def refresh_videos(self) -> None:
        if self.dirty and not self.save_annotation():
            return
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        previous = self.video_choice.currentData()
        self.video_choice.blockSignals(True)
        self.video_choice.clear()
        for video in sorted(VIDEO_DIR.iterdir()):
            if video.is_file() and video.suffix.lower() in VIDEO_SUFFIXES:
                self.video_choice.addItem(video.name, video)
        index = self.video_choice.findData(previous)
        self.video_choice.setCurrentIndex(max(0, index))
        self.video_choice.blockSignals(False)
        self.open_video(self.video_choice.currentIndex())

    def open_video(self, index: int) -> None:
        if self.dirty and not self.save_annotation():
            if self.annotation is not None:
                self.video_choice.blockSignals(True)
                self.video_choice.setCurrentIndex(self.video_choice.findData(self.annotation.video))
                self.video_choice.blockSignals(False)
            return
        self.player.stop()
        self.annotation = None
        self.dirty = False
        self.timeline.duration = 0
        self.timeline.position = 0
        self.timeline.anchor = None
        self.refresh_intervals()
        video = self.video_choice.itemData(index)
        if video is None:
            self.status.setText("No videos in app/data/video")
            self.player.setSource(QUrl())
            return
        try:
            if video.with_suffix(video.suffix + ".json").exists():
                self.annotation = Annotation.load(video)
            else:
                self.annotation = Annotation(video, 0)
            self.dirty = False
            self.status.setText("Saved" if self.annotation.path.exists() else "Not annotated")
            self.refresh_intervals()
            self.player.setSource(QUrl.fromLocalFile(str(video)))
            self.player.pause()
            LOGGER.info("Opened video: %s", video)
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))

    def duration_changed(self, duration: int) -> None:
        if self.annotation is None or duration <= 0:
            return
        if self.annotation.intervals and self.annotation.intervals[-1][1] > duration:
            self.show_error("Annotation exceeds video duration")
            self.annotation = None
            return
        self.annotation.duration_ms = duration
        self.timeline.duration = duration
        self.progress.setRange(0, duration)
        self.start_time.setMaximum(duration / 1000)
        self.end_time.setMaximum(duration / 1000)
        self.position_changed(self.player.position())

    @staticmethod
    def timestamp(milliseconds: int) -> str:
        minutes, remainder = divmod(milliseconds, 60000)
        return f"{minutes:02d}:{remainder / 1000:06.3f}"

    def position_changed(self, position: int) -> None:
        if not self.progress.isSliderDown():
            self.progress.setValue(position)
        self.timeline.position = position
        self.timeline.update()
        self.clock.setText(f"{self.timestamp(position)} / {self.timestamp(self.player.duration())}")

    def toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def playback_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        icon = (QStyle.StandardPixmap.SP_MediaPause
                if state == QMediaPlayer.PlaybackState.PlayingState
                else QStyle.StandardPixmap.SP_MediaPlay)
        self.play_button.setIcon(self.style().standardIcon(icon))

    def refresh_intervals(self) -> None:
        self.interval_list.blockSignals(True)
        self.interval_list.clear()
        self.timeline.intervals = list(self.annotation.intervals) if self.annotation else []
        self.timeline.selection = -1
        for start, end in self.timeline.intervals:
            self.interval_list.addItem(f"{self.timestamp(start)} - {self.timestamp(end)}")
        self.interval_list.blockSignals(False)
        self.timeline.update()

    def select_interval(self, index: int) -> None:
        self.timeline.selection = index
        self.interval_list.setCurrentRow(index)
        if self.annotation and 0 <= index < len(self.annotation.intervals):
            start, end = self.annotation.intervals[index]
            self.start_time.setValue(start / 1000)
            self.end_time.setValue(end / 1000)
        self.timeline.update()

    def add_interval(self, start: int, end: int) -> None:
        if self.annotation is not None:
            self.player.pause()
            self.annotation.add(start, end)
            self.dirty = True
            self.refresh_intervals()
            self.save_annotation()

    def apply_interval(self) -> None:
        index = self.timeline.selection
        if self.annotation is None or index < 0:
            return
        start, end = round(self.start_time.value() * 1000), round(self.end_time.value() * 1000)
        if start >= end:
            self.show_error("Start must precede end")
            return
        del self.annotation.intervals[index]
        self.add_interval(start, end)

    def delete_interval(self) -> None:
        if self.annotation is not None and self.timeline.selection >= 0:
            del self.annotation.intervals[self.timeline.selection]
            self.dirty = True
            self.refresh_intervals()
            self.save_annotation()

    def save_annotation(self) -> bool:
        if self.annotation is None or self.annotation.duration_ms <= 0:
            return False
        try:
            self.annotation.save()
            self.dirty = False
            self.status.setText(f"Saved: {self.annotation.path.name}")
            LOGGER.info("Saved %s: %d intervals", self.annotation.path, len(self.annotation.intervals))
            return True
        except OSError as exc:
            self.dirty = True
            self.show_error(str(exc))
            return False

    def start_training(self) -> None:
        if self.dirty and not self.save_annotation():
            return
        if not any(VIDEO_DIR.glob("*.json")):
            self.show_error("No saved annotations")
            return
        self.player.pause()
        self.training_log.clear()
        self.decoder.reset()
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.training.setProcessEnvironment(environment)
        self.training.setWorkingDirectory(str(APP_ROOT.parent))
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Training")
        self.training.start(sys.executable, ["-u", "-m", "app.train"])

    def read_training_log(self) -> None:
        text = self.decoder.decode(bytes(self.training.readAllStandardOutput()))
        cursor = self.training_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.training_log.setTextCursor(cursor)
        self.training_log.ensureCursorVisible()

    def training_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.read_training_log()
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText("Training complete" if exit_code == 0 else "Training stopped or failed")

    def training_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.show_error(self.training.errorString())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.dirty and not self.save_annotation():
            event.ignore()
            return
        if self.training.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(self, "Training", "Stop training and close?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.training.kill()
            self.training.waitForFinished(5000)
        self.player.stop()
        event.accept()


def main() -> int:
    """Open the annotation workspace."""
    configure_logging()
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = AnnotationWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

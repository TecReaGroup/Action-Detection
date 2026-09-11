"""Mouse-drawn action intervals on a shared video time axis."""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget


class Timeline(QWidget):
    """Draw positive spans and emit new half-open millisecond selections."""

    interval_drawn = Signal(int, int)
    selected = Signal(int)
    seek = Signal(int)
    drawing_started = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.duration = 0
        self.position = 0
        self.intervals: list[tuple[int, int]] = []
        self.selection = -1
        self.anchor: int | None = None
        self.cursor_ms = 0
        self.setMinimumHeight(110)
        self.setToolTip("Drag to mark an action interval; click an interval to select it")

    def time_at(self, x: float) -> int:
        return round(max(0, min(1, (x - 12) / max(1, self.width() - 24))) * self.duration)

    def x_at(self, timestamp: int) -> float:
        return 12 + timestamp / max(1, self.duration) * (self.width() - 24)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f4f5f7"))
        painter.fillRect(QRectF(12, 38, self.width() - 24, 44), QColor("#dfe3e7"))
        painter.setPen(QColor("#59616b"))
        for index in range(6):
            stamp = self.duration * index // 5
            x = self.x_at(stamp)
            painter.drawLine(int(x), 30, int(x), 88)
            label = f"{stamp / 1000:.1f}s"
            painter.drawText(int(min(x, self.width() - 65)), 22, label)
        for index, (start, end) in enumerate(self.intervals):
            rectangle = QRectF(self.x_at(start), 38, self.x_at(end) - self.x_at(start), 44)
            painter.fillRect(rectangle, QColor("#359f85"))
            if index == self.selection:
                painter.setPen(QPen(QColor("#17212b"), 2))
                painter.drawRect(rectangle)
        if self.anchor is not None:
            left, right = sorted((self.x_at(self.anchor), self.x_at(self.cursor_ms)))
            painter.fillRect(QRectF(left, 38, right - left, 44), QColor(49, 122, 202, 130))
        painter.setPen(QPen(QColor("#d44b55"), 2))
        painter.drawLine(int(self.x_at(self.position)), 28, int(self.x_at(self.position)), 92)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.duration > 0:
            self.drawing_started.emit()
            self.anchor = self.time_at(event.position().x())
            self.cursor_ms = self.anchor
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.anchor is not None:
            self.cursor_ms = self.time_at(event.position().x())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.anchor is None:
            return
        endpoint = self.time_at(event.position().x())
        start = self.anchor
        self.anchor = None
        if abs(self.x_at(endpoint) - self.x_at(start)) >= 4:
            self.interval_drawn.emit(min(start, endpoint), max(start, endpoint))
        else:
            self.selected.emit(next((index for index, (left, right) in enumerate(self.intervals)
                                     if left <= endpoint < right), -1))
            self.seek.emit(endpoint)
        self.update()

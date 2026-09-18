"""Shared confidence card for recognition previews."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen


def draw_confidence_card(painter: QPainter, bounds: QRectF, confidence: float | None) -> None:
    """Paint the target, confidence and a color-coded progress bar."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    card = QRectF(bounds.x() + 16, bounds.y() + 16, min(290, bounds.width() - 32), 112)
    painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
    painter.setBrush(QColor(17, 24, 34, 218))
    painter.drawRoundedRect(card, 12, 12)
    left, width = card.x() + 16, card.width() - 32
    font = QFont("Microsoft YaHei")
    font.setPixelSize(16)
    font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(font)
    painter.setPen(QColor("#f4f6fa"))
    painter.drawText(QRectF(left, card.y() + 14, width, 24), Qt.AlignmentFlag.AlignVCenter, "目标动作：摇摆大拇指")
    font.setPixelSize(13)
    font.setWeight(QFont.Weight.Normal)
    painter.setFont(font)
    value = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
    color = QColor("#46d99b" if value >= 0.8 else "#f2c94c" if value >= 0.6 else "#d1d5db")
    painter.setPen(color)
    text = f"置信率：{value:.1%}" if confidence is not None else "置信率：—"
    painter.drawText(QRectF(left, card.y() + 44, width, 21), Qt.AlignmentFlag.AlignVCenter, text)
    track = QRectF(left, card.y() + 79, width, 10)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#394350"))
    painter.drawRoundedRect(track, 5, 5)
    if value > 0:
        painter.setClipRect(QRectF(track.x(), track.y(), track.width() * value, track.height()))
        painter.setBrush(color)
        painter.drawRoundedRect(track, 5, 5)
    painter.restore()

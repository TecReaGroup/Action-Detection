"""Shared outlined hand skeleton rendering for camera and video previews."""

import cv2
import numpy as np

from .setting import HAND_EDGES, HAND_JOINT_COUNT, HAND_SELECTION, KEYPOINT_THRESHOLD

LEFT_COLOR = (225, 205, 65)
RIGHT_COLOR = (120, 145, 255)
OUTLINE_COLOR = (35, 35, 35)
JOINT_CENTER_COLOR = (245, 245, 245)
HAND_LINE_WIDTH = 2
HAND_JOINT_RADIUS = 3


def draw_hand_skeleton(frame: np.ndarray, points: np.ndarray, scores: np.ndarray) -> None:
    """Draw anatomical hand colors with resolution-scaled outlines and wrist centers."""
    height, width = frame.shape[:2]
    scale = max(0.5, min(height, width) / 720.0)
    outline = max(1, round(scale))
    thickness = max(1, round(HAND_LINE_WIDTH * scale))
    radius = max(1, round(HAND_JOINT_RADIUS * scale))
    colors = {"left": (LEFT_COLOR,), "right": (RIGHT_COLOR,),
              "both": (LEFT_COLOR, RIGHT_COLOR)}[HAND_SELECTION]
    for hand_index, color in enumerate(colors):
        offset = hand_index * HAND_JOINT_COUNT
        visible = {}
        for index in range(HAND_JOINT_COUNT):
            x, y = points[offset + index]
            if scores[offset + index] >= KEYPOINT_THRESHOLD and 0 <= x < width and 0 <= y < height:
                visible[index] = (min(width - 1, round(float(x))), min(height - 1, round(float(y))))
        for stroke_color, extra in ((OUTLINE_COLOR, 2 * outline), (color, 0)):
            for first, second in HAND_EDGES:
                if first in visible and second in visible:
                    cv2.line(frame, visible[first], visible[second], stroke_color,
                             thickness + extra, cv2.LINE_AA)
        for index, point in visible.items():
            cv2.circle(frame, point, radius + outline, OUTLINE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
            if index == 0:
                cv2.circle(frame, point, max(1, radius // 3), JOINT_CENTER_COLOR, -1, cv2.LINE_AA)

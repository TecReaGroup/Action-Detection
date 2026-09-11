"""Two-dimensional adaptation of SKETCH coordinate plots and range embedding.

Reference: capableofanything/SKETCH, shrec22/draw/shrec22_draw_3stack.py
and shrec22/train/model.py (e79743301725c6c6d5458b4065134e1b37ca1381).
Uses two coordinate panels, 21 joints and a binary head without boundary regressors.

MIT License
Copyright (c) 2025 LEE SEONHO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import deque
import logging
import os

import cv2
import numpy as np
import torch
from torch import nn

from .setting import CLIP_LENGTH, HAND_JOINT_COUNT, KEYPOINT_THRESHOLD, MODEL_DIR

BACKBONE = "swin_tiny_patch4_window7_224.ms_in1k"
IMAGE_SIZE = 224
PANEL_HEIGHT = IMAGE_SIZE // 2
COORDINATE_WIDTH = CLIP_LENGTH * HAND_JOINT_COUNT
# Joint identity is encoded by a fixed RGB palette across all windows.
JOINT_COLORS = (
    (0, 128, 0), (105, 105, 105), (0, 0, 255), (165, 42, 42), (127, 255, 0),
    (210, 105, 30), (255, 127, 80), (220, 20, 60), (138, 43, 226), (0, 0, 139),
    (0, 100, 0), (178, 34, 34), (255, 215, 0), (0, 128, 128), (128, 128, 128),
    (75, 0, 130), (70, 130, 180), (205, 92, 92), (218, 165, 32), (139, 0, 0),
    (255, 140, 0),
)


def coordinate_image(sequence: np.ndarray) -> np.ndarray:
    """Plot visible X/Y trajectories with per-axis window normalization."""
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    visible = sequence[2] >= KEYPOINT_THRESHOLD
    time_axis = np.linspace(0, IMAGE_SIZE - 1, sequence.shape[1]).astype(np.int32)
    for axis in range(2):
        coordinates = sequence[axis]
        valid = coordinates[visible]
        if not valid.size:
            continue
        span = float(valid.max() - valid.min())
        normalized = (coordinates - valid.min()) / max(span, 1e-6)
        height = axis * PANEL_HEIGHT + (1 - normalized.clip(0, 1)) * (PANEL_HEIGHT - 1)
        for joint in range(HAND_JOINT_COUNT):
            for timestamp in range(1, sequence.shape[1]):
                if visible[timestamp - 1, joint] and visible[timestamp, joint]:
                    cv2.line(
                        canvas,
                        (int(time_axis[timestamp - 1]), int(height[timestamp - 1, joint])),
                        (int(time_axis[timestamp]), int(height[timestamp, joint])),
                        JOINT_COLORS[joint], 1, cv2.LINE_AA,
                    )
    return canvas.transpose(2, 0, 1).astype(np.float32) / 255


class Sketch2D(nn.Module):
    """Encode a left-hand temporal window as a coordinate image for Swin."""

    warmup_frames = CLIP_LENGTH

    def __init__(self) -> None:
        super().__init__()
        cache = MODEL_DIR / "sketch_backbone"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache / "hub")
        torch.hub.set_dir(str(cache / "torch"))
        from timm import create_model

        self.backbone = create_model(BACKBONE, pretrained=False, num_classes=0)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(COORDINATE_WIDTH, COORDINATE_WIDTH),
            nn.LayerNorm(COORDINATE_WIDTH), nn.ReLU(),
            nn.Linear(COORDINATE_WIDTH, COORDINATE_WIDTH),
            nn.LayerNorm(COORDINATE_WIDTH), nn.ReLU(),
            nn.Linear(COORDINATE_WIDTH, 1),
        )
        self.classifier = nn.Linear(self.backbone.num_features, 1)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.window: deque[torch.Tensor] = deque(maxlen=CLIP_LENGTH)
        # Preserve pretrained features on the small local dataset; gradients still reach DRE.
        self.backbone.requires_grad_(False)

    def initialize_pretrained(self) -> None:
        """Load ImageNet weights for training, with downloads cached in the project."""
        cache = MODEL_DIR / "sketch_backbone"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        from timm import create_model

        logging.getLogger(__name__).info("Loading pretrained SKETCH backbone %s into %s", BACKBONE, cache)
        pretrained = create_model(BACKBONE, pretrained=True, num_classes=0)
        self.backbone.load_state_dict(pretrained.state_dict())

    def train(self, mode: bool = True) -> "Sketch2D":
        """Keep the frozen backbone deterministic while training DRE and classifier."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Render trajectories and add learned unscaled-coordinate embeddings."""
        images = np.stack([coordinate_image(clip) for clip in sequence.detach().cpu().numpy()])
        image_tensor = torch.from_numpy(images).to(device=sequence.device, dtype=sequence.dtype)
        image_tensor = (image_tensor - self.image_mean) / self.image_std
        coordinates = sequence[:, :2] * (sequence[:, 2:3] >= KEYPOINT_THRESHOLD)
        offsets = self.coordinate_embedding(coordinates.reshape(-1, COORDINATE_WIDTH))
        offsets = offsets.reshape(sequence.shape[0], 2)
        additive_map = offsets.repeat_interleave(PANEL_HEIGHT, dim=1)[:, None, :, None]
        return self.classifier(self.backbone(image_tensor + additive_map))

    def training_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return clip-level logits using the same representation as live inference."""
        return self(sequence)

    def forward_step(self, frame: torch.Tensor) -> torch.Tensor | None:
        """Recognize the configured window after collecting enough valid hand frames."""
        self.window.append(frame.detach())
        if len(self.window) < CLIP_LENGTH:
            return None
        return self(torch.stack(tuple(self.window), dim=2)).squeeze(-1)

    def reset_stream(self) -> None:
        """Discard trajectories after hand loss or a sampling gap."""
        self.window.clear()

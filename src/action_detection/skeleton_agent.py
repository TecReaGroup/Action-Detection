"""SkeletonAgent MS-TCN adapted to the local left-hand graph classifier.

Temporal module adapted from firework8/SkeletonAgent at 2440fda2c502ef07253e679e81b59468d912eda8,
skeletonagent/models/gcns/utils/tcn_utils.py. This is not the full agent framework.

MIT License
Copyright (c) 2025 Hongda Liu

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

import torch
from torch import nn

from .model import hand_adjacency
from .setting import CLIP_LENGTH, FEATURE_JOINT_COUNT

DILATIONS = (1, 2, 3, 4)
WINDOW_CHANNELS = (3, 32, 64, 64)


class MultiScaleTemporalConv(nn.Module):
    """Fuse six temporal branches and a learnable global-joint contribution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_width = channels // 6
        branches = []
        for index, dilation in enumerate(DILATIONS):
            width = channels - branch_width * 5 if index == 0 else branch_width
            branches.append(nn.Sequential(
                nn.Conv2d(channels, width, 1), nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width, (3, 1), padding=(dilation, 0), dilation=(dilation, 1)),
            ))
        branches.append(nn.Sequential(
            nn.Conv2d(channels, branch_width, 1), nn.BatchNorm2d(branch_width), nn.ReLU(),
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
        ))
        branches.append(nn.Conv2d(channels, branch_width, 1))
        self.branch = nn.ModuleList(branches)
        self.global_weight = nn.Parameter(torch.zeros(FEATURE_JOINT_COUNT))
        self.transform = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply symmetric temporal padding within the observed window."""
        augmented = torch.cat((sequence, sequence.mean(-1, keepdim=True)), dim=-1)
        combined = torch.cat([branch(augmented) for branch in self.branch], dim=1)
        local = combined[..., :FEATURE_JOINT_COUNT]
        global_feature = combined[..., FEATURE_JOINT_COUNT:]
        return self.transform(local + global_feature * self.global_weight)


class MultiScaleHandBlock(nn.Module):
    """Combine the existing hand graph partitioning with SkeletonAgent MS-TCN."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.register_buffer("adjacency", hand_adjacency())
        self.edge_weight = nn.Parameter(torch.ones_like(self.adjacency))
        self.spatial = nn.Conv2d(input_channels, output_channels * 3, 1)
        self.spatial_norm = nn.BatchNorm2d(output_channels)
        self.temporal = MultiScaleTemporalConv(output_channels)
        self.residual = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Aggregate hand neighbors before multiscale temporal modeling."""
        projected = self.spatial(sequence)
        batch, _, length, joints = projected.shape
        projected = projected.reshape(batch, 3, -1, length, joints)
        spatial = torch.einsum(
            "bkctv,kvw->bctw", projected, self.adjacency * self.edge_weight,
        )
        spatial = self.spatial_norm(spatial).relu()
        return (self.temporal(spatial) + self.residual(sequence)).relu()


class SkeletonAgentMSTCN(nn.Module):
    """Classify observed left-hand windows using SkeletonAgent temporal branches."""

    warmup_frames = CLIP_LENGTH

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(*(
            MultiScaleHandBlock(before, after)
            for before, after in zip(WINDOW_CHANNELS[:-1], WINDOW_CHANNELS[1:])
        ))
        self.classifier = nn.Linear(WINDOW_CHANNELS[-1], 1)
        self.window: deque[torch.Tensor] = deque(maxlen=CLIP_LENGTH)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return one binary logit per complete clip."""
        return self.classifier(self.block(sequence).mean(dim=(2, 3)))

    def training_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        """Supervise the same full-window prediction used during preview."""
        return self(sequence)

    def forward_step(self, frame: torch.Tensor) -> torch.Tensor | None:
        """Classify the trailing window once enough observed frames are available."""
        self.window.append(frame.detach())
        if len(self.window) < CLIP_LENGTH:
            return None
        return self(torch.stack(tuple(self.window), dim=2)).squeeze(-1)

    def reset_stream(self) -> None:
        """Discard the window after a missing left hand or a sampling gap."""
        self.window.clear()

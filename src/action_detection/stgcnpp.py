"""Hand adaptation of PySKL ST-GCN++ with a binary window classifier.

Reference: kennymckormick/pyskl, pyskl/models/gcns/stgcn.py and
utils/{gcn,tcn}.py; configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py.
"""

from collections import deque

import torch
from torch import nn

from .model import hand_adjacency
from .setting import CLIP_LENGTH, FEATURE_JOINT_COUNT

STAGE_CHANNELS = (3, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
DOWN_STAGES = (5, 8)
TEMPORAL_DILATIONS = (1, 2, 3, 4)


class LearnableHandGraph(nn.Module):
    """Learn all spatial edges from the normalized hand graph initialization."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.adjacency = nn.Parameter(hand_adjacency())
        self.projection = nn.Conv2d(input_channels, output_channels * 3, 1)
        self.norm = nn.BatchNorm2d(output_channels)
        self.residual = nn.Identity() if input_channels == output_channels else nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1), nn.BatchNorm2d(output_channels),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Aggregate spatial partitions and add the GCN's internal residual."""
        batch, _, length, joints = sequence.shape
        projected = self.projection(sequence).reshape(batch, 3, -1, length, joints)
        spatial = torch.einsum("bkctv,kvw->bctw", projected, self.adjacency)
        return (self.norm(spatial) + self.residual(sequence)).relu()


class STGCNPlusTemporal(nn.Module):
    """Fuse PySKL's four dilated, pooling and pointwise temporal branches."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        branch_width = channels // 6
        branches = []
        for index, dilation in enumerate(TEMPORAL_DILATIONS):
            width = channels - branch_width * 5 if index == 0 else branch_width
            branches.append(nn.Sequential(
                nn.Conv2d(channels, width, 1), nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width, (3, 1), stride=(stride, 1),
                          padding=(dilation, 0), dilation=(dilation, 1)),
            ))
        branches.append(nn.Sequential(
            nn.Conv2d(channels, branch_width, 1), nn.BatchNorm2d(branch_width), nn.ReLU(),
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
        ))
        branches.append(nn.Conv2d(channels, branch_width, 1, stride=(stride, 1)))
        self.branch = nn.ModuleList(branches)
        self.transform = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Use symmetric padding within the observed window."""
        return self.transform(torch.cat([branch(sequence) for branch in self.branch], dim=1))


class STGCNPlusBlock(nn.Module):
    """Combine adaptive spatial aggregation and multiscale temporal filtering."""

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        self.spatial = LearnableHandGraph(input_channels, output_channels)
        self.temporal = STGCNPlusTemporal(output_channels, stride)
        self.residual = (
            nn.Identity() if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Preserve the block residual through channel and temporal changes."""
        return (self.temporal(self.spatial(sequence)) + self.residual(sequence)).relu()


class STGCNPlusPlus(nn.Module):
    """Run the ten-stage ST-GCN++ backbone on single- or dual-hand windows."""

    warmup_frames = CLIP_LENGTH

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(3 * FEATURE_JOINT_COUNT)
        # The first upstream stage omits the outer residual, retaining the GCN residual.
        self.stem = nn.Sequential(
            LearnableHandGraph(STAGE_CHANNELS[0], STAGE_CHANNELS[1]),
            STGCNPlusTemporal(STAGE_CHANNELS[1], 1), nn.ReLU(),
        )
        self.block = nn.Sequential(*(
            STGCNPlusBlock(before, after, 2 if stage in DOWN_STAGES else 1)
            for stage, (before, after) in enumerate(
                zip(STAGE_CHANNELS[1:-1], STAGE_CHANNELS[2:]), start=2,
            )
        ))
        self.classifier = nn.Linear(STAGE_CHANNELS[-1], 1)
        self.window: deque[torch.Tensor] = deque(maxlen=CLIP_LENGTH)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return (batch, 1) logits from (batch, 3, time, joint) features."""
        batch, channels, length, joints = sequence.shape
        normalized = sequence.permute(0, 3, 1, 2).reshape(batch, joints * channels, length)
        normalized = self.input_norm(normalized)
        normalized = normalized.reshape(batch, joints, channels, length).permute(0, 2, 3, 1)
        encoded = self.block(self.stem(normalized))
        return self.classifier(encoded.mean(dim=(2, 3)))

    def training_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        """Supervise the same complete-window prediction used during preview."""
        return self(sequence)

    def forward_step(self, frame: torch.Tensor) -> torch.Tensor | None:
        """Classify the trailing window once enough observed frames are available."""
        self.window.append(frame.detach())
        if len(self.window) < CLIP_LENGTH:
            return None
        return self(torch.stack(tuple(self.window), dim=2)).squeeze(-1)

    def reset_stream(self) -> None:
        """Discard buffered frames after hand loss or a sampling gap."""
        self.window.clear()

"""Compact Hyper-GCN adaptation for the local hand skeleton stream."""

from collections import deque

import torch
from torch import nn

from .setting import CLIP_LENGTH, FEATURE_JOINT_COUNT

HIDDEN_CHANNELS = (32, 48, 96, 96)
VIRTUAL_JOINT_COUNT = 3
HEAD_COUNT = 4
TEMPORAL_DILATIONS = (1, 2, 3, 4)


class AdaptiveHyperGraph(nn.Module):
    """Build sample-dependent hypergraph affinities with virtual joints."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, channels // 8)
        self.value = nn.Conv1d(channels, HEAD_COUNT * hidden, 1, groups=HEAD_COUNT)
        self.incidence = nn.Sequential(
            nn.Conv1d(channels, HEAD_COUNT * hidden, 1, groups=HEAD_COUNT),
            nn.LeakyReLU(),
            nn.Conv1d(HEAD_COUNT * hidden, HEAD_COUNT, 1),
            nn.Tanh(),
        )
        self.virtual = nn.Parameter(torch.randn(VIRTUAL_JOINT_COUNT, channels))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, _, _, joints = sequence.shape
        virtual = self.virtual.T.view(1, -1, 1, VIRTUAL_JOINT_COUNT).expand(
            batch, -1, sequence.shape[2], -1
        )
        augmented = torch.cat((sequence, virtual), dim=-1)
        pooled = augmented.mean(2)
        values = self.value(pooled).view(batch, HEAD_COUNT, -1, joints + VIRTUAL_JOINT_COUNT)
        values = values.permute(0, 1, 3, 2)
        distances = torch.cdist(values, values)
        neighbors = distances.topk(min(9, joints + VIRTUAL_JOINT_COUNT), largest=False).indices
        incidence = torch.zeros_like(distances).scatter(3, neighbors, 1.0)
        weights = self.incidence(pooled).softmax(-1)
        normalized = incidence / incidence.sum(-1, keepdim=True).clamp_min(1)
        affinity = normalized @ torch.diag_embed(weights) @ normalized.transpose(-1, -2)
        return augmented, affinity.mean(1) * self.scale.relu()


class HyperGraphBlock(nn.Module):
    """Apply adaptive hypergraph convolution followed by temporal branches."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.hyper = AdaptiveHyperGraph(input_channels)
        self.project = nn.Conv2d(input_channels, output_channels, 1)
        self.temporal = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(output_channels, output_channels // 6, 1),
                nn.BatchNorm2d(output_channels // 6),
                nn.ReLU(),
                nn.Conv2d(output_channels // 6, output_channels // 6, (3, 1),
                          padding=(dilation, 0), dilation=(dilation, 1)),
            )
            for dilation in TEMPORAL_DILATIONS
        )
        branch = output_channels // 6
        self.pool = nn.Sequential(nn.Conv2d(output_channels, branch, 1), nn.ReLU(),
                                  nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)))
        self.point = nn.Conv2d(output_channels, branch, 1)
        self.norm = nn.BatchNorm2d(output_channels)
        self.residual = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        augmented, affinity = self.hyper(sequence)
        projected = self.project(augmented)
        projected = torch.einsum("nuv,nctv->nctu", affinity, projected)
        projected = projected[..., :FEATURE_JOINT_COUNT]
        branches = [branch(projected) for branch in self.temporal]
        branches.extend((self.pool(projected), self.point(projected)))
        return (self.norm(torch.cat(branches, dim=1)) + self.residual(sequence)).relu()


class HyperGCN(nn.Module):
    """Classify a complete hand action window with adaptive hypergraph blocks."""

    warmup_frames = CLIP_LENGTH

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(*(
            HyperGraphBlock(before, after)
            for before, after in zip(HIDDEN_CHANNELS[:-1], HIDDEN_CHANNELS[1:])
        ))
        self.embedding = nn.Conv2d(3, HIDDEN_CHANNELS[0], 1)
        self.classifier = nn.Linear(HIDDEN_CHANNELS[-1], 1)
        self.window: deque[torch.Tensor] = deque(maxlen=CLIP_LENGTH)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.block(self.embedding(sequence))
        return self.classifier(encoded.mean(dim=(2, 3)))

    def training_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return one binary logit for each complete training window."""
        return self(sequence)

    def forward_step(self, frame: torch.Tensor) -> torch.Tensor | None:
        """Classify the trailing window after enough frames are available."""
        self.window.append(frame.detach())
        if len(self.window) < CLIP_LENGTH:
            return None
        return self(torch.stack(tuple(self.window), dim=2)).squeeze(-1)

    def reset_stream(self) -> None:
        """Discard buffered frames after a stream gap."""
        self.window.clear()

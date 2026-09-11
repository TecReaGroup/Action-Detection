"""Hand-adapted causal Continual ST-GCN with per-layer temporal state."""

import torch
from torch import nn
from torch.nn import functional as F

from .setting import HAND_EDGES, HAND_JOINT_COUNT

TEMPORAL_KERNEL = 5
CHANNELS = (3, 32, 64, 64)
RECEPTIVE_FIELD = 1 + (TEMPORAL_KERNEL - 1) * (len(CHANNELS) - 1)


def hand_adjacency() -> torch.Tensor:
    """Build identity, inward and outward normalized hand graph partitions."""
    inward = torch.zeros(HAND_JOINT_COUNT, HAND_JOINT_COUNT)
    for parent, child in HAND_EDGES:
        inward[parent, child] = 1
    partitions = [torch.eye(HAND_JOINT_COUNT), inward, inward.T]
    return torch.stack([part / part.sum(0).clamp_min(1) for part in partitions])


class GraphTemporalBlock(nn.Module):
    """Spatial graph convolution followed by a causal temporal convolution."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.register_buffer("adjacency", hand_adjacency())
        self.edge_weight = nn.Parameter(torch.ones(3, HAND_JOINT_COUNT, HAND_JOINT_COUNT))
        self.spatial = nn.Conv2d(input_channels, output_channels * 3, 1)
        self.spatial_norm = nn.BatchNorm2d(output_channels)
        self.temporal = nn.Conv2d(output_channels, output_channels, (TEMPORAL_KERNEL, 1))
        self.temporal_norm = nn.BatchNorm2d(output_channels)
        self.residual = nn.Conv2d(input_channels, output_channels, 1)
        self.history: torch.Tensor | None = None

    def spatial_features(self, sequence: torch.Tensor) -> torch.Tensor:
        """Aggregate the three graph partitions at each timestamp."""
        projected = self.spatial(sequence)
        batch, _, length, joints = projected.shape
        projected = projected.reshape(batch, 3, -1, length, joints)
        aggregated = torch.einsum(
            "bkctv,kvw->bctw", projected, self.adjacency * self.edge_weight,
        )
        return F.relu(self.spatial_norm(aggregated))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Train all timestamps in parallel with left-only temporal padding."""
        spatial = self.spatial_features(sequence)
        temporal = self.temporal(F.pad(spatial, (0, 0, TEMPORAL_KERNEL - 1, 0)))
        return F.relu(self.temporal_norm(temporal) + self.residual(sequence))

    def forward_step(self, frame: torch.Tensor) -> torch.Tensor:
        """Consume one frame, retaining only this layer's required history."""
        spatial = self.spatial_features(frame)
        if self.history is None:
            self.history = spatial.new_zeros(
                spatial.shape[0], spatial.shape[1], TEMPORAL_KERNEL - 1, HAND_JOINT_COUNT,
            )
        temporal_input = torch.cat((self.history, spatial), dim=2)
        self.history = temporal_input[:, :, 1:].detach()
        temporal = self.temporal(temporal_input)
        return F.relu(self.temporal_norm(temporal) + self.residual(frame))


class ContinualSTGCN(nn.Module):
    """Binary thumb-action classifier with constant-memory online inference."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            GraphTemporalBlock(before, after)
            for before, after in zip(CHANNELS[:-1], CHANNELS[1:])
        )
        self.classifier = nn.Linear(CHANNELS[-1], 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return framewise logits for a (batch, channel, time, joint) clip."""
        for block in self.block:
            sequence = block(sequence)
        return self.classifier(sequence.mean(-1).transpose(1, 2)).squeeze(-1)

    def forward_step(self, frame: torch.Tensor) -> torch.Tensor:
        """Return the latest logit without recomputing previous frames."""
        frame = frame.unsqueeze(2)
        for block in self.block:
            frame = block.forward_step(frame)
        return self.classifier(frame.mean(-1).squeeze(-1)).squeeze(-1)

    def reset_stream(self) -> None:
        """Clear temporal state after a gap or a missing hand."""
        for block in self.block:
            block.history = None

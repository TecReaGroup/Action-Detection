"""Select and validate temporal models at the configuration boundary."""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import ContinualSTGCN
from .setting import CLIP_LENGTH, CONFIG_PATH, MODEL_DIR
from .skeleton_agent import SkeletonAgentMSTCN
from .sketch import Sketch2D

MODEL_TYPES = {
    "continual_stgcn": ContinualSTGCN,
    "skeleton_agent_mstcn": SkeletonAgentMSTCN,
    "sketch_2d": Sketch2D,
}


@dataclass(frozen=True)
class TemporalModel:
    """Bind a validated model constructor to its distinct checkpoint path."""

    name: str
    network_type: type[ContinualSTGCN] | type[SkeletonAgentMSTCN] | type[Sketch2D]
    checkpoint_path: Path
    device: torch.device

    def create_training_network(self) -> ContinualSTGCN | SkeletonAgentMSTCN | Sketch2D:
        """Initialize training weights required by the selected architecture."""
        network = self.network_type()
        if isinstance(network, Sketch2D):
            network.initialize_pretrained()
        return network.to(self.device)


def load_temporal_model() -> TemporalModel:
    """Resolve the configured model once for a training or preview session."""
    with CONFIG_PATH.open("rb") as stream:
        configuration = tomllib.load(stream)
    section = configuration.get("temporal")
    if not isinstance(section, dict):
        raise ValueError(f"Missing [temporal] section in {CONFIG_PATH}")
    name = section.get("model")
    if not isinstance(name, str) or name not in MODEL_TYPES:
        raise ValueError(f"temporal.model must be one of: {', '.join(MODEL_TYPES)}")
    checkpoint_path = MODEL_DIR / f"thumb_sway_left_{name}.pt"
    device_name = configuration.get("pose", {}).get("device")
    if device_name not in ("cpu", "cuda"):
        raise ValueError("pose.device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Temporal CUDA unavailable; install CUDA-enabled PyTorch with uv sync")
    device = torch.device(device_name)
    logging.getLogger(__name__).info("Temporal device=%s", device)
    logging.getLogger(__name__).info(
        "Temporal model=%s checkpoint=%s window_frames=%d",
        name, checkpoint_path, CLIP_LENGTH,
    )
    return TemporalModel(name, MODEL_TYPES[name], checkpoint_path, device)

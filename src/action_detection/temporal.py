"""Select and validate temporal models at the configuration boundary."""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .model import ContinualSTGCN
from .setting import CONFIG_PATH, MODEL_DIR
from .skeleton_agent import SkeletonAgentMSTCN

MODEL_TYPES = {
    "continual_stgcn": ContinualSTGCN,
    "skeleton_agent_mstcn": SkeletonAgentMSTCN,
}


@dataclass(frozen=True)
class TemporalModel:
    """Bind a validated model constructor to its distinct checkpoint path."""

    name: str
    network_type: type[ContinualSTGCN] | type[SkeletonAgentMSTCN]
    checkpoint_path: Path


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
    logging.getLogger(__name__).info("Temporal model=%s checkpoint=%s", name, checkpoint_path)
    return TemporalModel(name, MODEL_TYPES[name], checkpoint_path)

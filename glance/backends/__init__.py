"""Three backends behind one interface: dual encoder, open VLM with logit readout, frontier baseline."""

from __future__ import annotations

from ..config import Config
from .base import Backend, BackendUsage, LabelScores, PickItem, Statement, StatementScores, model_string

__all__ = [
    "Backend",
    "BackendUsage",
    "LabelScores",
    "PickItem",
    "Statement",
    "StatementScores",
    "model_string",
    "load_backend",
]


def load_backend(name: str, cfg: Config, **kwargs) -> Backend:
    """Build a backend by the name the caller uses in `model`. Heavy imports stay inside each backend module."""
    if name == "siglip":
        from .siglip import SiglipBackend

        return SiglipBackend(cfg, **kwargs)
    if name == "vlm":
        from .vlm_hf import VlmBackend

        return VlmBackend(cfg, **kwargs)
    if name == "frontier":
        from .frontier import FrontierBackend

        return FrontierBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend: {name}")

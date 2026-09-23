"""Three backends behind one interface: dual encoder, open VLM with logit readout, frontier baseline."""

from __future__ import annotations

from ..config import Config
from ..schema import BackendError
from .base import Backend, BackendUsage, LabelScores, PickItem, PickResult, Statement, StatementScores, model_string

__all__ = [
    "Backend",
    "BackendUsage",
    "LabelScores",
    "PickItem",
    "PickResult",
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
    if name == "vlm" and cfg.vlm.backend == "mlx":
        if cfg.models.generic is not None:
            raise BackendError(
                "--backend mlx currently supports only the pinned Qwen3-VL-2B 8-bit checkpoint; remove --model-id"
            )
        from .vlm_mlx import MlxVlmBackend

        return MlxVlmBackend(cfg, **kwargs)
    if name == "vlm" and cfg.models.generic is not None:  # the caller chose a model by id: the any-model backend
        from .generic_hf import GenericVlmBackend

        g = cfg.models.generic
        return GenericVlmBackend(cfg, model_id=g.id, revision=g.revision, longest_edge=g.image_longest_edge, dtype=g.dtype, **kwargs)
    if name == "vlm":
        from .vlm_hf import VlmBackend

        return VlmBackend(cfg, **kwargs)
    if name == "frontier":
        from .frontier import FrontierBackend

        return FrontierBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend: {name}")

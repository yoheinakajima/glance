"""Backend protocol. Backends return raw logits only; they never see question types.

Probabilities, confidence and calibration live downstream in `scorer` and `calibration`, so every backend is
scored and calibrated the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ..images import LoadedImage


@dataclass(frozen=True)
class Statement:
    """One thing that may be true of the image.

    `text` is the rendered Yes/No statement block a VLM reads. `candidate` is the bare candidate text a dual
    encoder embeds. The scorer fills both; each backend uses the one it needs.
    """

    text: str
    candidate: str


@dataclass
class BackendUsage:
    image_tokens: int = 0
    text_tokens: int = 0
    forward_passes: int = 0

    def add(self, other: "BackendUsage") -> None:
        self.image_tokens = max(self.image_tokens, other.image_tokens)  # same images, encoded once per request
        self.text_tokens += other.text_tokens
        self.forward_passes += other.forward_passes


@dataclass
class StatementScores:
    """Raw logits for a list of statements. `z` is the logit that each statement is true of the image."""

    z: np.ndarray  # [n]
    z_yes: np.ndarray  # [n]
    z_no: np.ndarray | None  # [n]; None for a dual encoder
    off_mass: np.ndarray | None  # [n]; 1 - P(yes variants) - P(no variants); None for a dual encoder
    prompt_hashes: list[str]
    usage: BackendUsage = field(default_factory=BackendUsage)
    timing_ms: dict[str, float] = field(default_factory=dict)  # "prefix", "score"
    cache_hit: bool | None = None  # prefix cache used; None when the backend has none


@dataclass
class LabelScores:
    """Raw logits over label tokens (A, B, C, ...) for each prompt. Used by the `letter` choice method."""

    logits: np.ndarray  # [n_prompts, n_labels]
    off_mass: np.ndarray  # [n_prompts]; 1 - P(label tokens)
    prompt_hashes: list[str]
    usage: BackendUsage = field(default_factory=BackendUsage)
    timing_ms: dict[str, float] = field(default_factory=dict)
    cache_hit: bool | None = None


@dataclass(frozen=True)
class PickItem:
    """Frontier baseline only: a prompt and the enumerated answers the model may return."""

    prompt: str
    allowed: list[str]


@dataclass
class PickResult:
    """Frontier baseline only: one enumerated answer per PickItem, in order."""

    picks: list[str]
    usage: BackendUsage = field(default_factory=BackendUsage)
    timing_ms: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str  # "siglip" | "vlm" | "frontier", as the caller names it in `model`
    kind: Literal["dual_encoder", "vlm", "frontier"]
    model_id: str
    revision: str | None
    device: str
    dtype: str
    image_token_budget: int | None

    def score_statements(
        self, images: list[LoadedImage], context: dict[str, Any] | None, statements: list[Statement]
    ) -> StatementScores: ...


def model_string(backend: Backend) -> str:
    """`vlm:Qwen/Qwen3-VL-2B-Instruct@<revision-sha>` as it appears in responses and logs."""
    rev = f"@{backend.revision}" if backend.revision else ""
    return f"{backend.name}:{backend.model_id}{rev}"

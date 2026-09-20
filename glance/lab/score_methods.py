"""Score lab: different ways to ask a VLM for a rating on an ordered scale, all without training.

v0 asks about every level separately ("is this candidate the correct answer?") and never shows the scale. That
ranks images well but puts the level boundaries in the wrong place. The lab compares it with:

- `cumulative`: show the scale, ask K-1 yes/no questions "is it step k or higher?". Boundaries become explicit.
- `digits`: show the numbered scale and read the next-token logits over the digits in one pass.
- `anchors_*`: the same two, with three reference images of known level in the request (the API already takes up
  to 4 images), so the scale is anchored visually and not only in words.

Everything here is prompt + readout + a handful of calibration numbers fit on a calibration split. Lab prompts are
versioned separately from the harness prompts (`LAB_PROMPT_VERSION`); the v0 `p1` templates are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax

from .. import prompts
from ..backends.base import Statement

LAB_PROMPT_VERSION = "s1"
METHODS = ("independent", "cumulative", "digits", "anchors_cumulative", "anchors_digits")

SCALE_BLOCK = "The answer scale, from lowest to highest:\n{steps}\n"
CUMULATIVE_TEMPLATE = (
    "Question: {instructions}\n" + SCALE_BLOCK + "Is the correct answer step {step} or higher on this scale? Answer Yes or No."
)
DIGITS_TEMPLATE = "Question: {instructions}\n" + SCALE_BLOCK + "Answer with the number of the correct step."
ANCHOR_SENTENCE = " For reference, {refs}."
ANCHOR_REF = "`{id}` is an example of step {step} (\"{text}\")"


@dataclass(frozen=True)
class Anchor:
    image_id: str  # id used in the request, e.g. "ref1"
    level: int
    path: str


def _steps(levels: list[str], start: int) -> str:
    return "\n".join(f"{i + start}. {text}" for i, text in enumerate(levels))


def with_anchors(instructions: str, levels: list[str], anchors: list[Anchor], start: int) -> str:
    refs = "; ".join(ANCHOR_REF.format(id=a.image_id, step=a.level + start, text=levels[a.level]) for a in anchors)
    return instructions + ANCHOR_SENTENCE.format(refs=refs)


def independent_statements(instructions: str, levels: list[str]) -> list[Statement]:
    """The v0 method, unchanged: one yes/no statement per level, scale never shown."""
    return [Statement(text=prompts.render_candidate(instructions, text), candidate=text) for text in levels]


def cumulative_statements(instructions: str, levels: list[str]) -> list[Statement]:
    """K-1 statements: "is the correct answer step k or higher?" for k = 2..K (steps are numbered from 1)."""
    steps = _steps(levels, start=1)
    return [
        Statement(text=CUMULATIVE_TEMPLATE.format(instructions=instructions, steps=steps, step=k + 1), candidate=f">={k}")
        for k in range(1, len(levels))
    ]


def digits_block(instructions: str, levels: list[str]) -> tuple[str, list[str]]:
    """One prompt; the readout is over the digit tokens 0..K-1."""
    return DIGITS_TEMPLATE.format(instructions=instructions, steps=_steps(levels, start=0)), [str(i) for i in range(len(levels))]


# --- logits -> level distributions ------------------------------------------------------------------


def dist_from_level_logits(z: np.ndarray, bias: np.ndarray | None = None, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if bias is not None:
        z = z + bias
    return softmax(z / temperature, axis=-1)


def dist_from_cumulative(c: np.ndarray, a: np.ndarray | float = 1.0, b: np.ndarray | float = 0.0) -> np.ndarray:
    """c[..., k-1] is the logit of "level >= k". Returns P(level = 0..K-1), with the thresholds made monotone."""
    q = expit(np.asarray(a) * np.asarray(c, dtype=np.float64) + np.asarray(b))
    q = np.minimum.accumulate(q, axis=-1)  # P(>=1) >= P(>=2) >= ...
    upper = np.concatenate([np.ones_like(q[..., :1]), q], axis=-1)
    lower = np.concatenate([q, np.zeros_like(q[..., :1])], axis=-1)
    p = np.clip(upper - lower, 1e-9, None)
    return p / p.sum(axis=-1, keepdims=True)


# --- calibration fits (calibration split only) --------------------------------------------------------


def fit_temperature(z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    def nll(theta):
        return -np.mean(log_softmax(z / np.exp(theta[0]), axis=1)[np.arange(len(y)), y])

    res = minimize(nll, np.zeros(1), method="L-BFGS-B", bounds=[(-5, 7)])
    return {"kind": "T", "T": float(np.exp(res.x[0])), "bias": None}


def fit_vector_scaling(z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """One bias per level (the first is pinned to 0) plus a temperature: K numbers in total."""
    k = z.shape[1]

    def nll(theta):
        bias = np.concatenate([[0.0], theta[: k - 1]])
        return -np.mean(log_softmax((z + bias) / np.exp(theta[-1]), axis=1)[np.arange(len(y)), y])

    res = minimize(nll, np.zeros(k), method="L-BFGS-B")
    return {"kind": "bias+T", "T": float(np.exp(res.x[-1])), "bias": [0.0] + [float(v) for v in res.x[: k - 1]]}


def fit_threshold_platt(c: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Platt scaling per threshold: P(level >= k) = sigmoid(a_k * c_k + b_k). 2 * (K - 1) numbers."""
    a, b = [], []
    for k in range(c.shape[1]):
        target = (y >= k + 1).astype(np.float64)

        def nll(theta, col=c[:, k], target=target):
            logits = theta[0] * col + theta[1]
            return float(np.mean(np.logaddexp(0.0, logits) - target * logits))

        res = minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B")
        a.append(float(res.x[0]))
        b.append(float(res.x[1]))
    return {"kind": "platt/threshold", "a": a, "b": b}


def apply_fit(method: str, logits: np.ndarray, fit: dict[str, Any] | None) -> np.ndarray:
    if "cumulative" in method:
        if fit is None:
            return dist_from_cumulative(logits)
        return dist_from_cumulative(logits, np.array(fit["a"]), np.array(fit["b"]))
    if fit is None:
        return dist_from_level_logits(logits)
    return dist_from_level_logits(logits, None if fit["bias"] is None else np.array(fit["bias"]), fit["T"])


def candidate_fits(method: str, logits: np.ndarray, y: np.ndarray) -> list[dict[str, Any] | None]:
    """Raw first, then every calibration this readout supports."""
    if "cumulative" in method:
        return [None, fit_threshold_platt(logits, y)]
    return [None, fit_temperature(logits, y), fit_vector_scaling(logits, y)]


def fit_name(fit: dict[str, Any] | None) -> str:
    return "raw" if fit is None else fit["kind"]

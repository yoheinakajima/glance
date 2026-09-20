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
# Round 2: the same three readouts with a second request image, `zoom`, a pixel-magnified crop from the centre of
# `img0`, so that fine artifacts (grain, compression blocks, lost texture) are large enough for the model to see.
ZOOM_METHODS = ("zoom_independent", "zoom_cumulative", "zoom_digits")
ZOOM_FACTOR = 3
ZOOM_SENTENCE = (
    " `zoom` is a {factor}x pixel-magnified crop from the centre of `img0`: use it to judge fine detail, and "
    "`img0` for the overall picture."
)

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


def with_zoom(instructions: str, factor: int = ZOOM_FACTOR) -> str:
    return instructions + ZOOM_SENTENCE.format(factor=factor)


def zoom_crop(image, factor: int = ZOOM_FACTOR):
    """Centre crop of 1/factor of each side, enlarged back to the full size with nearest-neighbour sampling so that
    pixel-level structure (noise grain, 8x8 JPEG blocks, resampling steps) is preserved and simply made bigger."""
    from PIL import Image

    w, h = image.size
    cw, ch = w // factor, h // factor
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((cw * factor, ch * factor), Image.NEAREST)


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


def fit_matrix_scaling(x: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 0.05) -> dict[str, Any]:
    """Matrix scaling: p = softmax(W x' + b) on standardized logits x'. A full affine map on the readout, fit by
    L2-regularized NLL. Still post-hoc calibration on frozen logits; the VLM is never touched. Works for any readout
    (x can be K level logits, K - 1 threshold logits, or several readouts concatenated)."""
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    xs = (x - mean) / std
    n, f = xs.shape
    onehot = np.eye(n_classes)[y]

    def loss_grad(theta):
        w = theta[: n_classes * f].reshape(n_classes, f)
        b = theta[n_classes * f:]
        logp = log_softmax(xs @ w.T + b, axis=1)
        resid = np.exp(logp) - onehot
        loss = -np.mean(logp[np.arange(n), y]) + l2 * np.sum(w * w)
        return loss, np.concatenate([(resid.T @ xs / n + 2 * l2 * w).ravel(), resid.mean(axis=0)])

    res = minimize(loss_grad, np.zeros(n_classes * f + n_classes), jac=True, method="L-BFGS-B")
    return {"kind": "matrix", "W": res.x[: n_classes * f].reshape(n_classes, f).tolist(), "b": res.x[n_classes * f:].tolist(),
            "mean": mean.tolist(), "std": std.tolist()}


def n_levels(method: str, logits: np.ndarray) -> int:
    return logits.shape[1] + (1 if "cumulative" in method else 0)


def apply_fit(method: str, logits: np.ndarray, fit: dict[str, Any] | None) -> np.ndarray:
    if fit is not None and fit["kind"] == "matrix":
        xs = (np.asarray(logits, dtype=np.float64) - np.array(fit["mean"])) / np.array(fit["std"])
        return softmax(xs @ np.array(fit["W"]).T + np.array(fit["b"]), axis=1)
    if "cumulative" in method:
        if fit is None:
            return dist_from_cumulative(logits)
        return dist_from_cumulative(logits, np.array(fit["a"]), np.array(fit["b"]))
    if fit is None:
        return dist_from_level_logits(logits)
    return dist_from_level_logits(logits, None if fit["bias"] is None else np.array(fit["bias"]), fit["T"])


FIT_KINDS = {"level": ("raw", "T", "bias+T", "matrix"), "cumulative": ("raw", "platt/threshold", "matrix")}


def fit_kind(method: str, kind: str, logits: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any] | None:
    if kind == "raw":
        return None
    if kind == "T":
        return fit_temperature(logits, y)
    if kind == "bias+T":
        return fit_vector_scaling(logits, y)
    if kind == "platt/threshold":
        return fit_threshold_platt(logits, y)
    if kind == "matrix":
        return fit_matrix_scaling(logits, y, k)
    raise ValueError(kind)


def kinds_for(method: str) -> tuple[str, ...]:
    return FIT_KINDS["cumulative" if "cumulative" in method else "level"]


def candidate_fits(method: str, logits: np.ndarray, y: np.ndarray) -> list[dict[str, Any] | None]:
    """Raw first, then every calibration this readout supports."""
    k = n_levels(method, logits)
    return [fit_kind(method, kind, logits, y, k) for kind in kinds_for(method)]


def cv_nll(method: str, kind: str, logits: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 7) -> float:
    """Cross-validated NLL on the fit data: how a calibration is chosen without ever touching evaluation data."""
    k = n_levels(method, logits)
    order = np.random.default_rng(seed).permutation(len(y))
    total = 0.0
    for fold in range(folds):
        held = order[fold::folds]
        train = np.setdiff1d(order, held)
        p = apply_fit(method, logits[held], fit_kind(method, kind, logits[train], y[train], k))
        total += -np.sum(np.log(np.clip(p[np.arange(len(held)), y[held]], 1e-12, None)))
    return float(total / len(y))


def fit_name(fit: dict[str, Any] | None) -> str:
    return "raw" if fit is None else fit["kind"]

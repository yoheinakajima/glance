"""Glance elicitation for `score` questions: digit readouts, a magnified crop, and a per-rubric matrix calibration.

This is the score lab's winner (`lab/NOTES.md` entries 12 and 13, `docs/paper/RESULTS_LAB.md`) promoted into the harness.
Nothing is trained. For one rating question the VLM is read four times, one forward pass each:

    digits          numbered scale shown lowest to highest, logits over the digit tokens
    digitsrev       the same scale listed highest to lowest (position and digit biases enter with the opposite sign)
    zoom_digits     as `digits`, with a second image `zoom`: a 3x pixel-magnified centre crop of the rated image
    zoom_digitsrev  both

Without a calibration the answer is the softmax of the mean of the four logit vectors. With one, the 4K logits go
through a K x 4K affine map (matrix scaling) fit on a few dozen labeled images of THIS rubric: `glance fit`. A
calibration belongs to one rubric (instructions + criteria) and one configuration; the lab showed it does not transfer
to another rating dimension, so there is no pooled fallback.

The prompt text must stay byte-identical to `glance.lab.score_methods` (version `s1`): the shipped calibrations were
fit on logits produced by those prompts. `tests/test_rating.py` checks that.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

RATING_PROMPT_VERSION = "s1"
SCORE_METHODS = ("statements", "digits", "ens4d")
MEMBERS: dict[str, tuple[str, ...]] = {
    "digits": ("digits",),
    "ens4d": ("digits", "zoom_digits", "digitsrev", "zoom_digitsrev"),
}
ZOOM_FACTOR = 3
ZOOM_IMAGE_ID = "zoom"
L2 = 0.05
ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "ratings"

_SCALE_BLOCK = "The answer scale, from lowest to highest:\n{steps}\n"
_DIGITS_TEMPLATE = "Question: {instructions}\n" + _SCALE_BLOCK + "Answer with the number of the correct step."
_ZOOM_SENTENCE = (
    " `zoom` is a {factor}x pixel-magnified crop from the centre of `{image_id}`: use it to judge fine detail, and "
    "`{image_id}` for the overall picture."
)


# --- prompts and the magnified crop ------------------------------------------------------------------


def digits_block(instructions: str, levels: list[str], reverse: bool = False) -> tuple[str, list[str]]:
    """One prompt; the readout is over the digit tokens 0..K-1. With `reverse` the scale is listed from the highest
    level down (digit 0 = highest level) and the caller flips the logits back into level order."""
    shown = list(reversed(levels)) if reverse else levels
    template = _DIGITS_TEMPLATE.replace("from lowest to highest", "from highest to lowest") if reverse else _DIGITS_TEMPLATE
    steps = "\n".join(f"{i}. {text}" for i, text in enumerate(shown))
    return template.format(instructions=instructions, steps=steps), [str(i) for i in range(len(levels))]


def with_zoom(instructions: str, image_id: str = "img0", factor: int = ZOOM_FACTOR) -> str:
    return instructions + _ZOOM_SENTENCE.format(factor=factor, image_id=image_id)


def zoom_crop(image, factor: int = ZOOM_FACTOR):
    """The central 1/factor of each side, enlarged back with nearest-neighbour sampling, so pixel-level structure
    (grain, 8x8 compression blocks, resampling steps) is kept and simply made bigger."""
    from PIL import Image

    w, h = image.size
    cw, ch = w // factor, h // factor
    left = min(max(int(round(w * 0.5 - cw / 2)), 0), w - cw)
    top = min(max(int(round(h * 0.5 - ch / 2)), 0), h - ch)
    return image.crop((left, top, left + cw, top + ch)).resize((cw * factor, ch * factor), Image.NEAREST)


def member_prompt(member: str, instructions: str, levels: list[str], image_id: str) -> tuple[str, list[str], bool]:
    """(prompt block, digit labels, reversed?) for one member readout."""
    reverse = member.endswith("rev")
    text = with_zoom(instructions, image_id) if member.startswith("zoom_") else instructions
    block, labels = digits_block(text, levels, reverse=reverse)
    return block, labels, reverse


def raw_level_logits(member_logits: list[np.ndarray]) -> np.ndarray:
    """Uncalibrated combination: the mean of the members' level logits (chosen on the lab's calibration split over
    averaging probabilities; `lab/NOTES.md` entry 17)."""
    return np.mean(np.asarray(member_logits, dtype=np.float64), axis=0)


# --- calibration ---------------------------------------------------------------------------------------


class RatingKey(BaseModel):
    """What a rating calibration is valid for. Any difference is a different calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    backend: str
    model: str  # model id @ revision
    prompt_version: str
    score_method: str
    image_token_budget: int | None
    instructions: str
    criteria: tuple[str, ...]

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:12]

    def version(self) -> str:
        return f"rate_{self.hash()[:6]}"


class RatingCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: RatingKey
    version: str
    name: str | None = None
    fit_date: str
    n: int
    n_per_level: list[int]
    members: list[str]
    W: list[list[float]]
    b: list[float]
    mean: list[float]
    std: list[float]
    l2: float
    rescale: float
    cv: dict[str, Any] = {}  # cross-validated quality on the fit data: accuracy, within_1, mae, nll, ece, ece_floor
    source: str | None = None


def fit_matrix(x: np.ndarray, y: np.ndarray, n_classes: int, l2: float = L2) -> dict[str, Any]:
    """Matrix scaling: p = softmax(W x' + b) on standardized logits x', L2-regularized NLL, then one scalar refit
    without the penalty to restore sharpness (it multiplies every logit alike, so no prediction changes)."""
    x = np.asarray(x, dtype=np.float64)
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
    w = res.x[: n_classes * f].reshape(n_classes, f)
    b = res.x[n_classes * f:]
    base = xs @ w.T + b
    scale = minimize(lambda t: -np.mean(log_softmax(np.exp(t[0]) * base, axis=1)[np.arange(n), y]), np.zeros(1),
                     method="L-BFGS-B", bounds=[(-3, 3)]).x[0]
    s = float(np.exp(scale))
    return {"W": (s * w).tolist(), "b": (s * b).tolist(), "mean": mean.tolist(), "std": std.tolist(), "l2": l2, "rescale": s}


def apply_matrix(fit: dict[str, Any] | RatingCalibration, features: np.ndarray) -> np.ndarray:
    get = (lambda k: getattr(fit, k)) if isinstance(fit, RatingCalibration) else fit.__getitem__
    xs = (np.asarray(features, dtype=np.float64) - np.array(get("mean"))) / np.array(get("std"))
    return softmax(xs @ np.array(get("W")).T + np.array(get("b")), axis=-1)


def cross_validate(x: np.ndarray, y: np.ndarray, n_classes: int, folds: int = 5, seed: int = 7) -> dict[str, Any]:
    """Quality of the calibration on held-out folds of its own fit data. Honest at the sample size the user has."""
    from .calibration import ece_equal_mass
    from .evals.metrics import ece_noise_floor

    folds = max(2, min(folds, int(np.bincount(y, minlength=n_classes).min())))
    order = np.random.default_rng(seed).permutation(len(y))
    p = np.zeros((len(y), n_classes))
    for fold in range(folds):
        held = order[fold::folds]
        train = np.setdiff1d(order, held)
        p[held] = apply_matrix(fit_matrix(x[train], y[train], n_classes), x[held])
    pred, conf = p.argmax(1), p.max(1)
    expected = (p * np.arange(n_classes)).sum(1)
    return {
        "folds": folds, "accuracy": float(np.mean(pred == y)), "within_1": float(np.mean(np.abs(pred - y) <= 1)),
        "mae": float(np.mean(np.abs(expected - y))),
        "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))),
        "ece": ece_equal_mass(conf, pred == y, 15), "ece_floor": ece_noise_floor(conf, 15),
    }


def build_calibration(key: RatingKey, features: np.ndarray, levels: np.ndarray, name: str | None = None,
                      source: str | None = None, with_cv: bool = True) -> RatingCalibration:
    x, y = np.asarray(features, dtype=np.float64), np.asarray(levels).astype(int)
    k = len(key.criteria)
    counts = np.bincount(y, minlength=k)
    if len(y) != len(x) or y.min() < 0 or y.max() >= k:
        raise ValueError(f"levels must be integers in 0..{k - 1}, one per example")
    if (counts == 0).any():
        raise ValueError(f"every level needs at least one labeled example; counts per level: {counts.tolist()}")
    fit = fit_matrix(x, y, k)
    cv = cross_validate(x, y, k) if with_cv and counts.min() >= 2 else {}
    return RatingCalibration(
        key=key, version=key.version(), name=name, fit_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        n=len(y), n_per_level=counts.tolist(), members=list(MEMBERS[key.score_method]), cv=cv, source=source, **fit,
    )


def save_calibration(directory: str | Path, cal: RatingCalibration) -> Path:
    path = Path(directory) / f"{cal.key.hash()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.model_dump(), indent=2) + "\n")
    return path


def load_calibration(directories: list[str | Path], key: RatingKey) -> RatingCalibration | None:
    """The calibration fit for exactly this rubric and configuration, or None. User fits shadow the shipped ones."""
    for directory in directories:
        path = Path(directory) / f"{key.hash()}.json"
        if path.exists():
            cal = RatingCalibration.model_validate_json(path.read_text())
            if cal.key == key:
                return cal
    return None

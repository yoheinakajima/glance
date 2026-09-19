"""Post-hoc calibration: fit, save, load, apply (HANDOFF section 7).

noul: Platt scaling p = sigmoid(a * z + b). choice and score: temperature p = softmax(z / T).
One parameter set per configuration key. Loading params for a different key is an error, never a warning.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax

from .schema import CalibrationMismatchError
from .scorer import QuestionScore, softmax

EPS = 1e-12
KEY_FIELDS = ("backend", "model", "prompt_version", "choice_method", "image_token_budget")


class CalibrationKey(BaseModel):
    """(backend, model id @ revision, prompt_version, choice_method, image_token_budget)"""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    backend: str
    model: str
    prompt_version: str
    choice_method: str
    image_token_budget: int | None

    def hash(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def version(self) -> str:
        return f"cal_{self.hash()[:6]}"


class TypeFit(BaseModel):
    """Fitted parameters for one question type, with before/after quality on the data they were fit on."""

    model_config = ConfigDict(extra="forbid")

    method: str  # "platt" | "temperature" | "isotonic"
    params: dict[str, Any]  # platt: a, b; temperature: T; isotonic: a, b plus x, y thresholds
    n: int
    suite_ids: list[str]
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float


class CalibrationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: CalibrationKey
    version: str
    fit_date: str
    source_run: str | None = None
    types: dict[str, TypeFit]  # "noul" | "choice" | "score" -> pooled fit over suites
    per_suite: dict[str, TypeFit] = {}  # suite id -> domain-specific fit, reported for comparison only


# --- metrics used while fitting (evals.metrics builds on these) ------------------------------------


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins over top-label confidence."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(conf, kind="stable")
    total = 0.0
    for idx in np.array_split(order, min(n_bins, len(conf))):
        if len(idx):
            total += len(idx) / len(conf) * abs(conf[idx].mean() - correct[idx].mean())
    return float(total)


def binary_nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def binary_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    return ece_equal_mass(np.maximum(p, 1 - p), pred == y, n_bins)


def multiclass_nll(zs: list[np.ndarray], labels: list[int], temperature: float = 1.0) -> float:
    return float(-np.mean([log_softmax(np.asarray(z, dtype=np.float64) / temperature)[y] for z, y in zip(zs, labels)]))


def multiclass_ece(zs: list[np.ndarray], labels: list[int], temperature: float = 1.0, n_bins: int = 15) -> float:
    probs = [softmax(z, temperature) for z in zs]
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(np.argmax(p)) == y for p, y in zip(probs, labels)])
    return ece_equal_mass(conf, correct, n_bins)


# --- fitting ------------------------------------------------------------------------------------


def fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Minimize NLL of sigmoid(a * z + b). Starts from the identity (a = 1, b = 0)."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def loss(theta: np.ndarray) -> float:
        logits = theta[0] * z + theta[1]
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    def grad(theta: np.ndarray) -> np.ndarray:
        residual = expit(theta[0] * z + theta[1]) - y
        return np.array([np.mean(residual * z), np.mean(residual)])

    result = minimize(loss, x0=np.array([1.0, 0.0]), jac=grad, method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def fit_temperature(zs: list[np.ndarray], labels: list[int]) -> float:
    """Minimize NLL of softmax(z / T) over log T. Items may have different numbers of options."""
    result = minimize_scalar(
        lambda log_t: multiclass_nll(zs, labels, float(np.exp(log_t))), bounds=(-5.0, 7.0), method="bounded"
    )
    return float(np.exp(result.x))


def fit_noul(z: np.ndarray, y: np.ndarray, suite_ids: list[str], n_bins: int = 15, isotonic: bool = False) -> TypeFit:
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y).astype(int)
    a, b = fit_platt(z, y)
    params: dict[str, Any] = {"a": a, "b": b}
    method = "platt"
    after = expit(a * z + b)
    if isotonic:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(z, y)
        params.update({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
        method = "isotonic"
        after = np.interp(z, iso.X_thresholds_, iso.y_thresholds_)
    before = expit(z)
    return TypeFit(
        method=method, params=params, n=len(z), suite_ids=sorted(set(suite_ids)),
        nll_before=binary_nll(before, y), nll_after=binary_nll(after, y),
        ece_before=binary_ece(before, y, n_bins), ece_after=binary_ece(after, y, n_bins),
    )


def fit_multiclass(zs: list[np.ndarray], labels: list[int], suite_ids: list[str], n_bins: int = 15) -> TypeFit:
    t = fit_temperature(zs, labels)
    return TypeFit(
        method="temperature", params={"T": t}, n=len(zs), suite_ids=sorted(set(suite_ids)),
        nll_before=multiclass_nll(zs, labels, 1.0), nll_after=multiclass_nll(zs, labels, t),
        ece_before=multiclass_ece(zs, labels, 1.0, n_bins), ece_after=multiclass_ece(zs, labels, t, n_bins),
    )


def fit_rows(rows: list[dict[str, Any]], n_bins: int = 15, isotonic: bool = False) -> tuple[dict[str, TypeFit], dict[str, TypeFit]]:
    """Fit pooled per-type params and per-suite params from calibration-split prediction rows.

    Each row needs: suite, type, z (list of floats), label_index (0/1 for noul, class or level index otherwise).
    """
    types: dict[str, TypeFit] = {}
    per_suite: dict[str, TypeFit] = {}

    def fit(group: list[dict[str, Any]], qtype: str) -> TypeFit:
        suites = [r["suite"] for r in group]
        if qtype == "noul":
            return fit_noul(
                np.array([r["z"][0] for r in group]), np.array([int(r["label_index"]) for r in group]), suites, n_bins, isotonic
            )
        return fit_multiclass([np.asarray(r["z"]) for r in group], [int(r["label_index"]) for r in group], suites, n_bins)

    for qtype in ("noul", "choice", "score"):
        group = [r for r in rows if r["type"] == qtype]
        if group:
            types[qtype] = fit(group, qtype)
    for suite in sorted({r["suite"] for r in rows}):
        group = [r for r in rows if r["suite"] == suite]
        per_suite[suite] = fit(group, group[0]["type"])
    return types, per_suite


def build_params(key: CalibrationKey, rows: list[dict[str, Any]], source_run: str | None, n_bins: int = 15,
                 isotonic: bool = False) -> CalibrationParams:
    types, per_suite = fit_rows(rows, n_bins, isotonic)
    return CalibrationParams(
        key=key, version=key.version(), source_run=source_run, types=types, per_suite=per_suite,
        fit_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )


# --- save, load, apply --------------------------------------------------------------------------


def params_path(calibration_dir: str | Path, key: CalibrationKey) -> Path:
    return Path(calibration_dir) / f"{key.hash()}.json"


def save_params(calibration_dir: str | Path, params: CalibrationParams) -> Path:
    path = params_path(calibration_dir, params.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.model_dump(), indent=2) + "\n")
    return path


def load_params(calibration_dir: str | Path, active: CalibrationKey) -> CalibrationParams:
    """Load the params fitted for exactly this configuration, or raise calibration_mismatch."""
    path = params_path(calibration_dir, active)
    if not path.exists():
        raise CalibrationMismatchError(
            "calibrated: true, but no fitted params match the active configuration",
            {"active_key": active.model_dump(), "expected_file": str(path)},
        )
    params = CalibrationParams.model_validate_json(path.read_text())
    if params.key != active:
        raise CalibrationMismatchError(
            "fitted params were made for a different configuration",
            {"active_key": active.model_dump(), "file_key": params.key.model_dump(), "file": str(path)},
        )
    return params


def apply_fit(fit: TypeFit, z: np.ndarray) -> float | np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if fit.method == "platt":
        return float(expit(fit.params["a"] * z[0] + fit.params["b"]))
    if fit.method == "isotonic":
        return float(np.interp(z[0], fit.params["x"], fit.params["y"]))
    return softmax(z, fit.params["T"])


def apply(qs: QuestionScore, params: CalibrationParams) -> float | np.ndarray:
    """Calibrated probabilities for one scored question."""
    fit = params.types.get(qs.qtype)
    if fit is None or qs.z is None:
        raise CalibrationMismatchError(
            f"fitted params {params.version} have no `{qs.qtype}` fit", {"question": qs.qid, "types": list(params.types)}
        )
    return apply_fit(fit, qs.z)

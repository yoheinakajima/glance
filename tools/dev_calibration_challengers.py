"""Dev experiment registered in lab/NOTES.md entry 19: prior-centred shrinkage (C1) and an ordinal cumulative-link
readout (C2) against the shipped recipe, on the lab CALIBRATION split only (fit on draws from its first half, judged on
its second half). CPU only, saved logits.

uv run python tools/dev_calibration_challengers.py
"""
import collections
import json
import pathlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax

from glance import rating
from glance.calibration import ece_equal_mass
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = rating.MEMBERS["ens4d"]
K, L2 = 4, 0.05


def folds_for(y, k, folds=5):
    return max(2, min(folds, int(np.bincount(y, minlength=k).min())))


def held_out_scalar(fit_fn, logits_fn, x, y, k, seed=7):
    """Sharpness scalar fit on held-out folds, as in rating.fit_matrix(rescale='cv')."""
    f = folds_for(y, k)
    order = np.random.default_rng(seed).permutation(len(y))
    z = np.zeros((len(y), k))
    for fold in range(f):
        held = order[fold::f]
        train = np.setdiff1d(order, held)
        z[held] = logits_fn(fit_fn(x[train], y[train]), x[held])
    return rating._sharpness(z, y)


# --- current recipe ----------------------------------------------------------------------------------------
def current(x, y):
    return rating.fit_matrix(x, y, K, rescale="cv")


def current_p(fit, x):
    return rating.apply_matrix(fit, x)


# --- C1: L2 toward the uncalibrated mean-member-logit readout ---------------------------------------------------
def c1_raw(x, y, l2=L2):
    mean, std = x.mean(0), x.std(0) + 1e-6
    xs = (x - mean) / std
    n, f = xs.shape
    avg = np.hstack([np.eye(K)] * (f // K)) / (f // K)  # mean of the members' level logits
    w0, b0 = avg * std, avg @ mean
    onehot = np.eye(K)[y]

    def loss_grad(theta):
        w, b = theta[: K * f].reshape(K, f), theta[K * f:]
        logp = log_softmax(xs @ w.T + b, axis=1)
        resid = np.exp(logp) - onehot
        d = w - w0
        return (-np.mean(logp[np.arange(n), y]) + l2 * np.sum(d * d),
                np.concatenate([(resid.T @ xs / n + 2 * l2 * d).ravel(), resid.mean(0)]))

    res = minimize(loss_grad, np.concatenate([w0.ravel(), b0]), jac=True, method="L-BFGS-B")
    return {"W": res.x[: K * f].reshape(K, f), "b": res.x[K * f:], "mean": mean, "std": std}


def c1_logits(fit, x):
    return ((x - fit["mean"]) / fit["std"]) @ fit["W"].T + fit["b"]


def c1(x, y):
    fit = c1_raw(x, y)
    fit["s"] = held_out_scalar(c1_raw, c1_logits, x, y, K) if folds_for(y, K) >= 2 else 1.0
    return fit


def c1_p(fit, x):
    return softmax(fit["s"] * c1_logits(fit, x), axis=1)


# --- C2: cumulative link on a learned 1-D projection ------------------------------------------------------------------
def c2_probs(w, theta, xs, s=1.0):
    eta = xs @ w
    cum = expit(s * (theta[None, :] - eta[:, None]))  # P(level <= k), k = 0..K-2
    upper = np.concatenate([cum, np.ones((len(xs), 1))], axis=1)
    lower = np.concatenate([np.zeros((len(xs), 1)), cum], axis=1)
    p = np.clip(upper - lower, 1e-9, None)
    return p / p.sum(1, keepdims=True)


def c2_raw(x, y, l2=L2):
    mean, std = x.mean(0), x.std(0) + 1e-6
    xs = (x - mean) / std
    f = xs.shape[1]

    def unpack(t):
        w = t[:f]
        theta = np.cumsum(np.concatenate([t[f:f + 1], np.exp(t[f + 1:])]))  # ordered thresholds
        return w, theta

    def loss(t):
        w, theta = unpack(t)
        p = c2_probs(w, theta, xs)
        return -np.mean(np.log(p[np.arange(len(y)), y])) + l2 * np.sum(w * w)

    t0 = np.concatenate([np.zeros(f), [-1.0], np.zeros(K - 2)])
    res = minimize(loss, t0, method="L-BFGS-B")
    w, theta = unpack(res.x)
    return {"w": w, "theta": theta, "mean": mean, "std": std}


def c2_fit(x, y):
    fit = c2_raw(x, y)
    # held-out sharpness: one scalar on (theta - eta), estimated on folds
    f = folds_for(y, K)
    order = np.random.default_rng(7).permutation(len(y))
    eta, th = np.zeros(len(y)), np.zeros((len(y), K - 1))
    for fold in range(f):
        held = order[fold::f]
        train = np.setdiff1d(order, held)
        sub = c2_raw(x[train], y[train])
        eta[held] = ((x[held] - sub["mean"]) / sub["std"]) @ sub["w"]
        th[held] = sub["theta"]

    def nll(t):
        s = np.exp(t[0])
        cum = expit(s * (th - eta[:, None]))
        upper = np.concatenate([cum, np.ones((len(y), 1))], 1)
        lower = np.concatenate([np.zeros((len(y), 1)), cum], 1)
        p = np.clip(upper - lower, 1e-9, None)
        return -np.mean(np.log((p / p.sum(1, keepdims=True))[np.arange(len(y)), y]))

    fit["s"] = float(np.exp(minimize(nll, np.zeros(1), method="L-BFGS-B", bounds=[(-3, 3)]).x[0]))
    return fit


def c2_p(fit, x):
    return c2_probs(fit["w"], fit["theta"], (x - fit["mean"]) / fit["std"], fit["s"])


METHODS = {"current (matrix, L2 to zero)": (current, current_p), "C1 prior-centred": (c1, c1_p), "C2 ordinal 1-D": (c2_fit, c2_p)}


def load(split):
    rows = [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if r["split"] == split and r["method_key"] in MEMBERS]
    per = collections.defaultdict(dict)
    for r in rows:
        per[(r["ladder"], r["item_id"])][r["method_key"]] = r
    data = collections.defaultdict(list)
    for (ladder, item), d in sorted(per.items()):
        if len(d) == len(MEMBERS):
            data[ladder].append((sum((d[m]["logits"] for m in MEMBERS), []), d[MEMBERS[0]]["level"]))
    return {k: (np.array([a for a, _ in v]), np.array([b for _, b in v])) for k, v in data.items()}


def score(p, y):
    return {"acc": float(np.mean(p.argmax(1) == y)), "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))),
            "mae": float(np.mean(np.abs((p * np.arange(p.shape[1])).sum(1) - y))), "ece": ece_equal_mass(p.max(1), p.argmax(1) == y, 15)}


def main():
    data = load("calibration")
    rng = np.random.default_rng(7)
    res = collections.defaultdict(list)
    for ladder, (x, y) in data.items():
        half = len(y) // 2
        xf, yf, xe, ye = x[:half], y[:half], x[half:], y[half:]
        cap = int(np.bincount(yf, minlength=K).min())
        for n in (8, 16, 32, 64, 128, 240):
            per = min(n // K, cap)
            for _ in range(10 if per * K < 200 else 1):
                idx = np.concatenate([rng.choice(np.flatnonzero(yf == lvl), per, replace=False) for lvl in range(K)])
                for name, (fit, prob) in METHODS.items():
                    for key, v in score(prob(fit(xf[idx], yf[idx]), xe), ye).items():
                        res[(name, n, key)].append(v)
    out = {name: {str(n): {k: float(np.mean(res[(name, n, k)])) for k in ("acc", "nll", "mae", "ece")} for n in (8, 16, 32, 64, 128, 240)} for name in METHODS}
    (ROOT / "lab/dev/calibration_challengers.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# Dev: calibration challengers (lab calibration split only; lab/NOTES.md entry 19)", "",
             "Fit on n labels drawn from the first half of each scale's calibration split, judged on its second half; mean over 5 scales x 10 draws.", ""]
    for metric in ("acc", "nll", "mae", "ece"):
        lines += [f"**{metric}**", "", "| method | " + " | ".join(f"n={n}" for n in (8, 16, 32, 64, 128, 240)) + " |", "| --- |" + " --- |" * 6]
        lines += [f"| {name} | " + " | ".join(f"{out[name][str(n)][metric]:.3f}" for n in (8, 16, 32, 64, 128, 240)) + " |" for name in METHODS] + [""]
    (ROOT / "lab/dev/calibration_challengers.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

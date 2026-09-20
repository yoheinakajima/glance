"""E1 of lab/NOTES.md entry 20: a ladder of readouts on the same frozen model, up to a fitted readout on the final hidden
state ("representation ceiling"). CPU only; needs a collection made with `glance.lab.collect --save-hidden`.

Rungs scored here (fit on the calibration split, test split scored once at the full label count):
  ens4d    matrix scaling on the 4 x K member logits (the shipped Glance recipe; must reproduce the published number)
  R4a      logistic readout on PCA components of the ONE `digits` pass's hidden state
  R4b      the same on the four ens4d passes' hidden states, concatenated
  R5       the same on [member logits + hidden-state components]
PCA is fit on all calibration features (labels not needed). PCA size and L2 are chosen by 5-fold cross-validated NLL
inside the fit labels; for the label curve they are chosen on the first draw of each size and reused for the other
draws (cost). Sharpness scalar on held-out folds, as in `rating.fit_matrix(rescale="cv")`.

uv run python tools/readout_ladder.py --bench ladders --logits lab/runs/lab_hidden.jsonl --out lab/READOUT_LADDER
"""
import argparse
import collections
import json
import pathlib

import numpy as np

from glance import rating
from glance.calibration import ece_equal_mass
from glance.evals.metrics import ece_noise_floor
from glance.lab.collect import BENCHES, LAB_DIR, load_ladder_meta
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = rating.MEMBERS["ens4d"]
PCA_GRID, L2_GRID = (32, 64, 128, 256), (0.01, 0.05, 0.2, 1.0, 5.0)
SIZES = (16, 32, 64, 128)


def load_scale(bench: str, scale: str, logit_rows: dict) -> dict | None:
    hidden = {}
    for m in MEMBERS:
        path = LAB_DIR / "hidden" / bench / f"{scale}.{m}.npz"
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=False)
        hidden[m] = dict(zip(z["item_ids"].tolist(), z["hidden"].astype(np.float32)))
    items = [i for i in read_jsonl(LAB_DIR / BENCHES[bench][2] / f"{scale}.jsonl")
             if all(i["item_id"] in hidden[m] and (scale, i["item_id"], m) in logit_rows for m in MEMBERS)]
    if not items:
        return None
    ids = [i["item_id"] for i in items]
    return {
        "split": np.array([i["split"] for i in items]), "y": np.array([i["level"] for i in items]),
        "logits": np.array([sum((logit_rows[(scale, i, m)] for m in MEMBERS), []) for i in ids]),
        "h_digits": np.stack([hidden["digits"][i] for i in ids]),
        "h_all": np.concatenate([np.stack([hidden[m][i] for i in ids]) for m in MEMBERS], axis=1),
    }


class Pca:
    def __init__(self, x: np.ndarray, m_max: int):
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        xs = (x - self.mean) / self.std
        # economy SVD on the (n x d) calibration matrix; components sorted by variance
        _, s, vt = np.linalg.svd(xs - xs.mean(0), full_matrices=False)
        self.center, self.components = xs.mean(0), vt[:m_max]

    def __call__(self, x: np.ndarray, m: int) -> np.ndarray:
        return (((x - self.mean) / self.std) - self.center) @ self.components[:m].T


def cv_nll(z: np.ndarray, y: np.ndarray, k: int, l2: float, seed: int = 7) -> float:
    folds = max(2, min(5, int(np.bincount(y, minlength=k).min())))
    order = np.random.default_rng(seed).permutation(len(y))
    total = 0.0
    for fold in range(folds):
        held = order[fold::folds]
        train = np.setdiff1d(order, held)
        p = rating.apply_matrix(rating.fit_matrix(z[train], y[train], k, l2=l2, rescale="none"), z[held])
        total += -np.sum(np.log(np.clip(p[np.arange(len(held)), y[held]], 1e-12, None)))
    return total / len(y)


def choose(features, y, k):
    """features: dict m -> array for each PCA size (or {0: array} when there is no PCA). Returns (m, l2)."""
    best = min(((cv_nll(z, y, k, l2), m, l2) for m, z in features.items() for l2 in L2_GRID), key=lambda t: t[0])
    return best[1], best[2]


def metrics(p, y):
    pred, conf = p.argmax(1), p.max(1)
    return {"accuracy": float(np.mean(pred == y)), "within_1": float(np.mean(np.abs(pred - y) <= 1)),
            "mae": float(np.mean(np.abs((p * np.arange(p.shape[1])).sum(1) - y))),
            "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))),
            "ece": ece_equal_mass(conf, pred == y, 15), "ece_floor": ece_noise_floor(conf, 15), "n": int(len(y))}


def rung_features(data, fit_mask, rung):
    """dict m -> (fit features, test features builder) for the rung."""
    if rung == "ens4d":
        return None
    base = data["h_digits"] if rung == "R4a" else data["h_all"]
    pca = Pca(base[data["split"] == "calibration"], max(PCA_GRID))
    def build(m):
        z = pca(base, m)
        if rung == "R5":
            lg = data["logits"]
            mu, sd = lg[data["split"] == "calibration"].mean(0), lg[data["split"] == "calibration"].std(0) + 1e-6
            z = np.concatenate([(lg - mu) / sd, z / (z[data["split"] == "calibration"].std(0) + 1e-6)], axis=1)
        return z
    return build


def run_rung(data, rung, idx_fit, test, k, fixed=None):
    y = data["y"]
    if rung == "ens4d":
        fit = rating.fit_matrix(data["logits"][idx_fit], y[idx_fit], k, rescale="cv")
        return metrics(rating.apply_matrix(fit, data["logits"][test]), y[test]), None
    build = rung_features(data, idx_fit, rung)
    grid = {m: build(m) for m in PCA_GRID if m <= len(idx_fit) * 4}
    m, l2 = fixed or choose({m: z[idx_fit] for m, z in grid.items()}, y[idx_fit], k)
    z = grid[m]
    fit = rating.fit_matrix(z[idx_fit], y[idx_fit], k, l2=l2, rescale="cv")
    return {**metrics(rating.apply_matrix(fit, z[test]), y[test]), "pca": m, "l2": l2}, (m, l2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default="ladders", choices=sorted(BENCHES))
    parser.add_argument("--logits", required=True, help="collector JSONL that was written together with the hidden states")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dev", action="store_true", help="calibration split only: fit on its first half, judge on its second half")
    parser.add_argument("--no-curve", action="store_true")
    parser.add_argument("--scales", default=None)
    args = parser.parse_args()

    logit_rows = {(r["ladder"], r["item_id"], r["method_key"]): r["logits"] for r in read_jsonl(ROOT / args.logits) if r["method_key"] in MEMBERS}
    scales = args.scales.split(",") if args.scales else list(load_ladder_meta(args.bench))
    rungs = ("ens4d", "R4a", "R4b", "R5")
    full, curve = collections.defaultdict(dict), collections.defaultdict(lambda: collections.defaultdict(dict))
    rng = np.random.default_rng(7)
    for scale in scales:
        data = load_scale(args.bench, scale, logit_rows)
        if data is None:
            continue
        k = int(data["y"].max()) + 1
        cal = np.flatnonzero(data["split"] == "calibration")
        if args.dev:
            fit_pool, test = cal[: len(cal) // 2], cal[len(cal) // 2:]
        else:
            fit_pool, test = cal, np.flatnonzero(data["split"] == "test")
        if len(fit_pool) < 5 * k or len(test) < 5 * k:
            continue
        for rung in rungs:
            full[rung][scale], _ = run_rung(data, rung, fit_pool, test, k)
        print(scale, {r: round(full[r][scale]["accuracy"], 3) for r in rungs}, flush=True)
        if args.no_curve:
            continue
        y_pool = data["y"][fit_pool]
        for n in SIZES:
            per = n // k
            if per < 2 or per > int(np.bincount(y_pool, minlength=k).min()):
                continue
            for rung in ("ens4d", "R4b", "R5"):
                accs, nlls, fixed = [], [], None
                for _ in range(10):
                    idx = np.concatenate([rng.choice(fit_pool[y_pool == lvl], per, replace=False) for lvl in range(k)])
                    m, fixed_new = run_rung(data, rung, idx, test, k, fixed=fixed)
                    fixed = fixed or fixed_new
                    accs.append(m["accuracy"]); nlls.append(m["nll"])
                curve[rung][scale][str(n)] = {"accuracy": float(np.mean(accs)), "nll": float(np.mean(nlls))}

    done = [s for s in scales if s in full["ens4d"]]
    summary = {r: {key: float(np.mean([full[r][s][key] for s in done])) for key in ("accuracy", "within_1", "mae", "nll", "ece")} for r in rungs} if done else {}
    names = {"ens4d": "`ens4d` + matrix scaling on 4 x K member logits (shipped Glance)", "R4a": "R4a: fitted readout on the hidden state of ONE `digits` pass",
             "R4b": "R4b: fitted readout on the four passes' hidden states", "R5": "R5: fitted readout on member logits + hidden states"}
    where = "DEV: calibration split only (first half fit, second half judged)" if args.dev else "fit on the calibration split, TEST split scored once"
    lines = [f"# Readout ladder on `{args.bench}` ({where})", "", "Registered in `lab/NOTES.md` entry 20 (E1). Same frozen Qwen3-VL-4B, same images, same forward passes; only the readout differs.", "",
             "| Readout | " + " | ".join(done) + " | mean accuracy | within one | MAE | NLL | mean ECE |", "| --- | " + " | ".join("---" for _ in done) + " | --- | --- | --- | --- | --- |"]
    for r in rungs:
        if done:
            s = summary[r]
            lines.append(f"| {names[r]} | " + " | ".join(f"{full[r][sc]['accuracy']:.3f}" for sc in done) + f" | **{s['accuracy']:.3f}** | {s['within_1']:.3f} | {s['mae']:.3f} | {s['nll']:.3f} | {s['ece']:.3f} |")
    if curve:
        sizes = [str(n) for n in SIZES if all(str(n) in curve["ens4d"].get(s, {}) for s in done)]
        lines += ["", "## Labels needed (mean accuracy over scales; 10 draws per size; full = all calibration labels)", "",
                  "| Readout | " + " | ".join(f"n={n}" for n in sizes) + " | full |", "| --- | " + " | ".join("---" for _ in sizes) + " | --- |"]
        for r in ("ens4d", "R4b", "R5"):
            lines.append(f"| {names[r].split(':')[0].split(' +')[0]} | " + " | ".join(f"{np.mean([curve[r][s][n]['accuracy'] for s in done]):.3f}" for n in sizes) + f" | {summary[r]['accuracy']:.3f} |")
    out = ROOT / args.out
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_scale": full, "curve": curve, "dev": args.dev}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

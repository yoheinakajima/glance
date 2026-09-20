"""Classical, no-reference image-quality baselines for the score lab (reviewer-requested honest
baseline: how well do cheap, task-specific features do on the same items and labels as the VLM?).

CPU only. No VLM is loaded, no `glance eval|fit|score|decide|baseline` and no `glance.lab.collect`
run here. Deterministic, seed 7.

Step 1: extract a fixed set of ~30 cheap features per image with numpy / Pillow / scipy only
(variance of the Laplacian, Tenengrad gradient energy, an FFT high-frequency energy ratio, an
Immerkaer noise-sigma estimate, JPEG blockiness at the 8-pixel grid, luminance/percentile/clipping
stats, saturation stats, Hasler-Suesstrunk colourfulness, edge density, local-contrast stats, a
unique-colour/posterization ratio, and grayscale entropy -- see FEATURE_NAMES). A few of the
strictly-positive, heavy-tailed ones (Laplacian variance, Tenengrad, noise sigma) are also kept in
log1p form, since a linear classifier benefits from that. The list was fixed before any test-split
number was looked at; see the "optional extras" section for what else was tried and skipped.

Features are cached to `.cache/classical_features/<bench>/<scale>.npz`, so a rerun is instant.

Step 2: per benchmark and scale, fits a standardized multinomial logistic regression on the
CALIBRATION split only (scikit-learn `LogisticRegression`, C chosen by 5-fold cross-validated
accuracy on the calibration split from {0.01, 0.1, 1, 10}), and reports on the TEST split:
accuracy, within-one-level accuracy, and the MAE of the expected level (the probability-weighted
mean level). For KADID it also reports the Spearman correlation between the expected level and
`-dmos` (both taken from the same TEST-split rows), per distortion. The same is repeated with a
small, class-balanced calibration subsample (32 labels for 4-level scales -- 8 per level; 30 labels
for 5-level scales -- 6 per level, which is the low end of the 30-35 the brief allows), mean and
std over 20 seeded draws, because the VLM method in this project is usually quoted at about 32
labels.

The test split is touched only to compute the numbers above; nothing here is fit on it.

Writes `results/lab/classical_baselines.json` and `results/lab/classical_baselines.md`.

uv run python tools/classical_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import convolve2d
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 7
C_GRID = (0.01, 0.1, 1.0, 10.0)
N_CV_FOLDS = 5
N_DRAWS = 20
MAX_WORKERS = 4

# Published VLM numbers (lab scales only; see docs/paper/RESULTS_LAB.md section 3). Not invented,
# not re-derived here: these are the committed `ens4d` test-split accuracies and the v0-as-shipped
# mean, copied verbatim.
VLM_ENS4D_ACCURACY = {"blur": 0.888, "noise": 0.864, "jpeg": 0.772, "exposure": 0.924, "resolution": 0.888}
VLM_V0_SHIPPED_MEAN_ACCURACY = 0.500

BENCHMARKS = {
    "lab": {
        "manifest_dir": PROJECT_ROOT / "lab" / "manifests",
        "title": "Lab scales (blur, noise, jpeg, exposure, resolution; 4 levels, 1,000 items each)",
    },
    "distort25": {
        "manifest_dir": PROJECT_ROOT / "lab" / "manifests_distort25",
        "title": "distort25 (25 distortions, 5 levels, 300 items each)",
    },
    "kadid": {
        "manifest_dir": PROJECT_ROOT / "lab" / "manifests_kadid",
        "title": "KADID-10k (25 distortions, 5 levels, 405 items each; evaluation only)",
    },
}


# ============================================================================================
# Step 1: features
# ============================================================================================

FEATURE_NAMES = (
    "lap_var", "lap_var_log",
    "tenengrad", "tenengrad_log", "edge_mean_mag", "edge_density",
    "hf_energy_ratio",
    "noise_sigma", "noise_sigma_log",
    "blockiness_col", "blockiness_row", "blockiness_mean", "blockiness_ratio",
    "lum_mean", "lum_std", "lum_p5", "lum_p50", "lum_p95", "clip_low_frac", "clip_high_frac",
    "sat_mean", "sat_std",
    "colourfulness",
    "local_contrast_mean", "local_contrast_std",
    "rms_contrast", "michelson_contrast",
    "unique_color_ratio", "gray_entropy",
)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian ([[0,1,0],[1,-4,1],[0,1,0]]). Lower for blurrier images."""
    padded = np.pad(gray, 1, mode="edge")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * padded[1:-1, 1:-1]
    return float(lap.var())


def _hf_energy_ratio(gray: np.ndarray, low_frac: float = 0.25) -> float:
    """Fraction of FFT magnitude-squared energy outside a central disk of radius `low_frac` times
    the half-min-dimension. Lower for blurrier / lower-resolution images."""
    f = np.fft.fft2(gray - gray.mean())
    mag2 = np.abs(np.fft.fftshift(f)) ** 2
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = low_frac * min(cy, cx)
    total = mag2.sum() + 1e-12
    return float(mag2[dist > radius].sum() / total)


def _noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise-sigma estimator: convolve with [[1,-2,1],[-2,4,-2],[1,-2,1]] (zero
    response on a smooth or linear ramp) and scale the mean absolute response."""
    h, w = gray.shape
    if h <= 2 or w <= 2:
        return 0.0
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = convolve2d(gray, mask, mode="valid")
    denom = 6.0 * (w - 2) * (h - 2)
    return float(np.sqrt(np.pi / 2.0) * np.sum(np.abs(conv)) / denom)


def _blockiness(gray: np.ndarray) -> tuple[float, float, float, float]:
    """JPEG blockiness at the 8-pixel grid: mean absolute pixel difference across an 8x8 block
    boundary minus the mean absolute difference inside blocks, separately for the column grid
    (vertical lines) and the row grid (horizontal lines). Returns (col, row, mean, ratio); ratio is
    (boundary energy) / (interior energy), robust to the overall gradient scale of the image."""
    col_diff = np.abs(np.diff(gray, axis=1))  # (h, w-1): differences across columns
    col_idx = np.arange(col_diff.shape[1])
    boundary = (col_idx + 1) % 8 == 0
    b_col = float(col_diff[:, boundary].mean()) if boundary.any() else 0.0
    nb_col = float(col_diff[:, ~boundary].mean()) if (~boundary).any() else 0.0

    row_diff = np.abs(np.diff(gray, axis=0))  # (h-1, w): differences across rows
    row_idx = np.arange(row_diff.shape[0])
    boundary_r = (row_idx + 1) % 8 == 0
    b_row = float(row_diff[boundary_r, :].mean()) if boundary_r.any() else 0.0
    nb_row = float(row_diff[~boundary_r, :].mean()) if (~boundary_r).any() else 0.0

    blockiness_col = b_col - nb_col
    blockiness_row = b_row - nb_row
    blockiness_mean = 0.5 * (blockiness_col + blockiness_row)
    blockiness_ratio = (b_col + b_row) / (nb_col + nb_row + 1e-6)
    return blockiness_col, blockiness_row, blockiness_mean, blockiness_ratio


def _colourfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _unique_color_ratio(rgb: np.ndarray) -> float:
    """Unique colours (quantized to 16 levels per channel, to be robust to +/-1 pixel noise) as a
    fraction of pixel count. Lower for posterized / colour-quantized images."""
    quant = (rgb // 16).astype(np.int32)
    codes = (quant[..., 0] * 16 + quant[..., 1]) * 16 + quant[..., 2]
    return float(len(np.unique(codes)) / codes.size)


def _entropy(gray: np.ndarray) -> float:
    """Shannon entropy (bits) of the 256-bin grayscale histogram."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_features(img: Image.Image) -> np.ndarray:
    """Fixed-order feature vector (len(FEATURE_NAMES),) for one PIL image. Deterministic: no
    randomness anywhere in this function."""
    img = img.convert("RGB")
    rgb = np.asarray(img, dtype=np.float64)
    gray = np.asarray(img.convert("L"), dtype=np.float64)

    lap_var = _laplacian_variance(gray)
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    grad_mag = np.hypot(gx, gy)
    tenengrad = float(np.mean(grad_mag**2))
    edge_mean_mag = float(np.mean(grad_mag))
    edge_thresh = grad_mag.mean() + grad_mag.std()
    edge_density = float(np.mean(grad_mag > edge_thresh))

    hf_ratio = _hf_energy_ratio(gray)
    noise_sigma = _noise_sigma(gray)
    b_col, b_row, b_mean, b_ratio = _blockiness(gray)

    lum_mean = float(gray.mean())
    lum_std = float(gray.std())
    p1, p5, p50, p95, p99 = np.percentile(gray, [1, 5, 50, 95, 99])
    clip_low = float(np.mean(gray <= 2.0))
    clip_high = float(np.mean(gray >= 253.0))

    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    sat = hsv[..., 1] / 255.0
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())

    colourfulness = _colourfulness(rgb)

    local_mean = ndi.uniform_filter(gray, size=8)
    local_diff = gray - local_mean
    local_contrast_mean = float(np.mean(np.abs(local_diff)))
    local_contrast_std = float(np.std(local_diff))

    rms_contrast = float(lum_std / (lum_mean + 1e-6))
    michelson_contrast = float((p99 - p1) / (p99 + p1 + 1e-6))

    unique_color_ratio = _unique_color_ratio(rgb)
    gray_entropy = _entropy(gray)

    values = {
        "lap_var": lap_var, "lap_var_log": float(np.log1p(lap_var)),
        "tenengrad": tenengrad, "tenengrad_log": float(np.log1p(tenengrad)),
        "edge_mean_mag": edge_mean_mag, "edge_density": edge_density,
        "hf_energy_ratio": hf_ratio,
        "noise_sigma": noise_sigma, "noise_sigma_log": float(np.log1p(noise_sigma)),
        "blockiness_col": b_col, "blockiness_row": b_row, "blockiness_mean": b_mean, "blockiness_ratio": b_ratio,
        "lum_mean": lum_mean, "lum_std": lum_std, "lum_p5": float(p5), "lum_p50": float(p50), "lum_p95": float(p95),
        "clip_low_frac": clip_low, "clip_high_frac": clip_high,
        "sat_mean": sat_mean, "sat_std": sat_std,
        "colourfulness": colourfulness,
        "local_contrast_mean": local_contrast_mean, "local_contrast_std": local_contrast_std,
        "rms_contrast": rms_contrast, "michelson_contrast": michelson_contrast,
        "unique_color_ratio": unique_color_ratio, "gray_entropy": gray_entropy,
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


def _extract_from_path(path_str: str) -> np.ndarray:
    with Image.open(path_str) as img:
        return extract_features(img)


# ============================================================================================
# Manifests, caching
# ============================================================================================


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cache_path(bench: str, scale: str) -> Path:
    return PROJECT_ROOT / ".cache" / "classical_features" / bench / f"{scale}.npz"


def build_or_load_features(bench: str, scale: str, rows: list[dict[str, Any]], workers: int, force: bool) -> dict[str, np.ndarray]:
    cpath = cache_path(bench, scale)
    item_ids = np.array([r["item_id"] for r in rows])
    if not force and cpath.exists():
        cached = np.load(cpath, allow_pickle=True)
        if list(cached["item_id"]) == list(item_ids) and list(cached["feature_names"]) == list(FEATURE_NAMES):
            return {k: cached[k] for k in cached.files}
    paths = [str(PROJECT_ROOT / r["path"]) for r in rows]
    workers = max(1, min(workers, MAX_WORKERS, len(paths)))
    if workers == 1:
        feats = np.array([_extract_from_path(p) for p in paths], dtype=np.float64)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            feats = np.array(list(ex.map(_extract_from_path, paths, chunksize=16)), dtype=np.float64)
    levels = np.array([int(r["level"]) for r in rows], dtype=np.int64)
    splits = np.array([r["split"] for r in rows])
    dmos = np.array([float(r["dmos"]) if "dmos" in r else np.nan for r in rows], dtype=np.float64)
    out = {"item_id": item_ids, "level": levels, "split": splits, "features": feats, "dmos": dmos,
           "feature_names": np.array(FEATURE_NAMES)}
    cpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cpath, **out)
    return out


# ============================================================================================
# Step 2: fit + evaluate
# ============================================================================================


def _cv_n_splits(y: np.ndarray) -> int:
    counts = np.bincount(y)
    counts = counts[counts > 0]
    return max(2, min(N_CV_FOLDS, int(counts.min())))


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, n_levels: int, seed: int) -> tuple[np.ndarray, float, dict[float, float]]:
    """Standardize on Xtr, choose C by cross-validated accuracy on Xtr, refit on all of Xtr, return
    test-split probabilities (n_test, n_levels), the chosen C, and the per-C CV accuracy."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    n_splits = _cv_n_splits(ytr)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_acc: dict[float, float] = {}
    for c in C_GRID:
        accs = []
        for tr_idx, va_idx in skf.split(Xtr_s, ytr):
            clf = LogisticRegression(C=c, max_iter=5000)
            clf.fit(Xtr_s[tr_idx], ytr[tr_idx])
            accs.append(clf.score(Xtr_s[va_idx], ytr[va_idx]))
        cv_acc[c] = float(np.mean(accs))
    best_c = max(cv_acc, key=lambda c: cv_acc[c])
    clf = LogisticRegression(C=best_c, max_iter=5000)
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)
    full_proba = np.zeros((len(Xte), n_levels))
    for j, cls in enumerate(clf.classes_):
        full_proba[:, int(cls)] = proba[:, j]
    return full_proba, best_c, cv_acc


def _metrics(proba: np.ndarray, y: np.ndarray, n_levels: int) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    expected = (proba * np.arange(n_levels)).sum(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "within_1": float(np.mean(np.abs(pred - y) <= 1)),
        "mae": float(np.mean(np.abs(expected - y))),
        "expected_level": expected,
    }


def _balanced_sample(y: np.ndarray, n_per_level: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for lvl in np.unique(y):
        lvl_idx = np.where(y == lvl)[0]
        n = min(n_per_level, len(lvl_idx))
        idx.append(rng.choice(lvl_idx, size=n, replace=False))
    return np.concatenate(idx)


def evaluate_scale(bench: str, scale: str, data: dict[str, np.ndarray]) -> dict[str, Any]:
    levels, splits, feats, dmos = data["level"], data["split"], data["features"], data["dmos"]
    n_levels = int(levels.max() + 1)
    cal = splits == "calibration"
    test = splits == "test"
    Xcal, ycal = feats[cal], levels[cal]
    Xtest, ytest = feats[test], levels[test]
    dmos_test = dmos[test]

    proba_full, c_full, cv_full = _fit_predict(Xcal, ycal, Xtest, n_levels, SEED)
    m_full = _metrics(proba_full, ytest, n_levels)
    result_full = {"chosen_C": c_full, "cv_accuracy_by_C": {str(k): v for k, v in cv_full.items()},
                   "accuracy": m_full["accuracy"], "within_1": m_full["within_1"], "mae": m_full["mae"], "n": m_full["n"]}
    if not np.all(np.isnan(dmos_test)):
        rho = spearmanr(m_full["expected_level"], -dmos_test).statistic
        result_full["spearman_vs_neg_dmos"] = float(rho)

    n_per_level = 8 if n_levels == 4 else 6
    n_total_target = n_per_level * n_levels
    ss = np.random.SeedSequence(SEED)
    draw_seeds = ss.spawn(N_DRAWS)
    accs, withs, maes, rhos = [], [], [], []
    for draw_seed in draw_seeds:
        rng = np.random.default_rng(draw_seed)
        idx = _balanced_sample(ycal, n_per_level, rng)
        proba_small, _, _ = _fit_predict(Xcal[idx], ycal[idx], Xtest, n_levels, seed=int(draw_seed.generate_state(1)[0]))
        m_small = _metrics(proba_small, ytest, n_levels)
        accs.append(m_small["accuracy"])
        withs.append(m_small["within_1"])
        maes.append(m_small["mae"])
        if not np.all(np.isnan(dmos_test)):
            rhos.append(float(spearmanr(m_small["expected_level"], -dmos_test).statistic))
    result_small = {"n_per_level": n_per_level, "n_total": n_total_target, "draws": N_DRAWS,
                     "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                     "within_1_mean": float(np.mean(withs)), "within_1_std": float(np.std(withs)),
                     "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes))}
    if rhos:
        result_small["spearman_vs_neg_dmos_mean"] = float(np.mean(rhos))
        result_small["spearman_vs_neg_dmos_std"] = float(np.std(rhos))

    out = {"n_levels": n_levels, "n_calibration": int(cal.sum()), "n_test": int(test.sum()),
           "n_features": feats.shape[1], "full": result_full, "small_n": result_small}
    if bench == "lab" and scale in VLM_ENS4D_ACCURACY:
        out["vlm_ens4d_accuracy"] = VLM_ENS4D_ACCURACY[scale]
    return out


# ============================================================================================
# Timing
# ============================================================================================


def measure_timing(n: int = 100) -> dict[str, Any]:
    """p50/p90 feature-extraction time per image, single process, on already-decoded PIL images
    (image file I/O is excluded on purpose: this measures the cost described in the module
    docstring, step 1, not disk/decode cost)."""
    manifest = PROJECT_ROOT / "lab" / "manifests" / "blur.jsonl"
    if not manifest.exists():
        return {"error": "lab/manifests/blur.jsonl missing; timing skipped"}
    rows = load_manifest(manifest)[:n]
    images = []
    for r in rows:
        with Image.open(PROJECT_ROOT / r["path"]) as im:
            images.append(im.convert("RGB").copy())
    for im in images[: min(5, len(images))]:  # warm up (first FFT plan, etc.)
        extract_features(im)
    times = []
    for im in images:
        t0 = time.perf_counter()
        extract_features(im)
        times.append(time.perf_counter() - t0)
    times_ms = np.array(times) * 1000.0
    return {"n_images": len(images), "p50_ms": float(np.percentile(times_ms, 50)),
            "p90_ms": float(np.percentile(times_ms, 90)), "mean_ms": float(np.mean(times_ms)),
            "source": "lab/manifests/blur.jsonl, calibration split, first 100 items",
            "note": "feature computation only, on an already-decoded RGB PIL image; excludes file I/O and JPEG/PNG decode"}


# ============================================================================================
# Optional extras: BRISQUE / NIQE / CLIP-IQA
# ============================================================================================

OPTIONAL_EXTRAS = {
    "brisque": {
        "status": "skipped",
        "package_checked": "piq 0.8.0 (Apache-2.0, https://github.com/photosynthesis-team/piq, "
                            "verified via PyPI classifiers and the repo's LICENSE file)",
        "reason": "piq's own code is Apache-2.0 and pip-installs cleanly (only extra dep is "
                  "torchvision, already required by this project). But piq.brisque() downloads a "
                  "pretrained SVR weight file (brisque_svm_weights.pt, ~112 KB, from piq's GitHub "
                  "releases) whose support vectors were fit on the LIVE lab's BRISQUE reference "
                  "release (https://live.ece.utexas.edu/research/Quality/index_algorithms.htm, "
                  "GitHub org utlive/BRISQUE). That upstream repo carries no LICENSE file (GitHub's "
                  "detected license is null) and the LIVE page only asks for a citation, so the "
                  "terms for reusing a model trained on their data are unclear. Per the task's own "
                  "rule (check the license of any weights a package downloads), this was skipped.",
    },
    "niqe": {
        "status": "skipped",
        "reason": "piq (the one package that passed the license check for its own code) does not "
                  "implement NIQE. The only other permissively-licensed candidate found in the time "
                  "available, `image-quality` (ocampor/image-quality, Apache-2.0), also only "
                  "implements BRISQUE, not NIQE, and would carry the same LIVE-derived-weights "
                  "concern as above. No verified-license NIQE implementation was added.",
    },
    "clip_iqa": {
        "status": "skipped",
        "reason": "piq.CLIPIQA() automatically downloads a CLIP RN50 checkpoint (piq's own mirror "
                  "of OpenAI's MIT-licensed CLIP weights, piq/releases/download/v0.7.1/RN50.pt, on "
                  "the order of a few hundred MB) the first time it runs. That is both a large, "
                  "unprompted download this task should not make on its own, and, in spirit, loading "
                  "a vision-language model -- which the task explicitly ruled out for this CPU-only "
                  "run. Skipped for both reasons; not attempted.",
    },
}


# ============================================================================================
# Orchestration
# ============================================================================================


def run_benchmark(bench: str, spec: dict[str, Any], workers: int, force: bool) -> dict[str, Any]:
    manifest_dir = spec["manifest_dir"]
    if not manifest_dir.exists():
        print(f"[{bench}] manifest dir missing ({manifest_dir}); skipping benchmark")
        return {"skipped": True, "reason": f"manifest dir missing: {manifest_dir}"}
    manifest_files = sorted(manifest_dir.glob("*.jsonl"))
    if not manifest_files:
        print(f"[{bench}] no manifests found in {manifest_dir}; skipping benchmark")
        return {"skipped": True, "reason": f"no manifests in {manifest_dir}"}

    scales: dict[str, Any] = {}
    for mpath in manifest_files:
        scale = mpath.stem
        rows = load_manifest(mpath)
        if not rows:
            print(f"[{bench}/{scale}] empty manifest; skipping scale")
            scales[scale] = {"skipped": True, "reason": "empty manifest"}
            continue
        first_image = PROJECT_ROOT / rows[0]["path"]
        if not first_image.exists():
            print(f"[{bench}/{scale}] images missing ({first_image}); skipping scale")
            scales[scale] = {"skipped": True, "reason": f"images missing: {first_image}"}
            continue
        t0 = time.time()
        data = build_or_load_features(bench, scale, rows, workers=workers, force=force)
        result = evaluate_scale(bench, scale, data)
        result["extract_and_fit_seconds"] = round(time.time() - t0, 2)
        scales[scale] = result
        print(f"[{bench}/{scale}] n={len(rows)} levels={result['n_levels']} "
              f"acc(full)={result['full']['accuracy']:.3f} acc(n={result['small_n']['n_total']})="
              f"{result['small_n']['accuracy_mean']:.3f} ({time.time() - t0:.1f}s)")

    ok_scales = {k: v for k, v in scales.items() if not v.get("skipped")}
    summary = {}
    if ok_scales:
        summary = {
            "n_scales": len(ok_scales),
            "mean_accuracy_full": float(np.mean([v["full"]["accuracy"] for v in ok_scales.values()])),
            "mean_within_1_full": float(np.mean([v["full"]["within_1"] for v in ok_scales.values()])),
            "mean_mae_full": float(np.mean([v["full"]["mae"] for v in ok_scales.values()])),
            "mean_accuracy_small_n": float(np.mean([v["small_n"]["accuracy_mean"] for v in ok_scales.values()])),
            "mean_within_1_small_n": float(np.mean([v["small_n"]["within_1_mean"] for v in ok_scales.values()])),
            "mean_mae_small_n": float(np.mean([v["small_n"]["mae_mean"] for v in ok_scales.values()])),
        }
        rhos = [v["full"]["spearman_vs_neg_dmos"] for v in ok_scales.values() if "spearman_vs_neg_dmos" in v["full"]]
        if rhos:
            summary["mean_spearman_vs_neg_dmos_full"] = float(np.mean(rhos))
    return {"title": spec["title"], "scales": scales, "summary": summary}


def render_markdown(results: dict[str, Any]) -> str:
    out = ["# Classical no-reference baselines vs the score lab", "",
           "Reviewer-requested honest baseline: how well do cheap, task-specific image-quality "
           "features do on the same items and labels the VLM is scored on. Features and the fit "
           "procedure are in `tools/classical_baselines.py`. CPU only, seed 7. The test split is "
           "used only to report; the calibration split is the only thing anything is fit on.", "",
           f"Feature extraction time: p50 {results['timing'].get('p50_ms', float('nan')):.2f} ms, "
           f"p90 {results['timing'].get('p90_ms', float('nan')):.2f} ms per image "
           f"(n={results['timing'].get('n_images', 0)}, single process, feature computation only, "
           "not counting file read or image decode). Machine: this development machine, one run.", ""]

    out += ["## Optional extras that were tried", ""]
    for name, info in results["optional_extras"].items():
        out.append(f"- **{name}**: {info['status']}. {info['reason']}")
    out.append("")

    out += ["## Summary (mean over scales in each benchmark)", "",
            "| Benchmark | Scales | Accuracy (full calibration) | Accuracy (small calibration) | "
            "Within 1 level | MAE (levels) | Spearman vs -DMOS |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for bench, spec in BENCHMARKS.items():
        b = results["benchmarks"].get(bench) or {"skipped": True, "reason": "not run"}
        if b.get("skipped") or not b.get("summary"):
            out.append(f"| {bench} | - | skipped: {b.get('reason', 'no scales evaluated')} | | | | |")
            continue
        s = b["summary"]
        rho = f"{s['mean_spearman_vs_neg_dmos_full']:.3f}" if "mean_spearman_vs_neg_dmos_full" in s else "-"
        out.append(f"| {bench} | {s['n_scales']} | {s['mean_accuracy_full']:.3f} | "
                   f"{s['mean_accuracy_small_n']:.3f} | {s['mean_within_1_full']:.3f} | "
                   f"{s['mean_mae_full']:.3f} | {rho} |")
    out.append("")
    out.append(f"For context, the lab scales' published VLM numbers: `ens4d` mean accuracy "
               f"{np.mean(list(VLM_ENS4D_ACCURACY.values())):.3f}; v0 as shipped, mean accuracy "
               f"{VLM_V0_SHIPPED_MEAN_ACCURACY:.3f} (per-scale v0 numbers are not published, so only "
               "the mean is shown). No VLM numbers are given for distort25 or KADID: those runs are "
               "not finished.")
    out.append("")

    for bench, spec in BENCHMARKS.items():
        b = results["benchmarks"].get(bench) or {"skipped": True, "reason": "not run"}
        out.append(f"## {spec['title']}")
        out.append("")
        if b.get("skipped") or not b.get("scales"):
            out.append(f"Skipped: {b.get('reason', 'no scales evaluated')}")
            out.append("")
            continue
        has_dmos = any("spearman_vs_neg_dmos" in v["full"] for v in b["scales"].values() if not v.get("skipped"))
        has_vlm = bench == "lab"
        header = ["Scale", "Levels", "n test", "Acc (full)", f"Acc (n={{n}})", "Within 1", "MAE"]
        if has_dmos:
            header.append("Spearman vs -DMOS")
        if has_vlm:
            header.append("VLM `ens4d` acc")
        table = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        for scale, v in b["scales"].items():
            if v.get("skipped"):
                row = [scale, "-", "-", f"skipped: {v['reason']}", "-", "-", "-"]
                if has_dmos:
                    row.append("-")
                if has_vlm:
                    row.append("-")
                table.append("| " + " | ".join(row) + " |")
                continue
            f, sm = v["full"], v["small_n"]
            row = [scale, str(v["n_levels"]), str(v["n_test"]), f"{f['accuracy']:.3f}",
                   f"{sm['accuracy_mean']:.3f} ± {sm['accuracy_std']:.3f} (n={sm['n_total']})",
                   f"{f['within_1']:.3f}", f"{f['mae']:.3f}"]
            if has_dmos:
                row.append(f"{f.get('spearman_vs_neg_dmos', float('nan')):.3f}" if "spearman_vs_neg_dmos" in f else "-")
            if has_vlm:
                row.append(f"{v['vlm_ens4d_accuracy']:.3f}" if "vlm_ens4d_accuracy" in v else "-")
            table.append("| " + " | ".join(row) + " |")
        header[4] = header[4].format(n="~32" if bench == "lab" else "~30")
        table[0] = "| " + " | ".join(header) + " |"
        out += table + [""]

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"feature-extraction worker processes (capped at {MAX_WORKERS})")
    parser.add_argument("--force", action="store_true", help="ignore the feature cache and recompute")
    parser.add_argument("--only", help="comma-separated benchmark names to run (default: all of lab,distort25,kadid)")
    parser.add_argument("--out-json", default="results/lab/classical_baselines.json")
    parser.add_argument("--out-md", default="results/lab/classical_baselines.md")
    args = parser.parse_args(argv)

    np.random.seed(SEED)
    workers = max(1, min(args.workers, MAX_WORKERS))
    only = set(args.only.split(",")) if args.only else set(BENCHMARKS)

    print("Measuring feature-extraction timing (single process, p50/p90 over 100 images)...")
    timing = measure_timing(100)
    print(f"  p50={timing.get('p50_ms', float('nan')):.2f} ms  p90={timing.get('p90_ms', float('nan')):.2f} ms")

    benchmarks_out = OrderedDict()
    for bench, spec in BENCHMARKS.items():
        if bench not in only:
            continue
        print(f"\n=== {bench}: {spec['title']} ===")
        benchmarks_out[bench] = run_benchmark(bench, spec, workers=workers, force=args.force)

    results = {
        "seed": SEED, "c_grid": list(C_GRID), "cv_folds": N_CV_FOLDS, "small_n_draws": N_DRAWS,
        "feature_names": list(FEATURE_NAMES), "n_features": len(FEATURE_NAMES),
        "timing": timing, "optional_extras": OPTIONAL_EXTRAS, "benchmarks": benchmarks_out,
        "vlm_context": {"ens4d_by_scale": VLM_ENS4D_ACCURACY, "v0_shipped_mean_accuracy": VLM_V0_SHIPPED_MEAN_ACCURACY},
    }

    out_json = PROJECT_ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"\nwrote {out_json}")

    md = render_markdown(results)
    out_md = PROJECT_ROOT / args.out_md
    out_md.write_text(md)
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Five synthetic image-degradation ladders for testing how well a VLM rates images on 4-level scales.

Source images come from the Oxford-IIIT Pet test split (parquet, already cached), excluding every
`image_id` used by `pets37` (see `glance/evals/manifests/pets37.jsonl`). Every source is first decoded,
converted to RGB, and resized so its longest side is exactly `BASE_SIDE` (never upscaled); every ladder
operates on that shared 448 px base image, so one parameter value means the same thing everywhere.

Each ladder (blur, noise, jpeg, exposure, resolution) has 4 fixed parameter values, frozen in `LADDERS`
below together with the level descriptions shown to the rater. `build()` renders `N_PER_LADDER` items per
ladder plus `N_ANCHOR_SOURCES` reference sources (rendered at all 4 levels, never used as items), writes a
committed manifest per ladder under `lab/manifests/`, and a shared `lab/anchors.json`. Generated images live
under `.cache/lab_images/` (gitignored).

Run `python -m glance.lab.ladders` to build everything and write one contact sheet per ladder under
`lab/sheets/`.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ..config import Config, PROJECT_ROOT
from ..evals.suites.base import MANIFEST_DIR as EVAL_MANIFEST_DIR

BASE_SIDE = 448
SEED = 7
N_PER_LADDER = 1000
N_ANCHOR_SOURCES = 3

LAB_DIR = PROJECT_ROOT / "lab"
IMAGES_DIR = PROJECT_ROOT / ".cache" / "lab_images"
MANIFESTS_DIR = LAB_DIR / "manifests"
SHEETS_DIR = LAB_DIR / "sheets"
ANCHORS_PATH = LAB_DIR / "anchors.json"
PETS37_MANIFEST = EVAL_MANIFEST_DIR / "pets37.jsonl"

# Populated by _filtered_pool() (via build()/contact_sheet()) for diagnostics; not part of the public API.
LAST_POOL_STATS: dict[str, Any] = {}

LADDERS: dict[str, dict[str, Any]] = {
    "blur": {
        "instructions": "How blurry is `img0`?",
        "levels": [
            "Sharp: edges are crisp and fine texture such as fur or fabric is visible",
            "Slightly soft: edges are a little smooth and the finest texture is lost, but everything is easy "
            "to make out",
            "Clearly blurred: only larger shapes and colors are left, the subject is still recognizable",
            "Heavily blurred: vague blobs of color, the subject is hard to identify",
        ],
        "params": [0, 1.2, 3.0, 7.0],  # Gaussian blur radius in pixels at BASE_SIDE; level 0 = untouched base
    },
    "noise": {
        "instructions": "How much grain or noise is in `img0`?",
        "levels": [
            "Clean: smooth areas look smooth, no visible grain",
            "Light grain: fine speckle is visible in smooth areas, details are unaffected",
            "Noisy: speckle covers the whole image and hides fine detail",
            "Very noisy: heavy speckle dominates, the subject is hard to see",
        ],
        "params": [0, 10, 25, 60],  # additive Gaussian noise sigma, 0-255 scale, per-pixel per-channel
    },
    "jpeg": {
        "instructions": "How strong are the compression artifacts in `img0`?",
        "levels": [
            "None: smooth gradients and clean edges",
            "Mild: faint blockiness or ringing around edges when you look closely",
            "Strong: obvious square blocks and smeared detail",
            "Severe: large flat blocks and banding of color, fine detail is gone",
        ],
        "params": [95, 25, 10, 3],  # JPEG quality
    },
    "exposure": {
        "instructions": "How underexposed (too dark) is `img0`?",
        "levels": [
            "Well exposed: bright areas and shadows both show detail",
            "Somewhat dark: the image is dim but everything is easy to see",
            "Dark: shadows are crushed and the subject is hard to see",
            "Nearly black: only the brightest parts are visible",
        ],
        "params": [1.0, 0.5, 0.25, 0.1],  # brightness gain, multiplied into pixel values and clipped
    },
    "resolution": {
        "instructions": "How low is the resolution of `img0`?",
        "levels": [
            "Full resolution: fine detail is rendered cleanly",
            "Slightly low: fine detail is soft, as if enlarged a little",
            "Low: visibly enlarged from a small image, edges are jagged or smeared",
            "Very low: coarse, enlarged from a tiny thumbnail, only rough shapes remain",
        ],
        "params": [1, 3, 6, 12],  # downscale factor (BOX down, BICUBIC back up); factor 1 = untouched
    },
}


@dataclass
class _Source:
    image_id: str
    sharpness: float
    luminance: float


# --- pure degradation -------------------------------------------------------------------------------


def degrade(base: Image.Image, ladder: str, level: int, seed: int) -> Image.Image:
    """Apply `ladder` at `level` to `base`. Deterministic given its arguments; returns an RGB image the
    same size as `base`."""
    if base.mode != "RGB":
        base = base.convert("RGB")
    param = LADDERS[ladder]["params"][level]
    if ladder == "blur":
        return base.copy() if param == 0 else base.filter(ImageFilter.GaussianBlur(param))
    if ladder == "noise":
        return _apply_noise(base, sigma=param, seed=seed)
    if ladder == "jpeg":
        return _apply_jpeg(base, quality=int(param))
    if ladder == "exposure":
        return _apply_exposure(base, gain=param)
    if ladder == "resolution":
        return _apply_resolution(base, factor=int(param))
    raise ValueError(f"unknown ladder {ladder!r}")


def _apply_noise(base: Image.Image, sigma: float, seed: int) -> Image.Image:
    if sigma == 0:
        return base.copy()
    rng = np.random.default_rng(seed)
    arr = np.asarray(base, dtype=np.float64)
    noisy = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8), mode="RGB")


def _apply_jpeg(base: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_exposure(base: Image.Image, gain: float) -> Image.Image:
    if gain == 1.0:
        return base.copy()
    arr = np.asarray(base, dtype=np.float64)
    dimmed = np.clip(arr * gain, 0, 255)
    return Image.fromarray(dimmed.astype(np.uint8), mode="RGB")


def _apply_resolution(base: Image.Image, factor: int) -> Image.Image:
    if factor == 1:
        return base.copy()
    w, h = base.size
    small = base.resize((max(1, w // factor), max(1, h // factor)), Image.BOX)
    return small.resize((w, h), Image.BICUBIC)


# --- sharpness / luminance ---------------------------------------------------------------------------


def _grayscale_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian ([[0,1,0],[1,-4,1],[0,1,0]]) over `gray`, via numpy slicing."""
    padded = np.pad(gray, 1, mode="edge")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * padded[1:-1, 1:-1]
    return float(lap.var())


def _luminance(gray: np.ndarray) -> float:
    return float(gray.mean())


# --- source pool --------------------------------------------------------------------------------------


def _to_base(img: Image.Image) -> Image.Image:
    """Resize so the longest side is exactly BASE_SIDE (LANCZOS). Never upscales."""
    scale = BASE_SIDE / max(img.size)
    if scale >= 1.0:
        return img
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _load_raw_bytes(cfg: Config) -> dict[str, bytes]:
    """image_id -> original encoded bytes, for every pets37 test-split row not used by the pets37 suite."""
    import pyarrow.parquet as pq

    from ..evals.suites import pets37
    from ..evals.suites.sources import hf_parquet
    from ..logging_utils import read_jsonl

    path = hf_parquet(cfg, pets37.REPO, pets37.REVISION, "data/test-00000-of-00001.parquet")
    excluded = {r["item_id"] for r in read_jsonl(PETS37_MANIFEST)}
    table = pq.read_table(path, columns=["image_id", "image"])
    out: dict[str, bytes] = {}
    for row in table.to_pylist():
        image_id = str(row["image_id"])
        if image_id not in excluded:
            out[image_id] = row["image"]["bytes"]
    return out


def _get_base_image(raw_bytes: dict[str, bytes], image_id: str) -> Image.Image:
    img = Image.open(io.BytesIO(raw_bytes[image_id])).convert("RGB")
    return _to_base(img)


def _filtered_pool(raw_bytes: dict[str, bytes]) -> list[_Source]:
    """Deterministic (image_id-sorted) list of sources passing the min-size, sharpness-percentile, and
    luminance filters. Updates the module-level `LAST_POOL_STATS` diagnostic dict as a side effect."""
    stats: dict[str, Any] = {"candidates": len(raw_bytes)}
    measured: list[tuple[str, float, float]] = []
    for image_id in sorted(raw_bytes):
        img = Image.open(io.BytesIO(raw_bytes[image_id])).convert("RGB")
        if max(img.size) < BASE_SIDE:
            continue
        base = _to_base(img)
        gray = _grayscale_array(base)
        measured.append((image_id, _laplacian_variance(gray), _luminance(gray)))
    stats["after_min_size"] = len(measured)

    sharpness_values = np.array([m[1] for m in measured])
    threshold = float(np.percentile(sharpness_values, 30)) if len(sharpness_values) else 0.0
    stats["sharpness_p30"] = threshold
    after_sharpness = [m for m in measured if m[1] >= threshold]
    stats["after_sharpness"] = len(after_sharpness)

    final = [m for m in after_sharpness if m[2] >= 70.0]
    stats["after_luminance"] = len(final)

    LAST_POOL_STATS.clear()
    LAST_POOL_STATS.update(stats)
    return [_Source(image_id=image_id, sharpness=sharp, luminance=lum) for image_id, sharp, lum in final]


def _order_for(pool: list[_Source], ladder: str) -> list[_Source]:
    order = list(pool)
    random.Random(f"{SEED}:{ladder}").shuffle(order)
    return order


def _split_for(index: int) -> str:
    """calibration/test with every (split, level) combination equally represented (level = index % 4)."""
    return "calibration" if (index // 4) % 2 == 0 else "test"


def _seed_for(ladder: str, image_id: str, level: int) -> int:
    digest = hashlib.sha256(f"{SEED}:{ladder}:{image_id}:{level}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- build ----------------------------------------------------------------------------------------------


def build(cfg: Config, n_per_ladder: int = N_PER_LADDER, force: bool = False) -> dict[str, list[dict]]:
    raw_bytes = _load_raw_bytes(cfg)
    pool = _filtered_pool(raw_bytes)
    needed = N_ANCHOR_SOURCES + n_per_ladder
    if len(pool) < needed:
        raise RuntimeError(f"filtered source pool has only {len(pool)} sources, need {needed}")

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    anchors_all: dict[str, list[dict]] = {}
    results: dict[str, list[dict]] = {}

    for ladder in LADDERS:
        order = _order_for(pool, ladder)
        anchor_sources = order[:N_ANCHOR_SOURCES]
        item_sources = order[N_ANCHOR_SOURCES:needed]

        ladder_dir = IMAGES_DIR / ladder
        anchor_dir = ladder_dir / "anchors"
        anchor_dir.mkdir(parents=True, exist_ok=True)

        anchor_rows = []
        for j, src in enumerate(anchor_sources):
            base = None
            paths = []
            for level in range(4):
                out_path = anchor_dir / f"ref{j}_L{level}.jpg"
                if force or not out_path.exists():
                    base = base if base is not None else _get_base_image(raw_bytes, src.image_id)
                    img = degrade(base, ladder, level, _seed_for(ladder, src.image_id, level))
                    img.save(out_path, format="JPEG", quality=97)
                paths.append(_relpath(out_path))
            anchor_rows.append({"ref": j, "source_image_id": src.image_id, "paths": paths})
        anchors_all[ladder] = anchor_rows

        rows = []
        for i, src in enumerate(item_sources):
            level = i % 4
            split = _split_for(i)
            out_path = ladder_dir / f"{src.image_id}_L{level}.jpg"
            if force or not out_path.exists():
                base = _get_base_image(raw_bytes, src.image_id)
                img = degrade(base, ladder, level, _seed_for(ladder, src.image_id, level))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path, format="JPEG", quality=97)
            sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
            rows.append({
                "item_id": f"{src.image_id}_L{level}",
                "ladder": ladder,
                "source_image_id": src.image_id,
                "level": level,
                "split": split,
                "path": _relpath(out_path),
                "sha256": sha256,
                "sharpness": src.sharpness,
                "luminance": src.luminance,
            })
        _write_jsonl(MANIFESTS_DIR / f"{ladder}.jsonl", rows)
        results[ladder] = rows

    ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCHORS_PATH.write_text(json.dumps(anchors_all, indent=2) + "\n")
    return results


# --- contact sheets ---------------------------------------------------------------------------------------


def contact_sheet(cfg: Config, ladder: str, out_path: str | Path, n_sources: int = 3) -> None:
    """A grid: n_sources rows (the first n_sources ITEM sources of `ladder`) x 4 columns (levels 0..3),
    each cell the source scaled to 320 px wide, with a header row of level/param labels."""
    from PIL import ImageDraw, ImageFont

    params = LADDERS[ladder]["params"]
    raw_bytes = _load_raw_bytes(cfg)
    pool = _filtered_pool(raw_bytes)
    order = _order_for(pool, ladder)
    sources = order[N_ANCHOR_SOURCES : N_ANCHOR_SOURCES + n_sources]

    cell_w = 320
    pad = 8
    header_h = 28
    font = ImageFont.load_default()

    rows_of_thumbs: list[list[Image.Image]] = []
    row_heights: list[int] = []
    for src in sources:
        base = _get_base_image(raw_bytes, src.image_id)
        cell_h = round(cell_w * base.height / base.width)
        row = []
        for level in range(4):
            degraded = degrade(base, ladder, level, _seed_for(ladder, src.image_id, level))
            row.append(degraded.resize((cell_w, cell_h), Image.LANCZOS))
        rows_of_thumbs.append(row)
        row_heights.append(cell_h)

    width = pad * 5 + cell_w * 4
    height = header_h + pad + sum(row_heights) + pad * len(rows_of_thumbs)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col in range(4):
        x = pad + col * (cell_w + pad)
        draw.text((x, 6), f"level {col}  param={params[col]}", fill="black", font=font)

    y = header_h + pad
    for row, h in zip(rows_of_thumbs, row_heights):
        for col, thumb in enumerate(row):
            sheet.paste(thumb, (pad + col * (cell_w + pad), y))
        y += h + pad

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=90)


if __name__ == "__main__":
    from ..config import load_config

    cfg = load_config()
    results = build(cfg)

    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for ladder_name in LADDERS:
        contact_sheet(cfg, ladder_name, SHEETS_DIR / f"{ladder_name}.jpg")

    stats = LAST_POOL_STATS
    print(
        f"pool: candidates={stats.get('candidates')} after_min_size={stats.get('after_min_size')} "
        f"after_sharpness={stats.get('after_sharpness')} (p30={stats.get('sharpness_p30', 0):.1f}) "
        f"after_luminance={stats.get('after_luminance')}"
    )
    for ladder_name, rows in results.items():
        counts: dict[tuple[str, int], int] = {}
        for row in rows:
            key = (row["split"], row["level"])
            counts[key] = counts.get(key, 0) + 1
        print(f"{ladder_name}: n={len(rows)} counts={sorted(counts.items())}")

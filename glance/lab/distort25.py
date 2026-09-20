"""25 image-distortion types at 5 severity levels each, modelled on the design of the KADID-10k
database (25 distortions x 5 levels) but implemented entirely from scratch with PIL, numpy, and
scipy.ndimage -- no code or data from KADID or any other project.

Source images are drawn from the Oxford-IIIT Pet **train** split (`glance.lab.ladders` and the
`pets37` suite both only ever draw from the **test** split, so no train-split image has been used
anywhere in this project; the train/test parquet files come from the same pinned, CC BY-SA 4.0
Hub repo/revision as `glance.evals.suites.pets37.REPO`/`REVISION`). `_load_train_raw_bytes()` is a
small loader local to this module (ladders.py's own loader is hard-wired to the test parquet and is
never modified); it then reuses `ladders._filtered_pool()` unchanged, so the same three filters
apply (longest side >= 448, sharpness p30 cutoff -- recomputed on this pool, luminance >= 70). Every
train `image_id` is prefixed `train_` so it can never collide with a test-split id, and
`_legacy_used_source_ids()` (ids already used as an item or anchor by the 4-level ladders in
`lab/manifests/*.jsonl` / `lab/anchors.json`) is still applied as a harmless belt-and-suspenders
check even though it can never remove a `train_`-prefixed id. There is no "level -1" undistorted
image, as in KADID-10k: level 0 is already a mild, just-visible distortion and level 4 is severe.

Each distortion has 5 fixed parameter values, frozen in `DISTORTIONS` below together with a short
description used to build the rating question in `SCALES`. `build()` renders `N_PER_SCALE` items
per distortion (distinct sources, deterministic per-distortion shuffle), writes a manifest per
distortion under `lab/manifests_distort25/`. Generated images live under
`.cache/lab_images_distort25/` (gitignored).

Run `python -m glance.lab.distort25` to build everything and write one contact sheet per
distortion under `lab/sheets_distort25/`.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy import ndimage as ndi

from ..config import Config, PROJECT_ROOT
from . import ladders

BASE_SIDE = ladders.BASE_SIDE  # 448; reuse ladders' base-image convention so params mean the same thing
SEED = 7
N_PER_SCALE = 300
N_LEVELS = 5

LAB_DIR = PROJECT_ROOT / "lab"
IMAGES_DIR = PROJECT_ROOT / ".cache" / "lab_images_distort25"
MANIFESTS_DIR = LAB_DIR / "manifests_distort25"
SHEETS_DIR = LAB_DIR / "sheets_distort25"

# Distortions whose artifacts a second lossy JPEG re-encode would blur or hide; saved as PNG instead.
PNG_KEYS = {"jpeg", "jpeg2000", "color_quantization", "quantization", "pixelate", "impulse_noise"}

LEVELS = ["Barely noticeable", "Slight", "Moderate", "Strong", "Very strong"]
INSTRUCTIONS_TEMPLATE = "How strong is the {description} in `img0`?"

# Populated by _source_pool() (via build()/contact_sheet()) for diagnostics; not part of the public API.
LAST_POOL_STATS: dict[str, Any] = {}

# Every param list is ordered mildest -> most severe (level 0 .. level 4). For `mean_shift` and
# `contrast_change`, which are bidirectional in KADID-10k, we use a single direction each (mean_shift:
# brighter; contrast_change: lower contrast) so severity stays monotone, per the task spec.
DISTORTIONS: dict[str, dict[str, Any]] = {
    "gaussian_blur": {
        "description": "Gaussian blur (even softness everywhere)",
        "params": [0.6, 1.2, 2.2, 3.6, 5.5],  # Gaussian blur sigma in pixels at BASE_SIDE
    },
    "lens_blur": {
        "description": "lens blur (out-of-focus softness with disc-shaped highlights)",
        "params": [2, 3, 5, 8, 12],  # disc-kernel radius in pixels (larger radii show clear bokeh discs)
    },
    "motion_blur": {
        "description": "motion blur (streaking in one direction)",
        "params": [4, 7, 11, 16, 22],  # streak length in pixels; direction is drawn from the seed
    },
    "color_diffusion": {
        "description": "color diffusion (colors bleeding beyond their edges)",
        "params": [5, 9, 15, 23, 33],  # chroma dilation footprint in pixels (grows color beyond edges)
    },
    "color_shift": {
        "description": "color shift (color fringes displaced from the edges they belong to)",
        "params": [1, 2, 3, 5, 7],  # pixel shift of the R/B channels in opposite directions
    },
    "color_quantization": {
        "description": "color quantization (too few distinct colors, visible banding)",
        "params": [48, 24, 14, 8, 5],  # target palette size (PIL Image.quantize, no dithering)
    },
    "color_saturation_up": {
        "description": "oversaturation (colors too vivid)",
        "params": [1.4, 1.8, 2.4, 3.2, 4.2],  # PIL ImageEnhance.Color factor
    },
    "color_saturation_down": {
        "description": "desaturation (colors washed out toward gray)",
        "params": [0.75, 0.55, 0.38, 0.2, 0.0],  # PIL ImageEnhance.Color factor
    },
    "jpeg2000": {
        "description": "JPEG 2000 compression artifacts (smearing and ringing)",
        "params": [30, 55, 90, 150, 260],  # target compression ratio (Pillow JPEG2000 quality_layers)
    },
    "jpeg": {
        "description": "JPEG compression artifacts (square blocks and ringing)",
        "params": [30, 18, 11, 6, 3],  # JPEG quality
    },
    "white_noise": {
        "description": "white noise (fine random speckle)",
        "params": [5, 10, 17, 26, 38],  # additive Gaussian sigma; same noise value added to all 3 channels
    },
    "color_noise": {
        "description": "color noise (random colored speckle)",
        "params": [5, 10, 17, 26, 38],  # additive Gaussian sigma; independent noise per channel
    },
    "impulse_noise": {
        "description": "impulse noise (isolated black and white pixels)",
        "params": [0.004, 0.01, 0.02, 0.04, 0.07],  # fraction of pixels replaced by salt/pepper
    },
    "multiplicative_noise": {
        "description": "multiplicative noise (speckle that is stronger in bright areas)",
        "params": [0.06, 0.12, 0.2, 0.3, 0.42],  # relative-noise std; pixel *= 1 + N(0, sigma)
    },
    "denoise": {
        "description": "over-aggressive denoising (noise removed, leaving waxy, smeared detail)",
        "params": [3, 5, 7, 9, 13],  # median-filter window size in pixels (+ light smoothing)
    },
    "brighten": {
        "description": "overexposure (too bright, highlights washed out)",
        "params": [1.3, 1.6, 2.0, 2.6, 3.4],  # multiplicative gain, clipped at 255
    },
    "darken": {
        "description": "underexposure (too dark, shadows crushed)",
        "params": [0.72, 0.52, 0.36, 0.22, 0.1],  # multiplicative gain, clipped at 0
    },
    "mean_shift": {
        "description": "brightness shift (the whole image uniformly lighter or darker, contrast unchanged)",
        "params": [12, 25, 42, 62, 85],  # additive offset added to every pixel, clipped (brighter direction)
    },
    "jitter": {
        "description": "pixel jitter (small random displacement of pixels, ragged edges)",
        "params": [0.6, 1.1, 1.8, 2.8, 4.2],  # max random per-pixel displacement in pixels
    },
    "patch_shuffle": {
        "description": "non-eccentricity patches (small square patches copied to nearby wrong positions)",
        "params": [(16, 6), (16, 12), (16, 22), (16, 34), (16, 50)],  # (patch size px, patch count)
    },
    "pixelate": {
        "description": "pixelation (large square pixels)",
        "params": [3, 5, 8, 13, 20],  # block size in pixels (nearest-neighbour down/upsample factor)
    },
    "quantization": {
        "description": "luminance quantization (too few brightness levels, posterized look)",
        "params": [5, 4, 3, 2, 1],  # bits retained per channel (PIL ImageOps.posterize)
    },
    "color_block": {
        "description": "color blocks (random solid-colored squares pasted over the image)",
        "params": [(10, 1), (16, 3), (22, 6), (28, 10), (36, 16)],  # (block size px, block count)
    },
    "sharpen": {
        "description": "over-sharpening (halos and harsh edges)",
        "params": [0.8, 1.5, 2.4, 3.6, 5.2],  # unsharp-mask amount at a fixed 2px radius
    },
    "contrast_change": {
        "description": "contrast change (contrast too low or too high)",
        "params": [0.78, 0.58, 0.4, 0.24, 0.1],  # PIL ImageEnhance.Contrast factor (lower-contrast direction)
    },
}

SCALES: dict[str, dict[str, Any]] = {
    key: {"instructions": INSTRUCTIONS_TEMPLATE.format(description=spec["description"]), "levels": LEVELS}
    for key, spec in DISTORTIONS.items()
}


# --- pure degradation -------------------------------------------------------------------------------


def degrade(base: Image.Image, key: str, level: int, seed: int) -> Image.Image:
    """Apply distortion `key` at `level` (0..4) to `base`. Deterministic given its arguments; returns
    an RGB image the same size as `base`."""
    if base.mode != "RGB":
        base = base.convert("RGB")
    param = DISTORTIONS[key]["params"][level]
    fn = _DISPATCH.get(key)
    if fn is None:
        raise ValueError(f"unknown distortion {key!r}")
    return fn(base, param, seed)


def _clip_to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _apply_gaussian_blur(base: Image.Image, sigma: float, seed: int) -> Image.Image:
    return base.filter(ImageFilter.GaussianBlur(sigma))


def _disk_kernel(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x**2 + y**2) <= (r**2 + 0.5)
    kernel = mask.astype(np.float64)
    kernel /= kernel.sum()
    return kernel


def _apply_lens_blur(base: Image.Image, radius: float, seed: int) -> Image.Image:
    kernel = _disk_kernel(radius)
    arr = np.asarray(base, dtype=np.float64)
    out = np.empty_like(arr)
    for c in range(3):
        out[..., c] = ndi.convolve(arr[..., c], kernel, mode="reflect")
    return _clip_to_image(out)


def _line_kernel(length: float, angle_deg: float) -> np.ndarray:
    length = max(1, int(round(length)))
    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    theta = np.deg2rad(angle_deg)
    samples = max(length * 3, 7)
    for t in np.linspace(-length / 2, length / 2, num=samples):
        xi = int(round(center + t * np.cos(theta)))
        yi = int(round(center + t * np.sin(theta)))
        if 0 <= xi < size and 0 <= yi < size:
            kernel[yi, xi] = 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    kernel /= kernel.sum()
    return kernel


def _apply_motion_blur(base: Image.Image, length: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, 180.0))
    kernel = _line_kernel(length, angle)
    arr = np.asarray(base, dtype=np.float64)
    out = np.empty_like(arr)
    for c in range(3):
        out[..., c] = ndi.convolve(arr[..., c], kernel, mode="reflect")
    return _clip_to_image(out)


def _bleed_channel(channel: np.ndarray, size: int) -> np.ndarray:
    """Grow (dilate) each chroma channel's deviation from neutral outward by `size` pixels -- positive
    deviations via a max filter, negative ones via a min filter -- so a saturated color spreads into
    its neutral surroundings instead of just being diluted by it (a plain symmetric blur on a small
    saturated patch inside a huge neutral field barely moves the surrounding pixels at all). A light
    Gaussian smooth afterwards keeps the grown region from looking like flat filter tiles."""
    dev = channel - 128.0
    pos = ndi.maximum_filter(np.clip(dev, 0, None), size=size)
    neg = ndi.minimum_filter(np.clip(dev, None, 0), size=size)
    grown = 128.0 + pos + neg
    return ndi.gaussian_filter(grown, sigma=max(0.5, size / 4.0))


def _apply_color_diffusion(base: Image.Image, size: float, seed: int) -> Image.Image:
    ycbcr = np.asarray(base.convert("YCbCr"), dtype=np.float64)
    y = ycbcr[..., 0]
    cb = _bleed_channel(ycbcr[..., 1], int(round(size)))
    cr = _bleed_channel(ycbcr[..., 2], int(round(size)))
    out = np.stack([y, cb, cr], axis=-1)
    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="YCbCr")
    return out_img.convert("RGB")


def _apply_color_shift(base: Image.Image, shift: float, seed: int) -> Image.Image:
    arr = np.asarray(base, dtype=np.float64)
    out = arr.copy()
    out[..., 0] = ndi.shift(arr[..., 0], shift=(0, shift), order=1, mode="reflect")
    out[..., 2] = ndi.shift(arr[..., 2], shift=(0, -shift), order=1, mode="reflect")
    return _clip_to_image(out)


def _apply_color_quantization(base: Image.Image, colors: int, seed: int) -> Image.Image:
    colors = max(2, int(colors))
    quantized = base.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return quantized.convert("RGB")


def _apply_color_saturation(base: Image.Image, factor: float, seed: int) -> Image.Image:
    return ImageEnhance.Color(base).enhance(factor)


def _apply_jpeg2000(base: Image.Image, ratio: float, seed: int) -> Image.Image:
    buf = io.BytesIO()
    base.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[float(ratio)])
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_jpeg(base: Image.Image, quality: int, seed: int) -> Image.Image:
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _apply_white_noise(base: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(base, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=arr.shape[:2])[..., None]
    return _clip_to_image(arr + noise)


def _apply_color_noise(base: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(base, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=arr.shape)
    return _clip_to_image(arr + noise)


def _apply_impulse_noise(base: Image.Image, prob: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.array(base, dtype=np.uint8)
    h, w = arr.shape[:2]
    roll = rng.random((h, w))
    salt = roll < (prob / 2.0)
    pepper = (roll >= (prob / 2.0)) & (roll < prob)
    arr[salt] = 255
    arr[pepper] = 0
    return Image.fromarray(arr, mode="RGB")


def _apply_multiplicative_noise(base: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(base, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=arr.shape)
    return _clip_to_image(arr * (1.0 + noise))


def _apply_denoise(base: Image.Image, size: int, seed: int) -> Image.Image:
    size = int(size)
    img = base.filter(ImageFilter.MedianFilter(size=size))
    extra_sigma = max(0.0, (size - 3) / 6.0)
    if extra_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(extra_sigma))
    return img


def _apply_brighten(base: Image.Image, gain: float, seed: int) -> Image.Image:
    arr = np.asarray(base, dtype=np.float64)
    return _clip_to_image(arr * gain)


def _apply_darken(base: Image.Image, gain: float, seed: int) -> Image.Image:
    arr = np.asarray(base, dtype=np.float64)
    return _clip_to_image(arr * gain)


def _apply_mean_shift(base: Image.Image, offset: float, seed: int) -> Image.Image:
    arr = np.asarray(base, dtype=np.float64)
    return _clip_to_image(arr + offset)


def _apply_jitter(base: Image.Image, max_disp: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(base, dtype=np.float64)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = rng.uniform(-max_disp, max_disp, size=(h, w))
    dy = rng.uniform(-max_disp, max_disp, size=(h, w))
    coords_y = np.clip(yy + dy, 0, h - 1)
    coords_x = np.clip(xx + dx, 0, w - 1)
    out = np.empty_like(arr)
    for c in range(3):
        out[..., c] = ndi.map_coordinates(arr[..., c], [coords_y, coords_x], order=1, mode="reflect")
    return _clip_to_image(out)


def _apply_patch_shuffle(base: Image.Image, param: tuple[int, int], seed: int) -> Image.Image:
    patch_size, count = param
    rng = np.random.default_rng(seed)
    arr = np.array(base, dtype=np.uint8)
    h, w = arr.shape[:2]
    p = max(1, min(int(patch_size), h, w))
    reach = p * 2
    for _ in range(int(count)):
        sy = int(rng.integers(0, h - p + 1))
        sx = int(rng.integers(0, w - p + 1))
        dy = int(np.clip(sy + rng.integers(-reach, reach + 1), 0, h - p))
        dx = int(np.clip(sx + rng.integers(-reach, reach + 1), 0, w - p))
        arr[dy : dy + p, dx : dx + p] = arr[sy : sy + p, sx : sx + p]
    return Image.fromarray(arr, mode="RGB")


def _apply_pixelate(base: Image.Image, block: int, seed: int) -> Image.Image:
    block = max(1, int(block))
    w, h = base.size
    small = base.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def _apply_quantization(base: Image.Image, bits: int, seed: int) -> Image.Image:
    return ImageOps.posterize(base, int(bits))


def _apply_color_block(base: Image.Image, param: tuple[int, int], seed: int) -> Image.Image:
    block_size, count = param
    rng = np.random.default_rng(seed)
    img = base.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(int(count)):
        bx = int(rng.integers(0, max(1, w - block_size)))
        by = int(rng.integers(0, max(1, h - block_size)))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        draw.rectangle([bx, by, bx + block_size, by + block_size], fill=color)
    return img


def _apply_sharpen(base: Image.Image, amount: float, seed: int) -> Image.Image:
    radius = 2.0
    arr = np.asarray(base, dtype=np.float64)
    blurred = np.asarray(base.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64)
    return _clip_to_image(arr + amount * (arr - blurred))


def _apply_contrast_change(base: Image.Image, factor: float, seed: int) -> Image.Image:
    return ImageEnhance.Contrast(base).enhance(factor)


_DISPATCH = {
    "gaussian_blur": _apply_gaussian_blur,
    "lens_blur": _apply_lens_blur,
    "motion_blur": _apply_motion_blur,
    "color_diffusion": _apply_color_diffusion,
    "color_shift": _apply_color_shift,
    "color_quantization": _apply_color_quantization,
    "color_saturation_up": _apply_color_saturation,
    "color_saturation_down": _apply_color_saturation,
    "jpeg2000": _apply_jpeg2000,
    "jpeg": _apply_jpeg,
    "white_noise": _apply_white_noise,
    "color_noise": _apply_color_noise,
    "impulse_noise": _apply_impulse_noise,
    "multiplicative_noise": _apply_multiplicative_noise,
    "denoise": _apply_denoise,
    "brighten": _apply_brighten,
    "darken": _apply_darken,
    "mean_shift": _apply_mean_shift,
    "jitter": _apply_jitter,
    "patch_shuffle": _apply_patch_shuffle,
    "pixelate": _apply_pixelate,
    "quantization": _apply_quantization,
    "color_block": _apply_color_block,
    "sharpen": _apply_sharpen,
    "contrast_change": _apply_contrast_change,
}

assert set(_DISPATCH) == set(DISTORTIONS)


# --- source pool ------------------------------------------------------------------------------------

TRAIN_ID_PREFIX = "train_"


def _load_train_raw_bytes(cfg: Config) -> dict[str, bytes]:
    """`train_<image_id>` -> original encoded bytes, for every row of the pets37 Hub repo's **train**
    split. Local to this module (not shared with `ladders._load_raw_bytes`, which is hard-wired to the
    test parquet): `ladders.py` and the `pets37` suite both only ever read the test split, so no
    exclusion is needed here beyond the harmless belt-and-suspenders check in `_source_pool`. The
    `train_` prefix keeps every id unambiguous from any (unprefixed) test-split id."""
    import pyarrow.parquet as pq

    from ..evals.suites import pets37
    from ..evals.suites.sources import hf_parquet

    path = hf_parquet(cfg, pets37.REPO, pets37.REVISION, "data/train-00000-of-00001.parquet")
    table = pq.read_table(path, columns=["image_id", "image"])
    return {f"{TRAIN_ID_PREFIX}{row['image_id']}": row["image"]["bytes"] for row in table.to_pylist()}


def _legacy_used_source_ids() -> set[str]:
    """`source_image_id`s already used as an item or anchor by the 4-level lab ladders, so distort25
    shares no photo with them. Kept as a belt-and-suspenders check even though the train-split pool's
    `train_`-prefixed ids can never appear here (those ladders only ever drew from the test split)."""
    used: set[str] = set()
    for path in sorted(ladders.MANIFESTS_DIR.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                used.add(json.loads(line)["source_image_id"])
    if ladders.ANCHORS_PATH.exists():
        anchors = json.loads(ladders.ANCHORS_PATH.read_text())
        for rows in anchors.values():
            for row in rows:
                used.add(row["source_image_id"])
    return used


# In-process cache so build() and 25 back-to-back contact_sheet() calls in __main__ don't each redecode
# and re-filter ~3700 source images; keyed on `id(cfg)` since Config isn't guaranteed hashable.
_POOL_CACHE: dict[int, tuple[dict[str, bytes], list[ladders._Source]]] = {}


def _source_pool(cfg: Config) -> tuple[dict[str, bytes], list[ladders._Source]]:
    """The pets37 train split, filtered with `ladders._filtered_pool()` unchanged (same three filters,
    p30 sharpness threshold recomputed on this pool), minus every source already used by the earlier
    (test-split) lab scales. Updates `LAST_POOL_STATS` as a side effect. Cached per `cfg` instance
    within the process (see `_POOL_CACHE`)."""
    cached = _POOL_CACHE.get(id(cfg))
    if cached is not None:
        LAST_POOL_STATS.clear()
        LAST_POOL_STATS.update(cached[2])
        return cached[0], cached[1]

    raw_bytes = _load_train_raw_bytes(cfg)
    full_pool = ladders._filtered_pool(raw_bytes)
    excluded = _legacy_used_source_ids()
    pool = [s for s in full_pool if s.image_id not in excluded]
    stats = dict(ladders.LAST_POOL_STATS)
    stats["legacy_excluded"] = len(excluded)
    stats["after_legacy_exclusion"] = len(pool)
    LAST_POOL_STATS.clear()
    LAST_POOL_STATS.update(stats)
    _POOL_CACHE[id(cfg)] = (raw_bytes, pool, dict(stats))
    return raw_bytes, pool


def _order_for(pool: list[ladders._Source], key: str) -> list[ladders._Source]:
    order = list(pool)
    random.Random(f"{SEED}:distort25:{key}").shuffle(order)
    return order


def _split_for(index: int) -> str:
    """calibration/test with every (split, level) combination equally represented (level = index % 5)."""
    return "calibration" if (index // N_LEVELS) % 2 == 0 else "test"


def _seed_for(key: str, image_id: str, level: int) -> int:
    digest = hashlib.sha256(f"{SEED}:distort25:{key}:{image_id}:{level}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- build ------------------------------------------------------------------------------------------


def build(cfg: Config, n_per_scale: int = N_PER_SCALE, force: bool = False) -> dict[str, list[dict]]:
    raw_bytes, pool = _source_pool(cfg)
    if len(pool) < n_per_scale:
        raise RuntimeError(
            f"filtered source pool (after excluding earlier lab scales) has only {len(pool)} sources, "
            f"need {n_per_scale}"
        )

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}

    for key in DISTORTIONS:
        order = _order_for(pool, key)
        item_sources = order[:n_per_scale]
        key_dir = IMAGES_DIR / key
        ext = "png" if key in PNG_KEYS else "jpg"

        rows = []
        for i, src in enumerate(item_sources):
            level = i % N_LEVELS
            split = _split_for(i)
            out_path = key_dir / f"{src.image_id}_L{level}.{ext}"
            if force or not out_path.exists():
                base = ladders._get_base_image(raw_bytes, src.image_id)
                img = degrade(base, key, level, _seed_for(key, src.image_id, level))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if ext == "png":
                    img.save(out_path, format="PNG")
                else:
                    img.save(out_path, format="JPEG", quality=97)
            sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
            rows.append({
                "item_id": f"{src.image_id}_L{level}",
                "ladder": key,
                "source_image_id": src.image_id,
                "level": level,
                "split": split,
                "path": _relpath(out_path),
                "sha256": sha256,
            })
        _write_jsonl(MANIFESTS_DIR / f"{key}.jsonl", rows)
        results[key] = rows

    return results


# --- contact sheets -----------------------------------------------------------------------------------


def contact_sheet(cfg: Config, key: str, out_path: str | Path, n_sources: int = 3) -> None:
    """A grid: n_sources rows x 5 level columns (levels 0..4), each cell 300 px wide, with a header
    row of level/param labels. Independent of build(): recomputes the pool and shuffle itself, so it
    only needs `n_sources` sources to exist, not the full `N_PER_SCALE`."""
    params = DISTORTIONS[key]["params"]
    raw_bytes, pool = _source_pool(cfg)
    order = _order_for(pool, key)
    sources = order[:n_sources]

    cell_w = 300
    pad = 8
    header_h = 28
    font = ImageFont.load_default()

    rows_of_thumbs: list[list[Image.Image]] = []
    row_heights: list[int] = []
    for src in sources:
        base = ladders._get_base_image(raw_bytes, src.image_id)
        cell_h = round(cell_w * base.height / base.width)
        row = []
        for level in range(N_LEVELS):
            degraded = degrade(base, key, level, _seed_for(key, src.image_id, level))
            row.append(degraded.resize((cell_w, cell_h), Image.LANCZOS))
        rows_of_thumbs.append(row)
        row_heights.append(cell_h)

    width = pad * (N_LEVELS + 1) + cell_w * N_LEVELS
    height = header_h + pad + sum(row_heights) + pad * len(rows_of_thumbs)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col in range(N_LEVELS):
        x = pad + col * (cell_w + pad)
        draw.text((x, 6), f"level {col}  param={params[col]}", fill="black", font=font)

    y = header_h + pad
    for row, h in zip(rows_of_thumbs, row_heights):
        for col, thumb in enumerate(row):
            sheet.paste(thumb, (pad + col * (cell_w + pad), y))
        y += h + pad

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=92)


if __name__ == "__main__":
    from ..config import load_config

    cfg = load_config()
    results = build(cfg)

    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for key in DISTORTIONS:
        contact_sheet(cfg, key, SHEETS_DIR / f"{key}.jpg")

    stats = LAST_POOL_STATS
    print(
        f"pool: candidates={stats.get('candidates')} after_min_size={stats.get('after_min_size')} "
        f"after_sharpness={stats.get('after_sharpness')} (p30={stats.get('sharpness_p30', 0):.1f}) "
        f"after_luminance={stats.get('after_luminance')} legacy_excluded={stats.get('legacy_excluded')} "
        f"after_legacy_exclusion={stats.get('after_legacy_exclusion')}"
    )
    for key, rows in results.items():
        counts: dict[tuple[str, int], int] = {}
        for row in rows:
            k = (row["split"], row["level"])
            counts[k] = counts.get(k, 0) + 1
        print(f"{key}: n={len(rows)} counts={sorted(counts.items())}")

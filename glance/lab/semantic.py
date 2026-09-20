"""Seven "creative QA" rating scales that have nothing to do with image-quality distortions (blur,
noise, compression, ...): framing, cutoff, off-centeredness, occlusion, tilt, text legibility and a
watermark overlay. These test whether Glance elicitation generalizes beyond blur/noise-style ladders
to scales with an exact, geometric ground truth.

Source photos are the same Oxford-IIIT Pet **train**-split pool `glance.lab.distort25` already builds
and filters (`distort25._source_pool()`, imported unchanged -- not modified here). On top of that pool
we require the official Oxford-IIIT Pet segmentation trimap for each photo (NOT present in the Hub
parquet), downloaded once from the dataset's own page (`https://www.robots.ox.ac.uk/~vgg/data/pets/`,
`annotations.tar.gz`, CC BY-SA 4.0 -- the same license as the Hub mirror `distort25`/`pets37` use) into
`.cache/datasets/oxford_pets_annotations/`. The tarball's sha256 is pinned in `ANNOTATIONS_SHA256`
below and checked before anything is extracted. Trimap pixel values are 1=pet, 2=background,
3=border/ambiguous; 1 and 3 both count as "animal" here. Trimaps are matched to photos by file stem
(the parquet's `image_id`, stripped of distort25's `train_` prefix); a photo without a matching trimap,
or whose trimap size doesn't match the photo, is skipped.

Four scales (subject_size, cutoff, off_center, occlusion) have an exact numeric target per level (a
fraction of frame or of animal pixels) that must be hit within a stated tolerance by searching a crop
or a grey rectangle against the photo's mask; sources for which no crop/rectangle hits a given level's
target are skipped in favor of another source, but the *tolerance itself* is never relaxed -- if a
level cannot find `n_per_scale // 5` feasible sources, `build()` raises. The other three (tilt,
text_legibility, watermark) apply a fixed, deterministic transform per level with no per-source search.

Generated images live under `.cache/lab_images_semantic/` (gitignored, PNG for text_legibility/
watermark, JPEG quality 97 otherwise, matching distort25). Manifests are committed under
`lab/manifests_semantic/`, sheets under `lab/sheets_semantic/`.

Run `python -m glance.lab.semantic` to download the trimaps (if needed), build everything, and write
one contact sheet per scale.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import Config, PROJECT_ROOT
from . import distort25, ladders

BASE_SIDE = ladders.BASE_SIDE  # 448
SEED = 7
N_PER_SCALE = 600
N_LEVELS = 5

LAB_DIR = PROJECT_ROOT / "lab"
IMAGES_DIR = PROJECT_ROOT / ".cache" / "lab_images_semantic"
MANIFESTS_DIR = LAB_DIR / "manifests_semantic"
SHEETS_DIR = LAB_DIR / "sheets_semantic"

# --- Oxford-IIIT Pet trimaps (not in the Hub parquet; official dataset page, CC BY-SA 4.0) -----------

ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
ANNOTATIONS_DIR = PROJECT_ROOT / ".cache" / "datasets" / "oxford_pets_annotations"
ANNOTATIONS_TARBALL = ANNOTATIONS_DIR / "annotations.tar.gz"
# Verified 2026-09-20 against https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz (~19 MB).
# build()/`_ensure_trimaps()` refuse to use a tarball whose sha256 doesn't match this.
ANNOTATIONS_SHA256 = "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"
TRIMAPS_DIR = ANNOTATIONS_DIR / "annotations" / "trimaps"

PNG_KEYS = {"text_legibility", "watermark"}
FIT_KEYS = {"subject_size", "cutoff", "off_center", "occlusion"}

# Populated by _semantic_pool() for diagnostics; not part of the public API.
LAST_POOL_STATS: dict[str, Any] = {}


class Infeasible(RuntimeError):
    """Raised by `render()` when a source's mask cannot hit a level's target within tolerance."""


# --- rating scales (exact texts; registered in advance, do not reword) -------------------------------

SCALES: dict[str, dict[str, Any]] = {
    "subject_size": {
        "instructions": "How much of the frame does the animal in `img0` fill?",
        "levels": [
            "The animal is tiny in the frame",
            "The animal is small in the frame",
            "The animal fills a moderate part of the frame",
            "The animal fills most of the frame",
            "The animal fills almost the entire frame",
        ],
    },
    "cutoff": {
        "instructions": "How much of the animal in `img0` is cut off by the edge of the frame?",
        "levels": [
            "The whole animal is inside the frame",
            "A small part of the animal is cut off",
            "A noticeable part of the animal is cut off",
            "About half of the animal is cut off",
            "Most of the animal is cut off",
        ],
    },
    "off_center": {
        "instructions": "How far from the centre of the frame is the animal in `img0`?",
        "levels": [
            "The animal is centred",
            "The animal is slightly off-centre",
            "The animal is clearly off-centre",
            "The animal is far off-centre",
            "The animal is at the very edge of the frame",
        ],
    },
    "occlusion": {
        "instructions": "How much of the animal in `img0` is hidden behind the grey rectangle?",
        "levels": [
            "None of the animal is hidden",
            "A small part of the animal is hidden",
            "A noticeable part of the animal is hidden",
            "About half of the animal is hidden",
            "Most of the animal is hidden",
        ],
    },
    "tilt": {
        "instructions": "How tilted is the photo `img0`, compared with a level camera?",
        "levels": ["Level", "Slightly tilted", "Noticeably tilted", "Strongly tilted", "Extremely tilted"],
    },
    "text_legibility": {
        "instructions": "How easy is it to read the caption printed on `img0`?",
        "levels": [
            "The caption cannot be read",
            "The caption is very hard to read",
            "The caption is readable with effort",
            "The caption is easy to read",
            "The caption is very easy to read",
        ],
    },
    "watermark": {
        "instructions": "How intrusive is the repeated SAMPLE watermark on `img0`?",
        "levels": [
            "No watermark is visible",
            "The watermark is barely visible",
            "The watermark is clearly visible",
            "The watermark is distracting",
            "The watermark dominates the image",
        ],
    },
}

assert set(SCALES) == PNG_KEYS | FIT_KEYS | {"tilt"}
for _key, _spec in SCALES.items():
    assert set(_spec) == {"instructions", "levels"}, _key
    assert len(_spec["levels"]) == N_LEVELS, _key

# Numeric ground-truth target per level (level 0 = lowest), separate from SCALES so SCALES keeps the
# same {"instructions", "levels"} shape as distort25.
SCALE_PARAMS: dict[str, list[Any]] = {
    "subject_size": [0.06, 0.15, 0.30, 0.50, 0.75],  # fraction of frame pixels that are animal
    "cutoff": [0.0, 0.10, 0.25, 0.45, 0.65],  # fraction of animal pixels outside the frame
    "off_center": [0.0, 0.2, 0.4, 0.6, 0.8],  # |centroid - frame centre| / half-width, horizontal only
    "occlusion": [0.0, 0.10, 0.25, 0.45, 0.65],  # fraction of animal pixels under the grey rectangle
    "tilt": [0, 4, 9, 17, 30],  # rotation magnitude in degrees; sign is seeded per item
    "watermark": [0.0, 0.06, 0.15, 0.30, 0.55],  # alpha of the tiled SAMPLE watermark layer
}

# Tolerance for the 4 search-based (FIT_KEYS) scales: ("relative"|"absolute", amount).
TOLERANCES: dict[str, tuple[str, float]] = {
    "subject_size": ("relative", 0.15),
    "cutoff": ("absolute", 0.04),
    "off_center": ("absolute", 0.04),
    "occlusion": ("absolute", 0.03),
}

# 20 fixed neutral 5-7 word captions for text_legibility (no brand names, nothing offensive).
# Kept short enough (each word's own render width matters more than the count) that a 2-line wrap at
# the LARGEST registered font size (see `_caption_lines`) always fits within `_TEXT_MAX_WIDTH_FRACTION`
# of a BASE_SIDE-wide frame; the original longer captions (e.g. "The lighthouse guided ships through
# foggy nights.") could not be wrapped to fit at font size 36 no matter where the line broke.
CAPTIONS = [
    "The cat sat on the red mat.",
    "Fresh bread is baked at dawn.",
    "A river runs by the old farm.",
    "She grows herbs in a small pot.",
    "The shop closes early on Sundays.",
    "Waves crash on the rocky shore.",
    "He fixed the bike before noon.",
    "Leaves fall on the quiet path.",
    "The baker sells warm rolls at noon.",
    "Kids play near the old stone bridge.",
    "A soft wind moved the tall grass.",
    "The show has art from local kids.",
    "Rain fell softly all through the night.",
    "They picked apples before the frost.",
    "Kids read books in the sunny yard.",
    "The old clock struck twelve at noon.",
    "Birds built a nest in the oak.",
    "The stalls open early on market day.",
    "Snow fell on the hills at dusk.",
    "A boat drifts on the calm lake.",
]
assert len(CAPTIONS) == 20
assert len(set(CAPTIONS)) == 20
for _c in CAPTIONS:
    assert 5 <= len(_c.split()) <= 7, _c

# text_legibility level params (level 0 = worst): font size in px at BASE_SIDE, alpha in [0,1], Gaussian
# blur radius applied to the text layer only, how the fill color is chosen, and outline width in px.
TEXT_LEVEL_PARAMS: list[dict[str, Any]] = [
    {"font_size": 14, "opacity": 0.18, "blur": 2.6, "color_mode": "background", "outline": 0},
    {"font_size": 18, "opacity": 0.38, "blur": 1.3, "color_mode": "blend35", "outline": 0},
    {"font_size": 23, "opacity": 0.62, "blur": 0.5, "color_mode": "blend70", "outline": 1},
    {"font_size": 29, "opacity": 0.88, "blur": 0.0, "color_mode": "white", "outline": 2},
    {"font_size": 36, "opacity": 1.0, "blur": 0.0, "color_mode": "white", "outline": 3},
]
assert len(TEXT_LEVEL_PARAMS) == N_LEVELS

WATERMARK_TEXT = "SAMPLE"
WATERMARK_FONT_SIZE = 34
WATERMARK_SPACING = 140  # px between tile repeats, fixed across every level and item
WATERMARK_ANGLE = -30


def _font(size: int) -> ImageFont.FreeTypeFont:
    import matplotlib.font_manager as fm

    return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)


# Max caption width as a fraction of the frame width, checked at the LARGEST registered font size
# (level 4's) so the line-break decision is the same at every level -- never itself a legibility cue --
# and so a caption that fits at the biggest, boldest rendering fits (with room to spare) at every
# smaller, fainter one too.
_TEXT_MAX_WIDTH_FRACTION = 0.92
_TEXT_LINE_GAP_FRACTION = 0.18  # extra vertical gap between wrapped lines, as a fraction of font size

_CAPTION_LINES_CACHE: dict[str, list[str]] = {}


def _caption_lines(caption: str, image_width: int) -> list[str]:
    """1 or 2 centred lines for `caption` that fit within `_TEXT_MAX_WIDTH_FRACTION * image_width` at
    the largest registered font size. The decision (and, if wrapping, exactly where the line breaks) is
    a pure function of the caption text and `image_width` -- not of `level` -- so every level of a given
    item uses the identical layout; only font size/opacity/blur/outline (the registered per-level
    params) change."""
    key = f"{caption}|{image_width}"
    cached = _CAPTION_LINES_CACHE.get(key)
    if cached is not None:
        return cached

    font = _font(TEXT_LEVEL_PARAMS[-1]["font_size"])
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    limit = _TEXT_MAX_WIDTH_FRACTION * image_width

    def width_of(s: str) -> float:
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    if width_of(caption) <= limit:
        lines = [caption]
    else:
        words = caption.split()
        best = None
        for cut in range(1, len(words)):
            line1, line2 = " ".join(words[:cut]), " ".join(words[cut:])
            score = max(width_of(line1), width_of(line2))
            if best is None or score < best[0]:
                best = (score, line1, line2)
        if best is None or best[0] > limit:
            raise Infeasible(f"text_legibility: caption {caption!r} does not fit in {image_width}px even wrapped")
        lines = [best[1], best[2]]

    _CAPTION_LINES_CACHE[key] = lines
    return lines


# --- trimap download --------------------------------------------------------------------------------


def _ensure_trimaps() -> Path:
    """Downloads and verifies `annotations.tar.gz` once, then extracts only the `trimaps/` subfolder.
    Refuses (raises) to use a tarball whose sha256 doesn't match `ANNOTATIONS_SHA256`."""
    if TRIMAPS_DIR.exists() and any(TRIMAPS_DIR.iterdir()):
        return TRIMAPS_DIR

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not ANNOTATIONS_TARBALL.exists():
        request = urllib.request.Request(ANNOTATIONS_URL, headers={"User-Agent": "glance-lab/0.1"})
        tmp = ANNOTATIONS_TARBALL.with_suffix(ANNOTATIONS_TARBALL.suffix + ".part")
        with urllib.request.urlopen(request, timeout=120) as resp:  # noqa: S310 - fixed https dataset host
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.rename(ANNOTATIONS_TARBALL)

    digest = hashlib.sha256(ANNOTATIONS_TARBALL.read_bytes()).hexdigest()
    if digest != ANNOTATIONS_SHA256:
        raise RuntimeError(
            f"{ANNOTATIONS_TARBALL} sha256 mismatch: got {digest}, expected {ANNOTATIONS_SHA256}; refusing to use it"
        )
    with tarfile.open(ANNOTATIONS_TARBALL) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("annotations/trimaps/")]
        tar.extractall(ANNOTATIONS_DIR, members=members, filter="data")
    return TRIMAPS_DIR


def _trimap_stem(image_id: str) -> str:
    prefix = distort25.TRAIN_ID_PREFIX
    return image_id[len(prefix):] if image_id.startswith(prefix) else image_id


def _trimap_path(image_id: str) -> Path:
    return TRIMAPS_DIR / f"{_trimap_stem(image_id)}.png"


def _load_mask_only(image_id: str) -> np.ndarray:
    """uint8 (H, W) array, 1 where the trimap says animal (values 1 or 3), 0 for background (2)."""
    trimap = np.array(Image.open(_trimap_path(image_id)))
    return (trimap != 2).astype(np.uint8)


def _open_source(raw_bytes: dict[str, bytes], image_id: str) -> tuple[Image.Image, np.ndarray]:
    photo = Image.open(io.BytesIO(raw_bytes[image_id])).convert("RGB")
    return photo, _load_mask_only(image_id)


# --- source pool (distort25's train pool, restricted to photos with a matching trimap) ---------------

_POOL_CACHE: dict[int, tuple[dict[str, bytes], list[ladders._Source], dict[str, Any]]] = {}


def _semantic_pool(cfg: Config) -> tuple[dict[str, bytes], list[ladders._Source]]:
    cached = _POOL_CACHE.get(id(cfg))
    if cached is not None:
        LAST_POOL_STATS.clear()
        LAST_POOL_STATS.update(cached[2])
        return cached[0], cached[1]

    raw_bytes, base_pool = distort25._source_pool(cfg)
    trimaps_dir = _ensure_trimaps()
    kept: list[ladders._Source] = []
    no_trimap = 0
    size_mismatch = 0
    for src in base_pool:
        tpath = trimaps_dir / f"{_trimap_stem(src.image_id)}.png"
        if not tpath.exists():
            no_trimap += 1
            continue
        photo = Image.open(io.BytesIO(raw_bytes[src.image_id]))
        trimap = Image.open(tpath)
        if trimap.size != photo.size:
            size_mismatch += 1
            continue
        kept.append(src)

    stats = dict(distort25.LAST_POOL_STATS)
    stats.update({
        "semantic_candidates": len(base_pool), "skipped_no_trimap": no_trimap,
        "skipped_size_mismatch": size_mismatch, "semantic_pool": len(kept),
    })
    LAST_POOL_STATS.clear()
    LAST_POOL_STATS.update(stats)
    _POOL_CACHE[id(cfg)] = (raw_bytes, kept, dict(stats))
    return raw_bytes, kept


def _order_for(pool: list[ladders._Source], key: str) -> list[ladders._Source]:
    order = list(pool)
    random.Random(f"{SEED}:semantic:{key}").shuffle(order)
    return order


def _split_for(index: int) -> str:
    """calibration/test with every (split, level) combination equally represented (level = index % 5)."""
    return "calibration" if (index // N_LEVELS) % 2 == 0 else "test"


def _seed_for(key: str, image_id: str, level: int) -> int:
    digest = hashlib.sha256(f"{SEED}:semantic:{key}:{image_id}:{level}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- geometry helpers --------------------------------------------------------------------------------


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1  # y0, y1, x0, x1


def _mask_to_base(mask: np.ndarray) -> np.ndarray:
    """Resize `mask` the same way `ladders._to_base` resizes the matching photo (longest side ->
    BASE_SIDE, never upscaled), so the two stay pixel-aligned."""
    h, w = mask.shape
    scale = BASE_SIDE / max(h, w)
    if scale >= 1.0:
        return mask
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((new_w, new_h), Image.NEAREST)
    return (np.asarray(resized) > 127).astype(np.uint8)


def _tol(key: str, target: float) -> float:
    mode, amount = TOLERANCES[key]
    return amount * target if mode == "relative" else amount


# --- subject_size: square crop centred on the mask centroid, sized to hit a frame-fill fraction ------


def _subject_size_fraction(mask: np.ndarray, cy: float, cx: float, s: float) -> tuple[float, tuple]:
    h, w = mask.shape
    s = min(s, min(h, w))
    y0 = min(max(cy - s / 2.0, 0.0), h - s)
    x0 = min(max(cx - s / 2.0, 0.0), w - s)
    y0, x0 = max(0.0, y0), max(0.0, x0)
    y1, x1 = y0 + s, x0 + s
    iy0, iy1, ix0, ix1 = int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))
    iy1, ix1 = max(iy1, iy0 + 1), max(ix1, ix0 + 1)
    sub = mask[iy0:iy1, ix0:ix1]
    area = sub.shape[0] * sub.shape[1]
    fraction = float(sub.sum()) / area if area else 0.0
    return fraction, (y0, y1, x0, x1)


def _fit_subject_size_box(mask: np.ndarray, target: float, seed: int) -> tuple[tuple, float] | None:
    h, w = mask.shape
    if mask.sum() == 0:
        return None
    cy, cx = _centroid(mask)
    tol = _tol("subject_size", target)

    f_small, _ = _subject_size_fraction(mask, cy, cx, 1.0)
    f_large, _ = _subject_size_fraction(mask, cy, cx, float(min(h, w)))
    if target > f_small + tol or target < f_large - tol:
        return None

    lo, hi = 1.0, float(min(h, w))
    for _ in range(40):
        mid = (lo + hi) / 2.0
        f_mid, _ = _subject_size_fraction(mask, cy, cx, mid)
        if f_mid > target:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2.0
    achieved, box = _subject_size_fraction(mask, cy, cx, s)
    if abs(achieved - target) > tol:
        return None
    if BASE_SIDE / s > 2.0:  # never upscale a crop by more than 2x
        return None
    return box, achieved


# --- cutoff: crop containing the whole animal with margin, slid in a seeded direction -----------------


def _cutoff_fraction(mask: np.ndarray, cy0: float, cx0: float, win_h: float, win_w: float,
                      direction: str, shift: float) -> tuple[float, tuple]:
    h, w = mask.shape
    a = float(mask.sum())
    cx, cy = cx0, cy0
    if direction == "right":
        cx += shift
    elif direction == "left":
        cx -= shift
    elif direction == "down":
        cy += shift
    else:
        cy -= shift
    y0, y1 = cy - win_h / 2.0, cy + win_h / 2.0
    x0, x1 = cx - win_w / 2.0, cx + win_w / 2.0
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if y1 > h:
        y0 -= y1 - h
        y1 = float(h)
    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if x1 > w:
        x0 -= x1 - w
        x1 = float(w)
    y0, x0 = max(0.0, y0), max(0.0, x0)
    iy0, iy1, ix0, ix1 = int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))
    inside = float(mask[iy0:iy1, ix0:ix1].sum())
    return (1.0 - inside / a if a else 0.0), (y0, y1, x0, x1)


#: Fraction of the image's available slack (image_dim - bbox_dim) spent as level-0 margin. A margin
#: defined as a multiple of the *bbox* size (an earlier version of this function used bbox*1.5) saturates
#: to the full image dimension whenever the bbox already fills most of the frame -- common in this
#: dataset's close-up pet photos -- which leaves *zero* room to slide the window and silently makes every
#: positive cutoff target unreachable. Defining the margin as a fraction of the actual leftover slack
#: guarantees most of that slack stays available for sliding, whatever the bbox's size relative to the
#: photo.
_CUTOFF_MARGIN_FRACTION = 0.03


def _fit_cutoff_box(mask: np.ndarray, target: float, seed: int) -> tuple[tuple, float] | None:
    h, w = mask.shape
    if mask.sum() == 0:
        return None
    y0b, y1b, x0b, x1b = _bbox(mask)
    avail_h, avail_w = h - (y1b - y0b), w - (x1b - x0b)
    win_h = (y1b - y0b) + _CUTOFF_MARGIN_FRACTION * avail_h
    win_w = (x1b - x0b) + _CUTOFF_MARGIN_FRACTION * avail_w
    cy0, cx0 = (y0b + y1b) / 2.0, (x0b + x1b) / 2.0
    tol = _tol("cutoff", target)

    rng = np.random.default_rng(seed)
    direction = rng.choice(["left", "right", "up", "down"])

    max_shift = float(max(h, w))
    f_max, _ = _cutoff_fraction(mask, cy0, cx0, win_h, win_w, direction, max_shift)
    if target > f_max + tol:
        return None
    f_min, _ = _cutoff_fraction(mask, cy0, cx0, win_h, win_w, direction, 0.0)
    if target < f_min - tol:
        return None

    lo, hi = 0.0, max_shift
    for _ in range(40):
        mid = (lo + hi) / 2.0
        f_mid, _ = _cutoff_fraction(mask, cy0, cx0, win_h, win_w, direction, mid)
        if f_mid < target:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2.0
    achieved, box = _cutoff_fraction(mask, cy0, cx0, win_h, win_w, direction, shift)
    if abs(achieved - target) > tol:
        return None
    return box, achieved


# --- off_center: crop containing the whole animal, shifted horizontally -------------------------------
#
# The window's SIZE (both dimensions) is a function of the mask alone -- never of `target` -- and only
# its horizontal POSITION depends on the target. An earlier version of this function sized the window's
# width directly from the target (`win_w = wb/(1-target)`, growing with the target so there'd always be
# "enough" slack): that made a single source's level 0 and level 4 crops wildly different in aspect
# ratio and zoom (extremely tall/narrow vs. wide), which is exactly the kind of giveaway cue the rater
# could use instead of the animal's actual position. Fixing the window per source and only checking
# whether *this* window has enough slack for a given target (rejecting the source for that level if not)
# costs some reach on the most extreme targets, but keeps every level of a given item looking like the
# same photo shot from the same distance.
_OFF_CENTER_V_MARGIN = 0.08  # vertical containment margin, a fraction of the bbox height
_OFF_CENTER_FILL_FLOOR = 0.10
_OFF_CENTER_FILL_CEIL = 0.35


def _off_center_window(mask: np.ndarray) -> tuple[float, float, float] | None:
    """(win_w, win_h, fill), sized only from `mask`. Both dimensions grow together, as a fraction of
    each axis's own available slack (image_dim - bbox_dim) -- the same scheme `_fit_cutoff_box` uses,
    for the same reason: sizing a dimension as a multiple of the *bbox* (or, worse, pushing it out to
    the fill floor regardless of the image's own width) can come within a hair of the image's own size,
    which leaves ~0 room to slide the window at all, making even target 0 unreachable. A fraction of the
    real leftover slack guarantees most of that slack stays available for the shift."""
    h, w = mask.shape
    a = float(mask.sum())
    if a == 0:
        return None
    y0b, y1b, x0b, x1b = _bbox(mask)
    wb, hb = x1b - x0b, y1b - y0b
    avail_w, avail_h = w - wb, h - hb
    if avail_w <= 0 or avail_h <= 0:
        return None

    margin_frac = _OFF_CENTER_V_MARGIN
    win_w = wb + margin_frac * avail_w
    win_h = hb + margin_frac * avail_h
    fill = a / (win_w * win_h)
    for _ in range(60):
        if fill <= _OFF_CENTER_FILL_CEIL:
            break
        margin_frac = min(margin_frac * 1.15 + 0.01, 1.0)
        win_w = wb + margin_frac * avail_w
        win_h = hb + margin_frac * avail_h
        fill = a / (win_w * win_h)
        if margin_frac >= 1.0:
            break
    if fill > _OFF_CENTER_FILL_CEIL + 1e-9 or fill < _OFF_CENTER_FILL_FLOOR - 1e-9:
        return None
    return win_w, win_h, fill


def _fit_off_center_box(mask: np.ndarray, target: float, seed: int) -> tuple[tuple, float] | None:
    h, w = mask.shape
    if mask.sum() == 0:
        return None
    y0b, y1b, x0b, x1b = _bbox(mask)
    wb = x1b - x0b
    cy_c, cx_c = _centroid(mask)
    tol = _tol("off_center", target)

    win = _off_center_window(mask)
    if win is None:
        return None
    win_w, win_h, _fill = win
    half = win_w / 2.0
    if target * half > (half - wb / 2.0) + tol * half:
        return None  # this source's (level-independent) window has no room for this big a shift

    rng = np.random.default_rng(seed)
    sign = 1.0 if rng.random() < 0.5 else -1.0

    win_cy = (y0b + y1b) / 2.0
    win_cx = cx_c - sign * target * half
    # The window itself (not just the bbox-containment slack checked above) must fit inside the image
    # at this shifted position without being clamped -- clamping would silently move win_cx and throw
    # off the achieved offset (this bit an earlier version of this function for very wide windows,
    # where win_w approaches the image width and there's almost no room to slide the window itself).
    if win_cx - half < 0 or win_cx + half > w:
        return None
    y0, y1 = win_cy - win_h / 2.0, win_cy + win_h / 2.0
    x0, x1 = win_cx - half, win_cx + half
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if y1 > h:
        y0 -= y1 - h
        y1 = float(h)
    y0 = max(0.0, y0)
    if y0 > y0b or y1 < y1b or x0 > x0b or x1 < x1b:
        return None  # can't keep the whole animal inside after clamping to the image
    win_cx_actual = (x0 + x1) / 2.0
    achieved = abs(cx_c - win_cx_actual) / half
    if abs(achieved - target) > tol:
        return None
    return (y0, y1, x0, x1), achieved


# --- occlusion: grey square sized/placed against the animal mask (at BASE_SIDE resolution) -----------


def _occlusion_side_for_target(mask: np.ndarray, target: float) -> tuple[float, float] | None:
    h, w = mask.shape
    a = float(mask.sum())
    if a == 0:
        return None
    cy, cx = _centroid(mask)

    def frac_for(side: float) -> float:
        top, left = cy - side / 2.0, cx - side / 2.0
        y0, x0 = int(round(top)), int(round(left))
        y1, x1 = min(h, y0 + int(round(side))), min(w, x0 + int(round(side)))
        y0, x0 = max(0, y0), max(0, x0)
        if y1 <= y0 or x1 <= x0:
            return 0.0
        return float(mask[y0:y1, x0:x1].sum()) / a

    tol = _tol("occlusion", target)
    lo, hi = 1.0, float(min(h, w))
    if frac_for(hi) < target - tol:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if frac_for(mid) < target:
            lo = mid
        else:
            hi = mid
    side = (lo + hi) / 2.0
    achieved = frac_for(side)
    if abs(achieved - target) > tol:
        return None
    return side, achieved


def _fit_occlusion_rect(mask: np.ndarray, target: float, seed: int) -> tuple[tuple, float] | None:
    h, w = mask.shape
    a = float(mask.sum())
    if a == 0:
        return None
    cy, cx = _centroid(mask)
    ref = _occlusion_side_for_target(mask, SCALE_PARAMS["occlusion"][1])  # "comparable size" reference
    if ref is None:
        return None
    ref_side, _ = ref
    rng = np.random.default_rng(seed)

    if target <= 1e-9:
        side = ref_side

        def overlap(top: float, left: float) -> float:
            y0, x0 = int(round(top)), int(round(left))
            y1, x1 = min(h, y0 + int(round(side))), min(w, x0 + int(round(side)))
            y0, x0 = max(0, y0), max(0, x0)
            if y1 <= y0 or x1 <= x0:
                return 0.0
            return float(mask[y0:y1, x0:x1].sum()) / a

        for _ in range(300):
            top = rng.uniform(0, max(1.0, h - side))
            left = rng.uniform(0, max(1.0, w - side))
            if overlap(top, left) <= 1e-9:
                return (top, left, side), 0.0
        return None

    fit = _occlusion_side_for_target(mask, target)
    if fit is None:
        return None
    side, achieved = fit
    top = min(max(cy - side / 2.0, 0.0), max(0.0, h - side))
    left = min(max(cx - side / 2.0, 0.0), max(0.0, w - side))
    return (top, left, side), achieved


# --- tilt: rotate then crop the largest axis-aligned inscribed rectangle -------------------------------


def _inscribed_scale(w: float, h: float, angle_deg: float) -> float:
    """Largest `k` in (0, 1] such that the k*w x k*h rectangle, centred at the same centre as the w x h
    rectangle, fits entirely inside the w x h rectangle once it's rotated by `angle_deg`. Found by
    bisection on the (monotone) containment test of all 4 corners under the inverse rotation."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def corners_ok(k: float) -> bool:
        hx, hy = k * w / 2.0, k * h / 2.0
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                x, y = sx * hx, sy * hy
                xr = x * cos_a + y * sin_a
                yr = -x * sin_a + y * cos_a
                if abs(xr) > w / 2.0 + 1e-6 or abs(yr) > h / 2.0 + 1e-6:
                    return False
        return True

    if corners_ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if corners_ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


# `_inscribed_scale`'s corner test is exact for the mathematical rectangle, but PIL's BICUBIC resample
# blends in a few pixels' worth of neighbourhood around every output pixel -- so a crop right at the
# theoretical edge can still pick up a trace of the black rotation fill just outside it (visible as dark
# wedges in the corners of the crop). `_TILT_BASE_MARGIN` shrinks the crop by this fraction per side
# (applied identically at every level, including level 0, so the margin itself is never a level cue);
# `_tilt_crop` verifies the result and adaptively grows the margin, per item, until it's provably clean.
_TILT_BASE_MARGIN = 0.02
_TILT_MARGIN_STEP = 0.01
_TILT_MAX_MARGIN = 0.14

#: Populated by `_tilt_crop()` as a side effect: counts of items rendered at the base margin, items that
#: needed a larger (adaptively grown) margin, and the largest margin actually used. Reset by `build()`.
LAST_TILT_MARGIN_STATS: dict[str, Any] = {"base": 0, "bumped": 0, "max_margin_used": _TILT_BASE_MARGIN}


def _tilt_mask_is_clean(w: int, h: int, angle: float, box: tuple[int, int, int, int]) -> bool:
    """Proof, not a heuristic: rotate a pure-white single-channel mask with exactly the transform used
    on the photo (same angle, BICUBIC resample, expand=True, fill=0), crop it with the same box, and
    check every pixel is still 255. If any pixel dipped below 255, the crop touched rotation fill or an
    interpolation blend of it, and the photo crop would show a dark wedge there too."""
    white = Image.new("L", (w, h), 255)
    rotated = white if angle == 0.0 else white.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
    crop = np.asarray(rotated.crop(box))
    return bool(crop.size) and bool(crop.min() == 255)


def _tilt_crop(base_image: Image.Image, angle: float, k_mag: float) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """The rotated (or, for angle 0, original) image and a crop box that `_tilt_mask_is_clean` proves
    contains no rotation-fill / interpolation-blend pixels, found by growing the safety margin from
    `_TILT_BASE_MARGIN` in `_TILT_MARGIN_STEP` increments. Raises `Infeasible` (extremely rare) if even
    `_TILT_MAX_MARGIN` isn't enough for this source."""
    w, h = base_image.size
    k = _inscribed_scale(w, h, k_mag)
    rotated = base_image if angle == 0.0 else base_image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    rw, rh = rotated.size

    margin = _TILT_BASE_MARGIN
    while margin <= _TILT_MAX_MARGIN + 1e-9:
        k_safe = k * (1.0 - margin)
        cw, ch = min(k_safe * w, rw), min(k_safe * h, rh)
        left, top = (rw - cw) / 2.0, (rh - ch) / 2.0
        box = (int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch)))
        if _tilt_mask_is_clean(w, h, angle, box):
            LAST_TILT_MARGIN_STATS["base" if margin <= _TILT_BASE_MARGIN + 1e-9 else "bumped"] += 1
            LAST_TILT_MARGIN_STATS["max_margin_used"] = max(LAST_TILT_MARGIN_STATS["max_margin_used"], margin)
            return rotated, box
        margin += _TILT_MARGIN_STEP
    raise Infeasible(f"tilt: no crop within margin<={_TILT_MAX_MARGIN} proved clean of rotation fill for this source")


# --- final resize (longest side -> BASE_SIDE, LANCZOS, never distorts aspect) --------------------------


def _finalize(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = BASE_SIDE / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if new_size == (w, h):
        return img.copy()
    return img.resize(new_size, Image.LANCZOS)


# --- render -------------------------------------------------------------------------------------------


def render(base_image: Image.Image, mask: np.ndarray, key: str, level: int, seed: int) -> Image.Image:
    """Deterministic given its arguments. `base_image` is the *original*-resolution RGB photo and
    `mask` is a same-size (H, W) uint8 array (1 = animal), except for `occlusion`, `text_legibility`
    and `watermark`, which work on the BASE_SIDE-resized image (see `_mask_to_base`). Raises
    `Infeasible` for subject_size/cutoff/off_center/occlusion when this source's mask can't hit the
    level's target within tolerance -- callers pick another source in that case."""
    if base_image.mode != "RGB":
        base_image = base_image.convert("RGB")

    if key == "subject_size":
        target = SCALE_PARAMS[key][level]
        fit = _fit_subject_size_box(mask, target, seed)
        if fit is None:
            raise Infeasible(f"subject_size: no crop hits target={target} for this source")
        (y0, y1, x0, x1), _achieved = fit
        crop = base_image.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
        return _finalize(crop)

    if key == "cutoff":
        target = SCALE_PARAMS[key][level]
        fit = _fit_cutoff_box(mask, target, seed)
        if fit is None:
            raise Infeasible(f"cutoff: no crop hits target={target} for this source")
        (y0, y1, x0, x1), _achieved = fit
        crop = base_image.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
        return _finalize(crop)

    if key == "off_center":
        target = SCALE_PARAMS[key][level]
        fit = _fit_off_center_box(mask, target, seed)
        if fit is None:
            raise Infeasible(f"off_center: no crop hits target={target} for this source")
        (y0, y1, x0, x1), _achieved = fit
        crop = base_image.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
        return _finalize(crop)

    if key == "occlusion":
        target = SCALE_PARAMS[key][level]
        base = ladders._to_base(base_image)
        mask_small = _mask_to_base(mask)
        fit = _fit_occlusion_rect(mask_small, target, seed)
        if fit is None:
            raise Infeasible(f"occlusion: no rectangle hits target={target} for this source")
        (top, left, side), _achieved = fit
        canvas = base.copy()
        draw = ImageDraw.Draw(canvas)
        x0i, y0i = int(round(left)), int(round(top))
        x1i, y1i = int(round(left + side)), int(round(top + side))
        draw.rectangle([x0i, y0i, x1i, y1i], fill=(128, 128, 128))
        return _finalize(canvas)

    if key == "tilt":
        magnitudes = SCALE_PARAMS["tilt"]
        mag = magnitudes[level]
        rng = np.random.default_rng(seed)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        angle = 0.0 if level == 0 else sign * mag
        k_mag = magnitudes[2] if level == 0 else mag
        rotated, box = _tilt_crop(base_image, angle, k_mag)
        return _finalize(rotated.crop(box))

    if key == "text_legibility":
        base = ladders._to_base(base_image)
        w, h = base.size
        params = TEXT_LEVEL_PARAMS[level]
        rng = np.random.default_rng(seed)
        caption = CAPTIONS[int(rng.integers(0, len(CAPTIONS)))]
        lines = _caption_lines(caption, w)  # same line break at every level; only styling below varies
        font = _font(params["font_size"])
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        line_boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
        line_heights = [b[3] - b[1] for b in line_boxes]
        line_gap = round(params["font_size"] * _TEXT_LINE_GAP_FRACTION)
        total_h = sum(line_heights) + line_gap * (len(lines) - 1)
        block_bottom = min(h - 2, int(h * 0.95))
        block_top = max(0, block_bottom - total_h)

        band = base.crop((0, max(0, block_top - 4), w, min(h, block_bottom + 4)))
        band_arr = np.asarray(band).reshape(-1, 3) if band.size else None
        bg_color = tuple(int(c) for c in band_arr.mean(axis=0)) if band_arr is not None and band_arr.size else (128, 128, 128)

        mode = params["color_mode"]
        if mode == "background":
            fill_rgb = bg_color
        elif mode.startswith("blend"):
            pct = int(mode[len("blend"):]) / 100.0
            fill_rgb = tuple(int(c * (1 - pct) + 255 * pct) for c in bg_color)
        else:
            fill_rgb = (255, 255, 255)

        alpha = int(round(params["opacity"] * 255))
        ow = params["outline"]
        y = block_top
        for line, bbox, lh in zip(lines, line_boxes, line_heights):
            tw = bbox[2] - bbox[0]
            tx, ty = (w - tw) // 2 - bbox[0], y - bbox[1]
            if ow > 0:
                for dx in range(-ow, ow + 1):
                    for dy in range(-ow, ow + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((tx + dx, ty + dy), line, font=font, fill=(0, 0, 0, alpha))
            draw.text((tx, ty), line, font=font, fill=(*fill_rgb, alpha))
            y += lh + line_gap
        if params["blur"] > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(params["blur"]))
        out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
        return _finalize(out)

    if key == "watermark":
        base = ladders._to_base(base_image)
        opacity = SCALE_PARAMS["watermark"][level]
        if opacity <= 0:
            return _finalize(base)
        layer = _watermark_layer(base.size)
        r, g, b, a = layer.split()
        a = a.point(lambda v: int(v * opacity))
        layer = Image.merge("RGBA", (r, g, b, a))
        out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
        return _finalize(out)

    raise ValueError(f"unknown semantic scale {key!r}")


def _watermark_layer(size: tuple[int, int]) -> Image.Image:
    """A full-alpha (255) diagonally tiled 'SAMPLE' layer covering `size`, with fixed spacing/font
    size. Same tiling for every level of a given item (and, size aside, every item); only the alpha
    blended in by the caller changes across levels."""
    w, h = size
    diag = int(math.hypot(w, h)) + WATERMARK_SPACING * 2
    tile = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    font = _font(WATERMARK_FONT_SIZE)
    for y in range(0, diag, WATERMARK_SPACING):
        row_offset = (WATERMARK_SPACING if (y // WATERMARK_SPACING) % 2 else 0)
        for x in range(-WATERMARK_SPACING, diag, WATERMARK_SPACING * 2):
            draw.text((x + row_offset, y), WATERMARK_TEXT, font=font, fill=(255, 255, 255, 255))
    rotated = tile.rotate(WATERMARK_ANGLE, expand=False, resample=Image.BICUBIC)
    left, top = (diag - w) // 2, (diag - h) // 2
    return rotated.crop((left, top, left + w, top + h))


_FIT_FUNCS = {
    "subject_size": _fit_subject_size_box,
    "cutoff": _fit_cutoff_box,
    "off_center": _fit_off_center_box,
    "occlusion": lambda mask, target, seed: _fit_occlusion_rect(_mask_to_base(mask), target, seed),
}


# --- build ----------------------------------------------------------------------------------------


#: Populated by `_build_simple_scale()` as a side effect: {scale_key: number of slots whose first-choice
#: source raised `Infeasible` and had to be replaced by the next unused source in the pool}. In practice
#: only `tilt` can ever raise (see `_tilt_crop`), and only when even `_TILT_MAX_MARGIN` isn't enough.
LAST_SIMPLE_BUILD_REPLACED: dict[str, int] = {}


def _build_simple_scale(raw_bytes: dict[str, bytes], pool: list[ladders._Source], key: str,
                         n_per_scale: int, force: bool) -> list[dict[str, Any]]:
    order = _order_for(pool, key)
    key_dir = IMAGES_DIR / key
    ext = "png" if key in PNG_KEYS else "jpg"

    used: set[str] = set()
    cursor = 0

    def next_unused() -> ladders._Source | None:
        nonlocal cursor
        while cursor < len(order):
            cand = order[cursor]
            cursor += 1
            if cand.image_id not in used:
                return cand
        return None

    rows = []
    replaced = 0
    for i in range(n_per_scale):
        level = i % N_LEVELS
        split = _split_for(i)
        attempts = 0
        while True:
            src = next_unused()
            if src is None:
                raise RuntimeError(f"semantic.{key}: ran out of sources building slot {i} (level {level})")
            seed = _seed_for(key, src.image_id, level)
            photo, mask = _open_source(raw_bytes, src.image_id)
            try:
                img = render(photo, mask, key, level, seed)
                break
            except Infeasible:
                attempts += 1
                continue
        if attempts > 0:
            replaced += 1
        used.add(src.image_id)

        out_path = key_dir / f"{src.image_id}_L{level}.{ext}"
        if force or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if ext == "png":
                img.save(out_path, format="PNG", optimize=True)
            else:
                img.save(out_path, format="JPEG", quality=97)
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
        param = TEXT_LEVEL_PARAMS[level] if key == "text_legibility" else SCALE_PARAMS[key][level]
        rows.append({
            "item_id": f"{src.image_id}_L{level}", "ladder": key, "source_image_id": src.image_id,
            "level": level, "split": split, "path": _relpath(out_path), "sha256": sha256, "param": param,
        })
    LAST_SIMPLE_BUILD_REPLACED[key] = replaced
    return rows


def _level_scarcity(pool: list[ladders._Source], order: list[ladders._Source], key: str) -> list[int]:
    """Levels ordered rarest-target-first: a full-pool feasibility count per level (ignoring which
    sources other levels might also want), ranked ascending. Processing the scarcest level first in
    `_build_fit_scale` means its few feasible sources are claimed before an abundant, easy level (which
    has plenty of alternatives either way) greedily eats into that same small pool -- without this,
    consuming sources for level 0 in plain 0..4 order can starve a much rarer level 4."""
    fit_fn = _FIT_FUNCS[key]
    counts = []
    for level in range(N_LEVELS):
        target = SCALE_PARAMS[key][level]
        n = 0
        for src in order:
            seed = _seed_for(key, src.image_id, level)
            mask = _load_mask_only(src.image_id)
            if fit_fn(mask, target, seed) is not None:
                n += 1
        counts.append(n)
    return sorted(range(N_LEVELS), key=lambda lvl: counts[lvl])


def _build_fit_scale(raw_bytes: dict[str, bytes], pool: list[ladders._Source], key: str,
                      n_per_scale: int, force: bool) -> list[dict[str, Any]]:
    order = _order_for(pool, key)
    needed = n_per_scale // N_LEVELS
    key_dir = IMAGES_DIR / key
    used: set[str] = set()
    accepted_by_level: dict[int, list[tuple[ladders._Source, int, float]]] = {}
    fit_fn = _FIT_FUNCS[key]

    for level in _level_scarcity(pool, order, key):
        target = SCALE_PARAMS[key][level]
        accepted: list[tuple[ladders._Source, int, float]] = []
        for src in order:
            if src.image_id in used:
                continue
            seed = _seed_for(key, src.image_id, level)
            mask = _load_mask_only(src.image_id)
            fit = fit_fn(mask, target, seed)
            if fit is None:
                continue
            _box, achieved = fit
            accepted.append((src, seed, achieved))
            used.add(src.image_id)
            if len(accepted) == needed:
                break
        if len(accepted) < needed:
            raise RuntimeError(
                f"semantic.{key} level {level} (target={target}): only found {len(accepted)}/{needed} "
                f"feasible sources in a pool of {len(pool)}; refusing to relax the tolerance"
            )
        accepted_by_level[level] = accepted

    rows = []
    for level in range(N_LEVELS):
        for idx, (src, seed, achieved) in enumerate(accepted_by_level[level]):
            split = "calibration" if idx % 2 == 0 else "test"
            out_path = key_dir / f"{src.image_id}_L{level}.jpg"
            if force or not out_path.exists():
                photo, mask = _open_source(raw_bytes, src.image_id)
                img = render(photo, mask, key, level, seed)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path, format="JPEG", quality=97)
            sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
            rows.append({
                "item_id": f"{src.image_id}_L{level}", "ladder": key, "source_image_id": src.image_id,
                "level": level, "split": split, "path": _relpath(out_path), "sha256": sha256, "param": achieved,
            })
    return rows


#: Populated by build() as a side effect: {scale_key: error message} for any scale that could not be
#: filled at the requested n_per_scale (build() still writes manifests for every OTHER scale before
#: raising, rather than aborting on the first failure -- see build()'s docstring).
LAST_BUILD_ERRORS: dict[str, str] = {}


def build(cfg: Config, n_per_scale: int = N_PER_SCALE, force: bool = False) -> dict[str, list[dict]]:
    """Builds every scale in SCALES. If one scale's search-based cells can't be filled (see
    `_build_fit_scale`), that scale is skipped -- its error is recorded in `LAST_BUILD_ERRORS` -- but
    every OTHER scale is still attempted and, on success, has its manifest written; only after every
    scale has been attempted does `build()` raise a single RuntimeError summarizing every failure (the
    tolerance itself is never relaxed to avoid that error, per the module's design). Callers that want
    partial results despite a raised error can inspect `LAST_BUILD_ERRORS` and re-read whichever
    `lab/manifests_semantic/<key>.jsonl` files exist."""
    if n_per_scale % N_LEVELS != 0:
        raise ValueError(f"n_per_scale ({n_per_scale}) must be a multiple of {N_LEVELS}")
    LAST_TILT_MARGIN_STATS.update({"base": 0, "bumped": 0, "max_margin_used": _TILT_BASE_MARGIN})
    LAST_SIMPLE_BUILD_REPLACED.clear()
    raw_bytes, pool = _semantic_pool(cfg)
    if len(pool) < n_per_scale:
        raise RuntimeError(
            f"semantic source pool (photos with a matching trimap) has only {len(pool)} sources, need {n_per_scale}"
        )

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for key in SCALES:
        try:
            if key in FIT_KEYS:
                rows = _build_fit_scale(raw_bytes, pool, key, n_per_scale, force)
            else:
                rows = _build_simple_scale(raw_bytes, pool, key, n_per_scale, force)
        except RuntimeError as exc:
            errors[key] = str(exc)
            continue
        _write_jsonl(MANIFESTS_DIR / f"{key}.jsonl", rows)
        results[key] = rows

    LAST_BUILD_ERRORS.clear()
    LAST_BUILD_ERRORS.update(errors)
    if errors:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(
            f"semantic.build: {len(errors)}/{len(SCALES)} scale(s) could not be filled at "
            f"n_per_scale={n_per_scale} (every other scale's manifest was still written): {detail}"
        )
    return results


# --- contact sheets ---------------------------------------------------------------------------------


def contact_sheet(cfg: Config, key: str, out_path: str | Path, n_sources: int = 3) -> None:
    """A grid: n_sources rows x 5 level columns, each cell 300 px wide. Recomputes the pool/order
    itself (independent of `build()`); for the 4 search-based scales it prefers sources feasible at
    *every* level, but falls back to whichever sources cover the most levels if fewer than `n_sources`
    are fully feasible (some (scale, level) targets are only feasible for a handful of sources in this
    dataset at all -- see `build()`/`LAST_BUILD_ERRORS`). A level a chosen source can't hit is drawn as
    a labeled placeholder cell instead of failing the whole sheet."""
    raw_bytes, pool = _semantic_pool(cfg)
    order = _order_for(pool, key)

    cell_w = 300
    pad = 8
    header_h = 28
    font = ImageFont.load_default()

    scored: list[tuple[int, ladders._Source, list[Image.Image | None]]] = []
    full_found = 0
    for src in order:
        photo, mask = _open_source(raw_bytes, src.image_id)
        thumbs: list[Image.Image | None] = []
        n_ok = 0
        for level in range(N_LEVELS):
            seed = _seed_for(key, src.image_id, level)
            try:
                thumbs.append(render(photo, mask, key, level, seed))
                n_ok += 1
            except Infeasible:
                thumbs.append(None)
        scored.append((n_ok, src, thumbs))
        full_found += n_ok == N_LEVELS
        if full_found >= n_sources:  # enough fully-feasible sources; no need to scan the rest of the pool
            break

    scored.sort(key=lambda t: -t[0])
    chosen = scored[:n_sources]
    if not chosen:
        raise RuntimeError(f"contact_sheet({key}): no sources available")

    placeholder_w, placeholder_h = cell_w, round(cell_w * 0.75)

    def placeholder(level: int) -> Image.Image:
        img = Image.new("RGB", (placeholder_w, placeholder_h), (230, 230, 230))
        d = ImageDraw.Draw(img)
        text = f"no source hits\nlevel {level} target"
        d.multiline_text((10, placeholder_h // 2 - 12), text, fill=(120, 0, 0), font=font)
        return img

    rows_of_thumbs: list[list[Image.Image]] = []
    row_heights: list[int] = []
    for _n_ok, _src, thumbs in chosen:
        real = next((t for t in thumbs if t is not None), None)
        cell_h = round(cell_w * real.height / real.width) if real is not None else placeholder_h
        row = []
        for level, t in enumerate(thumbs):
            cell = t.resize((cell_w, cell_h), Image.LANCZOS) if t is not None else placeholder(level).resize((cell_w, cell_h))
            row.append(cell)
        rows_of_thumbs.append(row)
        row_heights.append(cell_h)

    def label(level: int) -> str:
        if key == "text_legibility":
            p = TEXT_LEVEL_PARAMS[level]
            return f"L{level} font={p['font_size']} op={p['opacity']}"
        return f"level {level}  param={SCALE_PARAMS[key][level]}"

    width = pad * (N_LEVELS + 1) + cell_w * N_LEVELS
    height = header_h + pad + sum(row_heights) + pad * len(rows_of_thumbs)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col in range(N_LEVELS):
        x = pad + col * (cell_w + pad)
        draw.text((x, 6), label(col), fill="black", font=font)

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
    try:
        results = build(cfg)
    except RuntimeError as exc:
        print(f"[semantic] build() raised: {exc}")
        results = {}
        for key in SCALES:
            if key in LAST_BUILD_ERRORS:
                continue
            path = MANIFESTS_DIR / f"{key}.jsonl"
            if path.exists():
                from ..logging_utils import read_jsonl

                results[key] = read_jsonl(path)

    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for key in SCALES:  # every scale, even ones build() couldn't fill -- contact_sheet degrades gracefully
        contact_sheet(cfg, key, SHEETS_DIR / f"{key}.jpg")

    stats = LAST_POOL_STATS
    print(
        f"pool: semantic_candidates={stats.get('semantic_candidates')} "
        f"skipped_no_trimap={stats.get('skipped_no_trimap')} skipped_size_mismatch={stats.get('skipped_size_mismatch')} "
        f"semantic_pool={stats.get('semantic_pool')}"
    )
    for key, rows in results.items():
        counts: dict[tuple[str, int], int] = {}
        sums: dict[int, float] = {}
        for row in rows:
            k = (row["split"], row["level"])
            counts[k] = counts.get(k, 0) + 1
            if isinstance(row["param"], (int, float)):
                sums[row["level"]] = sums.get(row["level"], 0.0) + row["param"]
        means = {lvl: sums[lvl] / (counts.get(("calibration", lvl), 0) + counts.get(("test", lvl), 0))
                 for lvl in sums} if sums else {}
        print(f"{key}: n={len(rows)} counts={sorted(counts.items())} mean_param_by_level={means}")

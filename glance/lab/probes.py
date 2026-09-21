"""Six procedural probe suites (lab/NOTES.md entry 50): images drawn entirely by this module (PIL + numpy,
no photos), so every label is exact by construction and nothing here can be in any model's training set.

Every image is 640 x 480 RGB, rendered at `SCALE`x on an oversized canvas and downsampled with LANCZOS for
antialiasing. Generation is a pure function of (suite, item index) via seeded `random.Random`/`np.random.Generator`
instances derived from the fixed module `SEED`, so `build()` is fully deterministic and safe to re-run (images are
skipped when already on disk unless `force=True`; the source manifest is always rewritten).

- `count`: 1-8 non-overlapping coloured discs (balls); label = the count.
- `count_color`: 3-9 discs of several colours, 0-6 of them pure red; label = the red count.
- `spatial`: one red ball and one blue square, offset diagonally; two yes/no items per image (left/right,
  above/below).
- `largest`: four shapes (disc, square, triangle, diamond) in four colours, three at the same area and one
  scaled by a ratio r; label = the colour of the largest.
- `stripes`: full-frame stripes at one of 4 orientations; label = the orientation.
- `text`: one word from a group of six look-alike words, printed centred with a slight rotation; label = the
  word.

`build()` writes images to `.cache/lab_images_probes/<suite>/<file>` and one manifest per suite to
`glance/evals/manifests/probes_<suite>_source.jsonl` (keys: item_id, file, label, params, sha256). The suite
loaders in `glance/evals/suites/probes.py` read those manifests as their raw source, the same way
`fresh_inat.py` reads a fetched manifest.

Run `uv run python -c "from glance.lab import probes; probes.build()"` to build everything (900 images), or
`python -m glance.lab.probes` to also write contact sheets under `lab/sheets_probes/`.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import Config, PROJECT_ROOT
from ..evals.suites.base import MANIFEST_DIR as EVAL_MANIFEST_DIR

SEED = 7
SCALE = 2  # supersampling factor; every suite draws at (WIDTH*SCALE, HEIGHT*SCALE) then downsamples LANCZOS
WIDTH, HEIGHT = 640, 480

LAB_DIR = PROJECT_ROOT / "lab"
IMAGES_ROOT = PROJECT_ROOT / ".cache" / "lab_images_probes"
SHEETS_DIR = LAB_DIR / "sheets_probes"

SUITES = ("count", "count_color", "spatial", "largest", "stripes", "text")


# --- shared determinism / balance helpers ------------------------------------------------------------


def _seed_int(key: str) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _balanced_labels(values: list[Any], n: int, key: str) -> list[Any]:
    """A length-n list where each of `values` appears floor(n/len(values)) or that +1 times (as evenly as
    integer division allows), in a seeded-random order. Deterministic given `key`."""
    k = len(values)
    base, rem = divmod(n, k)
    labels: list[Any] = []
    for idx, v in enumerate(values):
        labels.extend([v] * (base + (1 if idx < rem else 0)))
    random.Random(f"{SEED}:{key}:balance").shuffle(labels)
    return labels


def _finalize(canvas_2x: Image.Image, np_rng: np.random.Generator | None, noise_sigma: float) -> Image.Image:
    """Downsamples the 2x canvas to (WIDTH, HEIGHT) with LANCZOS, then optionally adds light additive
    Gaussian noise (per-pixel, per-channel) at final resolution."""
    final = canvas_2x.resize((WIDTH, HEIGHT), Image.LANCZOS)
    if noise_sigma > 0 and np_rng is not None:
        arr = np.asarray(final, dtype=np.float64)
        arr = np.clip(arr + np_rng.normal(0.0, noise_sigma, size=arr.shape), 0, 255)
        final = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return final


def _write_manifest(filename: str, rows: list[dict[str, Any]]) -> None:
    path = EVAL_MANIFEST_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- disc placement (shared by `count` and `count_color`) ---------------------------------------------

DISC_RADIUS_RANGE = (18.0, 46.0)
DISC_MIN_GAP = 8.0
DISC_BORDER_MARGIN = 12.0


def _place_balls(
    seed_key: str, count: int, max_retries: int = 25
) -> tuple[random.Random, list[float], list[tuple[float, float]]]:
    """Rejection-samples `count` non-overlapping disc radii/centres in the (WIDTH, HEIGHT) frame: every pair
    of discs is at least `DISC_MIN_GAP` px apart edge-to-edge, and every disc is at least `DISC_BORDER_MARGIN`
    px from the border. Deterministic given `seed_key`. Places the largest discs first (harder to fit) and
    retries the *whole* configuration (not just one disc) with a derived sub-seed if rejection sampling can't
    place every disc; returns the `random.Random` used so the caller can keep drawing from the same stream."""
    for attempt in range(max_retries):
        rng = random.Random(f"{seed_key}:{attempt}")
        radii = [rng.uniform(*DISC_RADIUS_RANGE) for _ in range(count)]
        order = sorted(range(count), key=lambda idx: -radii[idx])
        centers: list[tuple[float, float] | None] = [None] * count
        placed: list[tuple[float, float, float]] = []
        ok = True
        for idx in order:
            r = radii[idx]
            found = False
            for _ in range(4000):
                cx = rng.uniform(r + DISC_BORDER_MARGIN, WIDTH - r - DISC_BORDER_MARGIN)
                cy = rng.uniform(r + DISC_BORDER_MARGIN, HEIGHT - r - DISC_BORDER_MARGIN)
                if all(math.hypot(cx - ox, cy - oy) >= r + orad + DISC_MIN_GAP for ox, oy, orad in placed):
                    centers[idx] = (cx, cy)
                    placed.append((cx, cy, r))
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            return rng, radii, [c for c in centers if c is not None]
    raise RuntimeError(f"probes: could not place {count} non-overlapping discs for seed_key={seed_key!r}")


def _draw_balls(canvas: Image.Image, centers: list[tuple[float, float]], radii: list[float], colors: list[tuple[int, int, int]]) -> None:
    draw = ImageDraw.Draw(canvas)
    for (cx, cy), r, color in zip(centers, radii, colors):
        bbox = [(cx - r) * SCALE, (cy - r) * SCALE, (cx + r) * SCALE, (cy + r) * SCALE]
        draw.ellipse(bbox, fill=color)


def _plain_or_noisy_background(rng: random.Random) -> tuple[int, bool]:
    bg_val = round(rng.uniform(228, 242))
    return bg_val, rng.random() < 0.5


# --- `count`: 1-8 non-overlapping balls, count balanced -------------------------------------------------

PALETTE_8: dict[str, tuple[int, int, int]] = {
    "red": (219, 40, 40), "orange": (230, 126, 34), "yellow": (232, 196, 16), "green": (39, 174, 96),
    "blue": (41, 98, 204), "purple": (142, 68, 173), "pink": (231, 84, 158), "teal": (26, 154, 148),
}


def _build_count(n: int, force: bool) -> list[dict]:
    counts = _balanced_labels(list(range(1, 9)), n, "count")
    palette_names = list(PALETTE_8)
    out_dir = IMAGES_ROOT / "count"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        count = counts[i]
        seed_key = f"{SEED}:count:{i}"
        rng, radii, centers = _place_balls(seed_key, count)
        colors_keys = [rng.choice(palette_names) for _ in range(count)]
        colors_rgb = [PALETTE_8[k] for k in colors_keys]
        bg_val, use_noise = _plain_or_noisy_background(rng)

        canvas2x = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), (bg_val, bg_val, bg_val))
        _draw_balls(canvas2x, centers, radii, colors_rgb)
        np_rng = np.random.default_rng(_seed_int(seed_key))
        noise_sigma = rng.uniform(4.0, 9.0) if use_noise else 0.0
        img = _finalize(canvas2x, np_rng, noise_sigma)

        item_id = f"count_{i:03d}"
        path = out_dir / f"{item_id}.png"
        if force or not path.exists():
            img.save(path, format="PNG")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "item_id": item_id, "file": f"{item_id}.png", "label": str(count), "sha256": sha256,
            "params": {
                "count": count, "centers": [list(c) for c in centers], "radii": radii,
                "colors": colors_keys, "background": "noise" if use_noise else "plain",
            },
        })
    _write_manifest("probes_count_source.jsonl", rows)
    return rows


# --- `count_color`: 3-9 balls, 0-6 pure red, red count balanced -----------------------------------------

RED_COLOR = (214, 26, 26)  # pure-ish red, clearly distinct from PALETTE_8's orange/pink
COUNT_COLOR_OTHERS: dict[str, tuple[int, int, int]] = {
    "blue": (41, 98, 204), "green": (39, 174, 96), "yellow": (232, 196, 16),
    "purple": (142, 68, 173), "teal": (26, 154, 148), "brown": (121, 85, 61),
}


def _build_count_color(n: int, force: bool) -> list[dict]:
    red_counts = _balanced_labels(list(range(0, 7)), n, "count_color")
    other_names = list(COUNT_COLOR_OTHERS)
    out_dir = IMAGES_ROOT / "count_color"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        red_count = red_counts[i]
        seed_key = f"{SEED}:count_color:{i}"
        lo, hi = max(0, 3 - red_count), 9 - red_count
        total = red_count + random.Random(f"{seed_key}:total").randint(lo, hi)

        rng, radii, centers = _place_balls(seed_key, total)
        color_labels = ["red"] * red_count + [rng.choice(other_names) for _ in range(total - red_count)]
        rng.shuffle(color_labels)  # decouple color-from-red from placement (largest-first) order
        colors_rgb = [RED_COLOR if c == "red" else COUNT_COLOR_OTHERS[c] for c in color_labels]
        bg_val, use_noise = _plain_or_noisy_background(rng)

        canvas2x = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), (bg_val, bg_val, bg_val))
        _draw_balls(canvas2x, centers, radii, colors_rgb)
        np_rng = np.random.default_rng(_seed_int(seed_key))
        noise_sigma = rng.uniform(4.0, 9.0) if use_noise else 0.0
        img = _finalize(canvas2x, np_rng, noise_sigma)

        item_id = f"count_color_{i:03d}"
        path = out_dir / f"{item_id}.png"
        if force or not path.exists():
            img.save(path, format="PNG")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "item_id": item_id, "file": f"{item_id}.png", "label": str(red_count), "sha256": sha256,
            "params": {
                "total": total, "red_count": red_count, "centers": [list(c) for c in centers], "radii": radii,
                "colors": color_labels, "background": "noise" if use_noise else "plain",
            },
        })
    _write_manifest("probes_count_color_source.jsonl", rows)
    return rows


# --- `spatial`: one red ball + one blue square, two yes/no items per image ------------------------------

SPATIAL_RED = (214, 26, 26)
SPATIAL_BLUE = (41, 98, 204)
SPATIAL_MIN_OFFSET = 40.0


def _spatial_scene(seed_key: str, label_h: bool, label_v: bool, max_attempts: int = 4000) -> dict[str, Any]:
    rng = random.Random(seed_key)
    word_h = rng.choice(["left", "right"])
    word_v = rng.choice(["above", "below"])
    # "left"/"above" true <=> the ball's centre has a smaller coordinate than the square's.
    ddx_negative = label_h if word_h == "left" else not label_h
    ddy_negative = label_v if word_v == "above" else not label_v

    square_side = rng.uniform(40.0, 60.0)
    disc_radius = rng.uniform(22.0, 34.0)
    half_sq = square_side / 2.0
    margin = 20.0

    for _ in range(max_attempts):
        mag_x = rng.uniform(SPATIAL_MIN_OFFSET, 160.0)
        mag_y = rng.uniform(SPATIAL_MIN_OFFSET, 110.0)
        ddx = -mag_x if ddx_negative else mag_x
        ddy = -mag_y if ddy_negative else mag_y

        sx = rng.uniform(half_sq + margin, WIDTH - half_sq - margin)
        sy = rng.uniform(half_sq + margin, HEIGHT - half_sq - margin)
        dx, dy = sx + ddx, sy + ddy
        if (disc_radius + margin <= dx <= WIDTH - disc_radius - margin
                and disc_radius + margin <= dy <= HEIGHT - disc_radius - margin):
            return {
                "word_h": word_h, "word_v": word_v, "square_center": (sx, sy), "square_side": square_side,
                "disc_center": (dx, dy), "disc_radius": disc_radius,
            }
    raise RuntimeError(f"probes: could not place spatial scene for seed_key={seed_key!r}")


def _build_spatial(n: int, force: bool) -> list[dict]:
    if n % 2 != 0:
        raise ValueError(f"probes.spatial needs an even n_per_suite (2 items/image), got {n}")
    n_images = n // 2
    labels_h = _balanced_labels([True, False], n_images, "spatial_h")
    labels_v = _balanced_labels([True, False], n_images, "spatial_v")
    out_dir = IMAGES_ROOT / "spatial"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_images):
        seed_key = f"{SEED}:spatial:{i}"
        scene = _spatial_scene(seed_key, labels_h[i], labels_v[i])

        canvas2x = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), (236, 236, 236))
        draw = ImageDraw.Draw(canvas2x)
        sx, sy = scene["square_center"]
        half = scene["square_side"] / 2.0
        draw.rectangle([(sx - half) * SCALE, (sy - half) * SCALE, (sx + half) * SCALE, (sy + half) * SCALE], fill=SPATIAL_BLUE)
        dx, dy = scene["disc_center"]
        r = scene["disc_radius"]
        draw.ellipse([(dx - r) * SCALE, (dy - r) * SCALE, (dx + r) * SCALE, (dy + r) * SCALE], fill=SPATIAL_RED)
        img = canvas2x.resize((WIDTH, HEIGHT), Image.LANCZOS)

        item_id_base = f"spatial_{i:03d}"
        path = out_dir / f"{item_id_base}.png"
        if force or not path.exists():
            img.save(path, format="PNG")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

        geom = {
            "disc_center": list(scene["disc_center"]), "square_center": list(scene["square_center"]),
            "disc_radius": scene["disc_radius"], "square_side": scene["square_side"],
        }
        question_h = f"Is the red ball to the {scene['word_h']} of the blue square in `img0`?"
        question_v = f"Is the red ball {scene['word_v']} the blue square in `img0`?"
        rows.append({
            "item_id": f"{item_id_base}_h", "file": f"{item_id_base}.png", "label": labels_h[i], "sha256": sha256,
            "params": {"axis": "horizontal", "question": question_h, "asked_word": scene["word_h"], **geom},
        })
        rows.append({
            "item_id": f"{item_id_base}_v", "file": f"{item_id_base}.png", "label": labels_v[i], "sha256": sha256,
            "params": {"axis": "vertical", "question": question_v, "asked_word": scene["word_v"], **geom},
        })
    _write_manifest("probes_spatial_source.jsonl", rows)
    return rows


# --- `largest`: four shapes, one enlarged by ratio r, label = colour of the largest ---------------------

LARGEST_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (219, 40, 40), "blue": (41, 98, 204), "green": (39, 174, 96), "yellow": (232, 196, 16),
}
SHAPE_NAMES = ("disc", "square", "triangle", "diamond")
RATIOS = (1.15, 1.3, 1.5, 2.0)
QUADRANTS = (
    (WIDTH * 0.25, HEIGHT * 0.25), (WIDTH * 0.75, HEIGHT * 0.25),
    (WIDTH * 0.25, HEIGHT * 0.75), (WIDTH * 0.75, HEIGHT * 0.75),
)


def _size_for_reach(shape: str, reach: float) -> float:
    """The linear measurement `_draw_shape` needs (radius / side / side / diagonal) so the shape's
    CIRCUMSCRIBED radius -- its farthest point from its own centre -- equals `reach` exactly.

    Sizing by circumscribed radius (not by literal shape area) is deliberate: a disc, an axis-aligned
    square, an equilateral triangle and a diamond (rotated square) are not equally "compact" for a given
    area -- e.g. at equal area a triangle's farthest point is ~55% farther from its centre than a
    square's. Sizing three shapes to the SAME literal area and one to `r * area` can then render the
    "largest" shape with a visibly SMALLER footprint than an unenlarged one (verified by eye while
    building this suite: an enlarged disc/square next to an unenlarged triangle looked smaller than the
    triangle at every ratio in RATIOS, including 2.0). Equalizing the circumscribed radius instead means
    no point of a baseline shape is ever farther from its centre than `reach`, while the enlarged shape's
    farthest point is strictly farther (`reach * sqrt(r)` for ratio r) -- so the enlarged shape is
    unambiguously the biggest by construction, at every ratio, regardless of which shape it is. Area
    still scales by exactly r *within a shape* (area is quadratic in every linear measurement of a
    fixed-proportion 2D shape), and `_build_largest` reports `area = pi * reach**2` uniformly (the
    "disc-equivalent" nominal area) so `three have the same area A` / `one has area ratio r * A` both
    still hold exactly on the stored ground truth, and the largest-by-area shape is always the
    largest-by-reach (hence largest-looking) shape too."""
    if shape == "disc":
        return reach
    if shape == "square":  # axis-aligned; circumradius = half-side * sqrt(2)
        return reach * math.sqrt(2.0)
    if shape == "triangle":  # equilateral; circumradius (centroid to any vertex) = side / sqrt(3)
        return reach * math.sqrt(3.0)
    if shape == "diamond":  # rhombus with equal diagonals d; circumradius (centre to vertex) = d / 2
        return 2.0 * reach
    raise ValueError(shape)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, center: tuple[float, float], size_final: float, color: tuple[int, int, int]) -> None:
    cx, cy = center[0] * SCALE, center[1] * SCALE
    if shape == "disc":
        r = size_final * SCALE
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        half = size_final * SCALE / 2.0
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)
    elif shape == "triangle":
        side = size_final * SCALE
        h = side * math.sqrt(3.0) / 2.0
        apex = (cx, cy - 2.0 * h / 3.0)
        base_l = (cx - side / 2.0, cy + h / 3.0)
        base_r = (cx + side / 2.0, cy + h / 3.0)
        draw.polygon([apex, base_l, base_r], fill=color)
    elif shape == "diamond":
        d = size_final * SCALE
        draw.polygon([(cx, cy - d / 2.0), (cx + d / 2.0, cy), (cx, cy + d / 2.0), (cx - d / 2.0, cy)], fill=color)
    else:
        raise ValueError(shape)


def _build_largest(n: int, force: bool) -> list[dict]:
    ratios = _balanced_labels(list(RATIOS), n, "largest_ratio")
    big_colors = _balanced_labels(list(LARGEST_COLORS), n, "largest_color")
    out_dir = IMAGES_ROOT / "largest"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        seed_key = f"{SEED}:largest:{i}"
        rng = random.Random(seed_key)
        r = ratios[i]
        big_color = big_colors[i]

        # All four shapes of an image are the SAME kind (disc, square, ...), so "largest" is unambiguous: with different kinds
        # a shape can have the larger area and the smaller visual extent at once (reviewed 2026-09-21, lab/NOTES.md entry 50b).
        one_kind = rng.choice(SHAPE_NAMES)
        shape_order = [one_kind] * 4
        big_slot = rng.randrange(4)
        big_shape = one_kind
        other_colors = [c for c in LARGEST_COLORS if c != big_color]
        rng.shuffle(other_colors)
        slot_colors = other_colors[:big_slot] + [big_color] + other_colors[big_slot:]

        base_reach = rng.uniform(16.0, 26.0)  # shared circumscribed radius of the 3 baseline shapes

        canvas2x = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), (240, 240, 240))
        draw = ImageDraw.Draw(canvas2x)
        shapes_meta = []
        for qi, shape in enumerate(shape_order):
            qx, qy = QUADRANTS[qi]
            center = (qx + rng.uniform(-20.0, 20.0), qy + rng.uniform(-15.0, 15.0))
            reach = base_reach * math.sqrt(r) if qi == big_slot else base_reach  # same kind of shape: area ratio is exactly r
            size_final = _size_for_reach(shape, reach)
            area = math.pi * reach**2  # nominal (disc-equivalent) area; see _size_for_reach's docstring
            color_name = slot_colors[qi]
            _draw_shape(draw, shape, center, size_final, LARGEST_COLORS[color_name])
            shapes_meta.append({
                "shape": shape, "color": color_name, "area": area, "reach": reach, "size": size_final, "center": list(center),
            })
        img = canvas2x.resize((WIDTH, HEIGHT), Image.LANCZOS)

        item_id = f"largest_{i:03d}"
        path = out_dir / f"{item_id}.png"
        if force or not path.exists():
            img.save(path, format="PNG")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "item_id": item_id, "file": f"{item_id}.png", "label": big_color, "sha256": sha256,
            "params": {"ratio": r, "largest_color": big_color, "largest_shape": big_shape, "shapes": shapes_meta},
        })
    _write_manifest("probes_largest_source.jsonl", rows)
    return rows


# --- `stripes`: full-frame stripes at one of 4 orientations ---------------------------------------------

ORIENTATIONS = ("horizontal", "vertical", "diagonal_rising", "diagonal_falling")


def _stripe_colors(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int], float]:
    hue = rng.uniform(0.0, 1.0)
    sat = rng.uniform(0.35, 0.75)
    val1 = rng.uniform(0.55, 0.88)
    contrast = rng.uniform(0.08, 0.85)  # subtle .. strong
    val2 = max(0.04, val1 - contrast * 0.7)
    if rng.random() < 0.5:
        val_a, val_b = val1, val2
    else:
        val_a, val_b = val2, val1

    def to_rgb(v: float) -> tuple[int, int, int]:
        red, green, blue = colorsys.hsv_to_rgb(hue, sat, v)
        return (round(red * 255), round(green * 255), round(blue * 255))

    return to_rgb(val_a), to_rgb(val_b), contrast


def _render_stripes(orientation: str, period_px: float, color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> Image.Image:
    w2, h2 = WIDTH * SCALE, HEIGHT * SCALE
    period2 = period_px * SCALE
    ys, xs = np.mgrid[0:h2, 0:w2]
    if orientation == "horizontal":
        u = ys
    elif orientation == "vertical":
        u = xs
    elif orientation == "diagonal_rising":  # "/" -- rises to the right
        u = xs + ys
    elif orientation == "diagonal_falling":  # "\" -- falls to the right
        u = xs - ys
    else:
        raise ValueError(orientation)
    band = (np.floor(u / period2).astype(np.int64)) % 2
    palette = np.array([color_a, color_b], dtype=np.uint8)
    arr = palette[band]
    return Image.fromarray(arr, mode="RGB")


def _build_stripes(n: int, force: bool) -> list[dict]:
    orientations = _balanced_labels(list(ORIENTATIONS), n, "stripes")
    out_dir = IMAGES_ROOT / "stripes"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        seed_key = f"{SEED}:stripes:{i}"
        rng = random.Random(seed_key)
        orientation = orientations[i]
        period_px = rng.uniform(16.0, 80.0)
        color_a, color_b, contrast = _stripe_colors(rng)

        canvas2x = _render_stripes(orientation, period_px, color_a, color_b)
        img = canvas2x.resize((WIDTH, HEIGHT), Image.LANCZOS)

        item_id = f"stripes_{i:03d}"
        path = out_dir / f"{item_id}.png"
        if force or not path.exists():
            img.save(path, format="PNG")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "item_id": item_id, "file": f"{item_id}.png", "label": orientation, "sha256": sha256,
            "params": {
                "orientation": orientation, "period_px": period_px,
                "color_a": list(color_a), "color_b": list(color_b), "contrast": contrast,
            },
        })
    _write_manifest("probes_stripes_source.jsonl", rows)
    return rows


# --- `text`: one word from a group of six look-alikes, printed centred ----------------------------------

WORD_GROUPS: list[list[str]] = [
    ["barn", "burn", "born", "bran", "barb", "bare"],
    ["cast", "cost", "cart", "coat", "cash", "case"],
    ["lane", "line", "lone", "lame", "late", "lake"],
    ["fort", "form", "fork", "fore", "ford", "fold"],
    ["mist", "most", "mast", "must", "mint", "mend"],
    ["ring", "king", "sing", "wing", "ping", "zing"],
]
assert len(WORD_GROUPS) >= 5
assert all(len(g) == 6 for g in WORD_GROUPS)
assert len({w for g in WORD_GROUPS for w in g}) == sum(len(g) for g in WORD_GROUPS)  # every word unique

_FONT_CANDIDATES = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


TEXT_BG = (238, 238, 238)
TEXT_COLOR = (30, 30, 40, 255)
TEXT_MARGIN_PX = 20  # final-resolution safety margin the rotated glyph box must stay inside


def _build_text(n: int, force: bool) -> list[dict]:
    group_indices = _balanced_labels(list(range(len(WORD_GROUPS))), n, "text_group")
    per_group_word_idx: dict[int, list[int]] = {
        gi: _balanced_labels(list(range(6)), group_indices.count(gi), f"text_word_{gi}") for gi in range(len(WORD_GROUPS))
    }
    cursor = {gi: 0 for gi in range(len(WORD_GROUPS))}
    out_dir = IMAGES_ROOT / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        seed_key = f"{SEED}:text:{i}"
        rng = random.Random(seed_key)
        np_rng = np.random.default_rng(_seed_int(seed_key))

        gi = group_indices[i]
        wi = per_group_word_idx[gi][cursor[gi]]
        cursor[gi] += 1
        word = WORD_GROUPS[gi][wi]
        font_size = rng.uniform(28.0, 72.0)
        rotation = rng.uniform(-8.0, 8.0)
        noise_sigma = rng.uniform(3.0, 9.0)

        w2, h2 = WIDTH * SCALE, HEIGHT * SCALE
        margin2x = TEXT_MARGIN_PX * SCALE
        cur_font_size = font_size
        rotated = None
        for _try in range(8):
            font = _font(max(1, round(cur_font_size * SCALE)))
            layer = Image.new("RGBA", (w2, h2), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            bbox = d.textbbox((0, 0), word, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = (w2 - tw) // 2 - bbox[0], (h2 - th) // 2 - bbox[1]
            d.text((tx, ty), word, font=font, fill=TEXT_COLOR)
            rotated = layer.rotate(rotation, resample=Image.BICUBIC, expand=False)
            alpha = np.asarray(rotated)[..., 3]
            ys, xs = np.nonzero(alpha)
            if len(xs) == 0 or (
                xs.min() >= margin2x and xs.max() <= w2 - margin2x and ys.min() >= margin2x and ys.max() <= h2 - margin2x
            ):
                break
            cur_font_size *= 0.85

        canvas2x = Image.new("RGBA", (w2, h2), (*TEXT_BG, 255))
        canvas2x = Image.alpha_composite(canvas2x, rotated).convert("RGB")
        img = _finalize(canvas2x, np_rng, noise_sigma)

        item_id = f"text_{i:03d}"
        path = out_dir / f"{item_id}.png"
        if force or not path.exists():
            img.save(path, format="PNG")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "item_id": item_id, "file": f"{item_id}.png", "label": word, "sha256": sha256,
            "params": {
                "word": word, "group_index": gi, "group": list(WORD_GROUPS[gi]),
                "font_size": font_size, "rotation": rotation, "noise_sigma": noise_sigma,
            },
        })
    _write_manifest("probes_text_source.jsonl", rows)
    return rows


# --- build / contact sheets -------------------------------------------------------------------------

SUITE_BUILDERS = {
    "count": _build_count, "count_color": _build_count_color, "spatial": _build_spatial,
    "largest": _build_largest, "stripes": _build_stripes, "text": _build_text,
}


def build(cfg: Config | None = None, n_per_suite: int = 150, force: bool = False) -> dict[str, list[dict]]:
    """Builds all six suites. `cfg` is accepted for interface parity with the other `lab/` builders
    (`ladders.build`, `semantic.build`) but unused: every image here is drawn from scratch, deterministic
    from the fixed module `SEED` and the item index alone, with no external dataset or config dependency."""
    del cfg
    return {name: builder(n_per_suite, force) for name, builder in SUITE_BUILDERS.items()}


def contact_sheet(suite: str, out_path: str | Path, n: int = 24) -> None:
    """A grid contact sheet of the first `n` rows of `suite`'s source manifest, each thumbnail labeled
    with its ground-truth label."""
    from ..logging_utils import read_jsonl

    source = EVAL_MANIFEST_DIR / f"probes_{suite}_source.jsonl"
    rows = read_jsonl(source)[:n]
    if not rows:
        raise RuntimeError(f"contact_sheet({suite!r}): no rows in {source}; run build() first")

    images_dir = IMAGES_ROOT / suite
    cell_w = 200
    pad, label_h = 6, 16
    cols = 6
    n_rows = math.ceil(len(rows) / cols)

    thumbs: list[tuple[Image.Image, str]] = []
    for row in rows:
        img = Image.open(images_dir / row["file"]).convert("RGB")
        cell_h = round(cell_w * img.height / img.width)
        thumbs.append((img.resize((cell_w, cell_h), Image.LANCZOS), str(row["label"])))
    cell_h = thumbs[0][0].height

    font = ImageFont.load_default()
    sheet_w = cols * (cell_w + pad) + pad
    sheet_h = n_rows * (cell_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (thumb, label) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x, y = pad + c * (cell_w + pad), pad + r * (cell_h + label_h + pad)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + cell_h + 2), label, fill="black", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=90)


if __name__ == "__main__":
    results = build()
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for name in SUITES:
        contact_sheet(name, SHEETS_DIR / f"{name}.jpg")
    for name, rows in results.items():
        print(f"{name}: n={len(rows)}")

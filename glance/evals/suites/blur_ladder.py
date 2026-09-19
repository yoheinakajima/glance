"""Synthetic Gaussian blur at 4 fixed strengths on Caltech-101 images that `caltech101` never uses.

Recast as a 4-level score. Levels are described as situations, never as numbers. Every image is first resized so
its longest side is BASE_SIDE, which makes one blur radius mean the same thing on every image.

The strengths below were frozen after rendering 8 examples and checking them against the level descriptions
(see STATUS.md, M3).
"""

from __future__ import annotations

from pathlib import Path

from ...config import Config
from . import caltech101
from .base import EvalItem, RawItem, SuiteInfo, materialize

INFO = SuiteInfo(
    name="blur_ladder", qtype="score",
    source="derived from Caltech-101 (https://data.caltech.edu/records/mzrjq-6wc02)",
    license="CC BY 4.0 (derived)",
)
BASE_SIDE = 384
BLUR_RADII = (0.0, 1.6, 4.0, 10.0)  # Gaussian radius in pixels at BASE_SIDE; frozen
INSTRUCTIONS = "How blurry is `img0`?"
LEVELS = [
    "Sharp: edges are crisp and fine texture is visible",
    "Slightly soft: edges are a little smooth and fine texture is lost, but the subject is obvious",
    "Clearly blurred: only larger shapes and colors are left, the subject is still recognizable",
    "Heavily blurred: only vague blobs of color, the subject is hard to identify",
]


def render(src: Path, level: int):
    from PIL import Image, ImageFilter

    img = Image.open(src).convert("RGB")
    scale = BASE_SIDE / max(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    radius = BLUR_RADII[level]
    return img.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else img


def build(cfg: Config, n: int) -> list[EvalItem]:
    _, files = caltech101.ordered_files(cfg)
    free = files[caltech101.reserved_count(cfg) :]
    raw_items = []
    for index, (cat, path) in enumerate(free[: 2 * cfg.eval.manifest_n]):
        level = index % len(BLUR_RADII)
        raw_items.append(
            RawItem(
                item_id=f"{cat}/{path.stem}_blur{level}",
                question={"type": "score", "instructions": INSTRUCTIONS, "criteria": LEVELS},
                label=level,
                write_image=lambda dest, src=path, level=level: render(src, level).save(dest, quality=92),
                meta={"blur_radius": BLUR_RADII[level], "category": cat},
            )
        )
    return materialize(cfg, INFO, raw_items, n)

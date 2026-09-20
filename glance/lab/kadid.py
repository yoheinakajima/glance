"""KADID-10k as a score-lab benchmark. EVALUATION ONLY (see DATASETS.md: no formal license, exception approved by the
project owner on 2026-09-20; nothing from the database is redistributed, only ids, levels, DMOS-derived metrics and
model logits are committed).

81 reference images x 25 distortions x 5 levels, with human DMOS (higher = better quality). Items are split by
REFERENCE image (seeded): 41 references for calibration, 40 for test, so no test content is seen during calibration.
The question is the same generic template for every distortion (lab/NOTES.md, entry 14); descriptions were written
from looking at the images and at mean DMOS per level, before any model output existed.

uv run python -m glance.lab.kadid      # writes lab/manifests_kadid/<key>.jsonl
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from ..config import PROJECT_ROOT

ROOT = PROJECT_ROOT / ".cache" / "datasets" / "kadid10k" / "kadid10k"
MANIFEST_DIR = PROJECT_ROOT / "lab" / "manifests_kadid"
SEED = 7
N_CALIBRATION_REFS = 41
INSTRUCTIONS_TEMPLATE = "How strong is the {description} in `img0`?"
LEVELS = ["Barely noticeable", "Slight", "Moderate", "Strong", "Very strong"]

# KADID-10k distortion index -> (key, description). 18 and 25 are NOT severity scales in KADID: their five levels
# sweep from one direction through "almost unchanged" to the other (mean DMOS per level is not monotone). They keep the
# generic question on purpose and are reported separately.
DISTORTIONS = {
    1: ("gaussian_blur", "Gaussian blur (even softness everywhere)"),
    2: ("lens_blur", "lens blur (out-of-focus softness with disc-shaped highlights)"),
    3: ("motion_blur", "motion blur (streaking in one direction)"),
    4: ("color_diffusion", "color diffusion (colors bleeding beyond their edges)"),
    5: ("color_shift", "color shift (color fringes displaced from the edges they belong to)"),
    6: ("color_quantization", "color quantization (too few distinct colors, visible banding)"),
    7: ("color_saturation_1", "desaturation (colors washed out toward gray, hues wrong at the extreme)"),
    8: ("color_saturation_2", "oversaturation (colors too vivid)"),
    9: ("jpeg2000", "JPEG 2000 compression artifacts (smearing and ringing)"),
    10: ("jpeg", "JPEG compression artifacts (square blocks and ringing)"),
    11: ("white_noise", "white noise (fine random speckle)"),
    12: ("color_noise", "color noise (random colored speckle)"),
    13: ("impulse_noise", "impulse noise (isolated black and white pixels)"),
    14: ("multiplicative_noise", "multiplicative noise (speckle that is stronger in bright areas)"),
    15: ("denoise", "denoising artifacts (mottled, smeared texture left after noise removal)"),
    16: ("brighten", "overexposure (too bright, highlights washed out)"),
    17: ("darken", "underexposure (too dark, shadows crushed)"),
    18: ("mean_shift", "brightness shift (the whole image uniformly lighter or darker)"),
    19: ("jitter", "pixel jitter (small random displacement of pixels, ragged edges)"),
    20: ("patch_shuffle", "non-eccentricity patches (small square patches copied to nearby wrong positions)"),
    21: ("pixelate", "pixelation (large square pixels)"),
    22: ("quantization", "luminance quantization (too few brightness levels, posterized look)"),
    23: ("color_block", "color blocks (random solid-colored squares pasted over the image)"),
    24: ("sharpen", "over-sharpening (halos and harsh edges)"),
    25: ("contrast_change", "contrast change (contrast too high or too low)"),
}
NOT_SEVERITY_SCALES = ("mean_shift", "contrast_change")
SCALES = {key: {"instructions": INSTRUCTIONS_TEMPLATE.format(description=desc), "levels": LEVELS, "kadid_index": idx}
          for idx, (key, desc) in DISTORTIONS.items()}


def reference_split() -> dict[str, str]:
    refs = sorted({f"I{i:02d}" for i in range(1, 82)})
    random.Random(f"{SEED}:kadid:refs").shuffle(refs)
    return {ref: ("calibration" if i < N_CALIBRATION_REFS else "test") for i, ref in enumerate(refs)}


def build() -> dict[str, int]:
    split = reference_split()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_key: dict[str, list[dict]] = {key: [] for key in SCALES}
    with open(ROOT / "dmos.csv") as f:
        for row in csv.DictReader(f):
            ref, dist, level = row["dist_img"].removesuffix(".png").split("_")
            key = DISTORTIONS[int(dist)][0]
            path = ROOT / "images" / row["dist_img"]
            rows_by_key[key].append({
                "item_id": row["dist_img"].removesuffix(".png"), "ladder": key, "source_image_id": ref,
                "level": int(level) - 1, "split": split[ref], "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dmos": float(row["dmos"]), "dmos_var": float(row["var"]),
            })
    for key, rows in rows_by_key.items():
        rows.sort(key=lambda r: r["item_id"])
        (MANIFEST_DIR / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {key: len(rows) for key, rows in rows_by_key.items()}


if __name__ == "__main__":
    counts = build()
    split = reference_split()
    print(f"{sum(counts.values())} items over {len(counts)} distortions; "
          f"{sum(v == 'calibration' for v in split.values())} calibration / {sum(v == 'test' for v in split.values())} test references")

"""KADID-10k as a score-lab benchmark. EVALUATION ONLY (see DATASETS.md: no formal license, exception approved by the
project owner on 2026-09-20; nothing from the database is redistributed, only ids, levels, DMOS-derived metrics and
model logits are committed).

81 reference images x 25 distortions x 5 levels, with human DMOS (higher = better quality). Items are split by
REFERENCE image (seeded): 41 references for calibration, 40 for test, so no test content is seen during calibration.
The question is the same generic template for every distortion (lab/NOTES.md, entry 14); descriptions were written
from looking at the images and at mean DMOS per level, before any model output existed.

This repo does not ship the KADID-10k images or DMOS labels (see DATASETS.md). To build the manifests you first need
a local, unpacked copy of the database at `.cache/datasets/kadid10k/kadid10k/` (containing `dmos.csv` and `images/`).
Three ways to run this module:

uv run python -m glance.lab.kadid
    Build manifests from data that is already unpacked at `.cache/datasets/kadid10k/kadid10k/`. If that path is
    missing, this prints instructions instead of a manifest and exits with code 2.

uv run python -m glance.lab.kadid --zip PATH/TO/kadid10k.zip
    Verify the sha256 of a zip you downloaded yourself, unpack it to `.cache/datasets/kadid10k/`, then build
    manifests. A hash mismatch refuses to unpack and exits with code 2.

uv run python -m glance.lab.kadid --download --accept-evaluation-only
    Download the zip (about 3.07 GB) to `.cache/datasets/kadid10k/kadid10k.zip`, verify its sha256, unpack it, then
    build manifests. Without `--accept-evaluation-only` this refuses and prints the terms: KADID-10k has no formal
    license, it is offered "freely available to the research community" (Lin, Hosu, Saupe, QoMEX 2019), and this
    project uses it for evaluation only, never redistributes the images or DMOS, and cites the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

from ..config import PROJECT_ROOT

DATASET_DIR = PROJECT_ROOT / ".cache" / "datasets" / "kadid10k"
ROOT = DATASET_DIR / "kadid10k"
MANIFEST_DIR = PROJECT_ROOT / "lab" / "manifests_kadid"
SEED = 7

# Direct download link found on the authors' page (https://database.mmsp-kn.de/kadid-10k-database.html) by fetching
# it and reading the link target for "kadid10k.zip". Checked 2026-09-20: a HEAD request to this URL returned
# content-length 3067408471 bytes (about 3.07 GB), matching the size recorded in DATASETS.md.
ZIP_URL = "https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip"
ZIP_SHA256 = "fe59ace86a2525d5785ff011a2119fa88839e0329f5029f7b23994727efd185c"

EVALUATION_ONLY_TERMS = """\
KADID-10k has no formal license. The authors' page says it is "freely available to the research community" and
asks that the paper be cited (Lin, Hosu, Saupe, "KADID-10k: A Large-scale Artificially Distorted IQA Database",
QoMEX 2019). This project's use of it is an evaluation-only exception approved by the project owner: it is never
used to fit anything that ships, and the images and DMOS labels are never redistributed (see DATASETS.md).

Pass --accept-evaluation-only to confirm you agree to the same terms before downloading.
"""

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
                "ref_path": str((ROOT / "images" / f"{ref}.png").relative_to(PROJECT_ROOT)),
            })
    for key, rows in rows_by_key.items():
        rows.sort(key=lambda r: r["item_id"])
        (MANIFEST_DIR / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {key: len(rows) for key, rows in rows_by_key.items()}


# --- acquiring the data (not part of the public import surface; other modules only use SCALES/build/etc. above) ----


def _missing_data_message() -> str:
    return f"""\
KADID-10k data not found at {ROOT}.

This repo does not ship the KADID-10k images or DMOS labels (see DATASETS.md). To build the manifests, get a copy
of the database yourself, in one of two ways:

  1. Download kadid10k.zip yourself from https://database.mmsp-kn.de/kadid-10k-database.html, then run:
       uv run python -m glance.lab.kadid --zip PATH/TO/kadid10k.zip

  2. Let this tool download it for you (about 3.07 GB):
       uv run python -m glance.lab.kadid --download --accept-evaluation-only

Either way, the sha256 of the zip is verified ({ZIP_SHA256}) before it is unpacked.
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_and_unpack(zip_path: Path) -> bool:
    """Verify `zip_path` against ZIP_SHA256 and unpack it to DATASET_DIR. Returns False (and unpacks
    nothing) on a hash mismatch; the caller turns that into exit code 2."""
    print(f"verifying sha256 of {zip_path} ...", file=sys.stderr)
    actual = _sha256_file(zip_path)
    if actual != ZIP_SHA256:
        print(
            f"refusing to unpack {zip_path}: sha256 mismatch\n  expected {ZIP_SHA256}\n  got      {actual}",
            file=sys.stderr,
        )
        return False
    print(f"sha256 OK, unpacking to {DATASET_DIR} ...", file=sys.stderr)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASET_DIR)
    return True


def _download_zip() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATASET_DIR / "kadid10k.zip"
    print(f"downloading {ZIP_URL} -> {dest} ...", file=sys.stderr)
    report_every = 100 * 1024 * 1024  # ~100 MB
    next_report = report_every

    def _progress(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal next_report
        done = block_count * block_size
        if done >= next_report or (total_size > 0 and done >= total_size):
            mb = done / (1024 * 1024)
            if total_size > 0:
                print(f"  {mb:.0f} MiB / {total_size / (1024 * 1024):.0f} MiB", file=sys.stderr)
            else:
                print(f"  {mb:.0f} MiB", file=sys.stderr)
            next_report += report_every

    urllib.request.urlretrieve(ZIP_URL, dest, reporthook=_progress)
    return dest


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m glance.lab.kadid",
        description="Build KADID-10k score-lab manifests (evaluation only; see DATASETS.md).",
    )
    parser.add_argument(
        "--zip", type=Path, default=None, metavar="PATH",
        help="path to a kadid10k.zip you already downloaded; verified by sha256 and unpacked before building",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="download kadid10k.zip (about 3.07 GB) before building; requires --accept-evaluation-only",
    )
    parser.add_argument(
        "--accept-evaluation-only", action="store_true",
        help="confirm you agree to the evaluation-only terms printed with --download",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.zip is not None:
        if not _verify_and_unpack(args.zip):
            return 2
    elif args.download:
        if not args.accept_evaluation_only:
            print(EVALUATION_ONLY_TERMS, file=sys.stderr)
            return 2
        zip_path = _download_zip()
        if not _verify_and_unpack(zip_path):
            return 2
    elif not (ROOT / "dmos.csv").exists():
        print(_missing_data_message(), file=sys.stderr)
        return 2

    counts = build()
    split = reference_split()
    print(f"{sum(counts.values())} items over {len(counts)} distortions; "
          f"{sum(v == 'calibration' for v in split.values())} calibration / {sum(v == 'test' for v in split.values())} test references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

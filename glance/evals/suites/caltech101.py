"""Caltech-101 recast as a 101-option choice: "What is the main subject of `img0`?"

Source is the official CaltechDATA record (CC BY 4.0). The background clutter class is dropped, leaving the 101
object categories. The seeded order is shared with `blur_ladder`, which takes images this suite never uses.
"""

from __future__ import annotations

import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from ...config import Config
from .base import EvalItem, RawItem, SuiteInfo, materialize
from .sources import datasets_dir, download

INFO = SuiteInfo(
    name="caltech101", qtype="choice",
    source="https://data.caltech.edu/records/mzrjq-6wc02 (doi:10.22002/D1.20086)",
    license="CC BY 4.0",
)
ZIP_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
ZIP_MD5 = "3138e1922a9193bfa496528edbbc45d0"
BACKGROUND = "BACKGROUND_Google"
INSTRUCTIONS = "What is the main subject of `img0`?"
# Keys read fine as text once underscores become spaces, except for these.
DESCRIPTIONS = {
    "Faces": "a person's face with some background around it",
    "Faces_easy": "a person's face, tightly cropped",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
    "airplanes": "airplane",
    "car_side": "car seen from the side",
    "ketch": "ketch, a two-masted sailboat",
    "schooner": "schooner, a sailing ship",
    "gerenuk": "gerenuk, a long-necked antelope",
    "inline_skate": "inline skate",
    "snoopy": "Snoopy, the cartoon dog",
    "garfield": "Garfield, the cartoon cat",
    "yin_yang": "yin yang symbol",
    "stop_sign": "stop sign",
    "cougar_body": "cougar, whole body",
    "cougar_face": "cougar, face close-up",
    "crocodile_head": "crocodile, head close-up",
    "flamingo_head": "flamingo, head close-up",
    "wild_cat": "wild cat",
    "water_lilly": "water lily",
    "sea_horse": "seahorse",
    "dollar_bill": "dollar bill",
    "ceiling_fan": "ceiling fan",
    "cellphone": "cell phone",
    "soccer_ball": "soccer ball",
    "grand_piano": "grand piano",
    "electric_guitar": "electric guitar",
    "hawksbill": "hawksbill sea turtle",
    "ibis": "ibis, a wading bird",
    "okapi": "okapi",
    "metronome": "metronome",
    "minaret": "minaret, a mosque tower",
    "euphonium": "euphonium, a brass instrument",
    "binocular": "binoculars",
    "brontosaurus": "brontosaurus dinosaur",
    "stegosaurus": "stegosaurus dinosaur",
    "trilobite": "trilobite fossil",
    "nautilus": "nautilus shell",
    "joshua_tree": "Joshua tree",
    "windsor_chair": "Windsor chair",
}


def image_root(cfg: Config) -> Path:
    root = datasets_dir(cfg) / "caltech101"
    categories = root / "101_ObjectCategories"
    if not categories.is_dir():
        archive = download(ZIP_URL, root / "caltech-101.zip", md5=ZIP_MD5)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        with tarfile.open(root / "caltech-101" / "101_ObjectCategories.tar.gz") as tf:
            tf.extractall(root, filter="data")
    return categories


def ordered_files(cfg: Config) -> tuple[list[str], list[tuple[str, Path]]]:
    """(category keys, [(category, file)]) in the seeded order both Caltech-based suites share."""
    root = image_root(cfg)
    categories = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != BACKGROUND)
    files = [(cat, path) for cat in categories for path in sorted((root / cat).glob("*.jpg"))]
    random.Random(cfg.eval.seed).shuffle(files)
    return categories, files


def reserved_count(cfg: Config) -> int:
    """How many images at the head of the seeded order belong to `caltech101` (the rest are free for blur_ladder)."""
    return cfg.eval.manifest_n


def build(cfg: Config, n: int) -> list[EvalItem]:
    categories, files = ordered_files(cfg)
    criteria = {cat: DESCRIPTIONS.get(cat) for cat in categories}
    head = files[: reserved_count(cfg)]
    raw_items = [
        RawItem(
            item_id=f"{cat}/{path.name}",
            question={"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
            label=cat,
            write_image=lambda dest, src=path: shutil.copyfile(src, dest),
        )
        for cat, path in head
    ]
    # `head` is already in seeded order; materialize() shuffles again with the same seed, which is still deterministic.
    return materialize(cfg, INFO, raw_items, n)

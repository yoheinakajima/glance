"""Oxford-IIIT Pet, test split, recast as a 37-option choice: "Which breed is the animal in `img0`?" """

from __future__ import annotations

import io
from pathlib import Path

from ...config import Config
from .base import EvalItem, RawItem, SuiteInfo, materialize
from .sources import hf_parquet

REPO = "timm/oxford-iiit-pet"
REVISION = "089695c834a7deb60505b7cc506672db1c31a6aa"
INFO = SuiteInfo(
    name="pets37", qtype="choice",
    source=f"https://huggingface.co/datasets/{REPO} (test split)",
    license="CC BY-SA 4.0",
)
CAT_BREEDS = {
    "abyssinian", "bengal", "birman", "bombay", "british_shorthair", "egyptian_mau", "maine_coon", "persian",
    "ragdoll", "russian_blue", "siamese", "sphynx",
}
INSTRUCTIONS = "Which breed is the animal in `img0`?"


def build(cfg: Config, n: int) -> list[EvalItem]:
    import pyarrow.parquet as pq

    path = hf_parquet(cfg, REPO, REVISION, "data/test-00000-of-00001.parquet")
    table = pq.read_table(path, columns=["image_id", "label"])
    names = _label_names(path)
    keys = [name.lower().replace(" ", "_") for name in names]
    criteria = {
        key: f"{name.replace('_', ' ').title()}, a {'cat' if key in CAT_BREEDS else 'dog'} breed"
        for key, name in zip(keys, names)
    }
    rows = table.to_pylist()
    image_bytes: dict[str, bytes] = {}

    def write_image(dest: Path, image_id: str) -> None:
        from PIL import Image

        if not image_bytes:  # read the image column once, the first time an image has to be exported
            for row in pq.read_table(path, columns=["image_id", "image"]).to_pylist():
                image_bytes[str(row["image_id"])] = row["image"]["bytes"]
        Image.open(io.BytesIO(image_bytes[image_id])).convert("RGB").save(dest, quality=95)

    raw_items = [
        RawItem(
            item_id=str(r["image_id"]),
            question={"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
            label=keys[int(r["label"])],
            write_image=lambda dest, image_id=str(r["image_id"]): write_image(dest, image_id),
        )
        for r in sorted(rows, key=lambda r: str(r["image_id"]))
    ]
    return materialize(cfg, INFO, raw_items, n)


def _label_names(path: Path) -> list[str]:
    """Class names from the Hub features stored in the parquet metadata."""
    import json

    import pyarrow.parquet as pq

    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]

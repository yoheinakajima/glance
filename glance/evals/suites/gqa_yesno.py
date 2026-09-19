"""GQA testdev-balanced, yes/no questions only, recast as noul with the question text as instructions.

No `criteria` are given, so the dual encoder cannot answer these (it needs a caption for `true`); this suite runs on
the VLM and the frontier baseline only.
"""

from __future__ import annotations

import io
from pathlib import Path

from ...config import Config
from .base import EvalItem, RawItem, SuiteInfo, materialize
from .sources import hf_parquet

REPO = "lmms-lab/GQA"
REVISION = "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
INFO = SuiteInfo(
    name="gqa_yesno", qtype="noul",
    source=f"https://huggingface.co/datasets/{REPO} (testdev_balanced)",
    license="MIT on the Hub card; CC BY 4.0 upstream (GQA / Visual Genome)",
    backends=("vlm", "frontier"),
)


def build(cfg: Config, n: int) -> list[EvalItem]:
    import pyarrow.parquet as pq

    questions = pq.read_table(
        hf_parquet(cfg, REPO, REVISION, "testdev_balanced_instructions/testdev-00000-of-00001.parquet")
    ).to_pylist()
    images_path = hf_parquet(cfg, REPO, REVISION, "testdev_balanced_images/testdev-00000-of-00001.parquet")
    image_bytes: dict[str, bytes] = {}

    def load_images() -> None:
        if not image_bytes:
            for row in pq.read_table(images_path).to_pylist():
                image_bytes[str(row["id"])] = row["image"]["bytes"]

    raw_items: list[RawItem] = []
    for row in sorted(questions, key=lambda r: str(r["id"])):
        answer = str(row["answer"]).strip().lower()
        if answer not in ("yes", "no"):
            continue
        image_id = str(row["imageId"])

        def write_image(dest: Path, image_id: str = image_id) -> None:
            from PIL import Image

            load_images()
            Image.open(io.BytesIO(image_bytes[image_id])).convert("RGB").save(dest, quality=95)

        raw_items.append(
            RawItem(
                item_id=str(row["id"]),
                question={"type": "noul", "instructions": f"About `img0`: {row['question']}"},
                label=answer == "yes",
                write_image=write_image,
                meta={"gqa_image": image_id, "question": row["question"]},
            )
        )
    return materialize(cfg, INFO, raw_items, n)

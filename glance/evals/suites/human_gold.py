"""Hand-labeled local images. Private: never leaves the machine, never committed.

One JSON object per line in the file named by `paths.human_gold`:

{"image": "gold/img_0001.jpg", "question": {"type": "choice", "instructions": "...", "criteria": {...}},
 "label": "receipt", "annotators": {"a1": "receipt", "a2": "invoice"}}

`label` is true/false for noul, an option key for choice, a level index for score. `annotators` is optional and
feeds the human-disagreement column. Questions may be of any type, so metrics are reported per type.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ...config import PROJECT_ROOT, Config
from ...logging_utils import read_jsonl
from ...schema import DecideRequest
from .base import EvalItem, RawItem, SuiteInfo, SuiteSkipped, materialize

INFO = SuiteInfo(
    name="human_gold", qtype="any", source="local JSONL (paths.human_gold)", license="private", private=True
)


def build(cfg: Config, n: int) -> list[EvalItem]:
    gold_path = cfg.path("human_gold")
    rows = read_jsonl(gold_path)
    if not rows:
        raise SuiteSkipped(f"no hand-labeled items at {gold_path}")
    raw_items = []
    for index, row in enumerate(rows):
        src = Path(row["image"])
        src = src if src.is_absolute() else PROJECT_ROOT / src
        if not src.is_file():
            raise FileNotFoundError(f"human_gold row {index}: no image at {row['image']}")
        # Validate the question with the same schema the API uses.
        DecideRequest.model_validate(
            {"model": "vlm", "state": {"images": [{"id": "img0", "path": str(src)}]}, "questions": {"q": row["question"]}}
        )
        annotators = row.get("annotators") or {}
        votes = list(annotators.values())
        disagreement = None if len(votes) < 2 else 1.0 - max(votes.count(v) for v in votes) / len(votes)
        raw_items.append(
            RawItem(
                item_id=f"gold_{index:05d}", question=row["question"], label=row["label"], ext=src.suffix.lower(),
                write_image=lambda dest, src=src: shutil.copyfile(src, dest),
                meta={"annotators": annotators, "human_disagreement": disagreement, "source_image": row["image"]},
            )
        )
    # The manifest sits next to the gold file, outside git.
    return materialize(cfg, INFO, raw_items, n, manifest_path=gold_path.parent / "human_gold.manifest.jsonl")

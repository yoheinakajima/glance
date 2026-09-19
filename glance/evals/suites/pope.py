"""POPE (all 3 splits) on COCO val2014 images, recast as noul: "Is there a {object} in `img0`?"

Questions come from the official MIT-licensed repo at a pinned commit. Only the COCO images that the selected
items need are fetched, from COCO's own host.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...config import Config
from .base import EvalItem, RawItem, SuiteInfo, materialize
from .sources import datasets_dir, download

INFO = SuiteInfo(
    name="pope", qtype="noul",
    source="https://github.com/RUCAIBox/POPE (questions), http://images.cocodataset.org/val2014 (images)",
    license="MIT (questions); COCO terms of use (images)",
)
COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
SPLITS = ("random", "popular", "adversarial")
QUESTION_URL = "https://raw.githubusercontent.com/RUCAIBox/POPE/{commit}/output/coco/coco_pope_{split}.json"
IMAGE_URL = "http://images.cocodataset.org/val2014/{name}"
PATTERN = re.compile(r"^Is there (an? .+) in the (?:image|imange)\?$")


def build(cfg: Config, n: int) -> list[EvalItem]:
    root = datasets_dir(cfg) / "pope"
    raw_items: list[RawItem] = []
    for split in SPLITS:
        path = download(QUESTION_URL.format(commit=COMMIT, split=split), root / f"coco_pope_{split}.json", quiet=True)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            match = PATTERN.match(row["text"].strip())
            if not match:
                raise ValueError(f"unexpected POPE question: {row['text']!r}")
            thing = match.group(1)
            noun = thing.split(" ", 1)[1]
            name = row["image"]

            def write_image(dest: Path, name: str = name) -> None:
                download(IMAGE_URL.format(name=name), dest, quiet=True)

            raw_items.append(
                RawItem(
                    item_id=f"{split}_{row['question_id']}",
                    question={
                        "type": "noul",
                        "instructions": f"Is there {thing} in `img0`?",
                        "criteria": {"true": f"a photo with {thing} in it", "false": f"a photo with no {noun} in it"},
                    },
                    label=row["label"].strip().lower() == "yes",
                    write_image=write_image,
                    meta={"pope_split": split, "object": noun, "coco_image": name},
                )
            )
    return materialize(cfg, INFO, raw_items, n)

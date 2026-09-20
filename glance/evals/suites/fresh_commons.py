"""Fresh real photos with labels nobody on this project made (opt-in suites `fresh_choice` and `fresh_yesno`).

Source: Wikimedia Commons files taken after the evaluated models were released, own work, CC0 / CC BY / CC BY-SA, whose
structured data (property P180, "depicts", set by uploaders and editors) names one of 13 everyday classes. Built by
`tools/fetch_fresh_depicts.py`; labels and attributions are in `manifests/fresh_commons_source.jsonl`. The images are
public, so the frontier baseline may see them; they are not redistributed here.

- `fresh_choice`: "What is the main subject of `img0`?" with the 13 classes as options.
- `fresh_yesno`: for every photo one question about its own class (answer yes) and one about a seeded other class
  (answer no), POPE-style. A negative can be wrong when the other object happens to be in the frame; that noise, like a
  "dog" tag on a dog statue, is the same for every system and is reported as a caveat.
"""

from __future__ import annotations

import random
import shutil
from types import SimpleNamespace

from ...config import PROJECT_ROOT, Config
from ...logging_utils import read_jsonl
from .base import MANIFEST_DIR, EvalItem, RawItem, SuiteInfo, SuiteSkipped, materialize

SOURCE = MANIFEST_DIR / "fresh_commons_source.jsonl"
IMAGES = PROJECT_ROOT / ".cache" / "datasets" / "fresh_commons" / "images"
CLASSES = {"dog": "A dog", "cat": "A cat", "bird": "A bird", "horse": "A horse", "bicycle": "A bicycle", "car": "A car", "train": "A train",
           "boat": "A boat", "bridge": "A bridge", "church": "A church building", "mountain": "A mountain", "beach": "A beach", "flower": "A flower"}
ARTICLE = {k: v[0].lower() + v[1:] for k, v in CLASSES.items()}


def _rows() -> list[dict]:
    rows = read_jsonl(SOURCE)
    if not rows:
        raise SuiteSkipped("no fresh Commons manifest; run `uv run python tools/fetch_fresh_depicts.py`")
    if not (IMAGES / rows[0]["file"]).is_file():
        raise SuiteSkipped("fresh Commons images are missing; run `uv run python tools/fetch_fresh_depicts.py` to fetch them again")
    return rows


def _raw(item_id: str, row: dict, question: dict, label) -> RawItem:
    src = IMAGES / row["file"]
    return RawItem(item_id=item_id, question=question, label=label, ext=".jpg", write_image=lambda dest, src=src: shutil.copyfile(src, dest),
                   meta={"title": row["title"], "license": row["license"], "taken": row["taken"], "depicts": row["label"]})


def _choice(cfg: Config, n: int) -> list[EvalItem]:
    question = {"type": "choice", "instructions": "What is the main subject of `img0`?", "criteria": dict(CLASSES)}
    return materialize(cfg, CHOICE_INFO, [_raw(r["item_id"], r, question, r["label"]) for r in _rows()], n)


def _yesno(cfg: Config, n: int) -> list[EvalItem]:
    rng = random.Random(f"{cfg.eval.seed}:fresh_yesno")
    items = []
    for r in _rows():
        other = rng.choice([k for k in CLASSES if k != r["label"]])
        for key, label in ((r["label"], True), (other, False)):
            question = {"type": "noul", "instructions": f"Is there {ARTICLE[key]} in `img0`?"}
            items.append(_raw(f"{r['item_id']}__{key}", r, question, label))
    return materialize(cfg, YESNO_INFO, items, n)


_SOURCE = "Wikimedia Commons, photos taken after the models' release, labels from structured 'depicts' statements"
CHOICE_INFO = SuiteInfo(name="fresh_choice", qtype="choice", source=_SOURCE, license="CC0 / CC BY / CC BY-SA per file (see manifest)",
                        backends=("siglip", "vlm", "frontier"))
YESNO_INFO = SuiteInfo(name="fresh_yesno", qtype="noul", source=_SOURCE, license="CC0 / CC BY / CC BY-SA per file (see manifest)",
                       backends=("vlm", "frontier"))
MODULES = {"fresh_choice": SimpleNamespace(INFO=CHOICE_INFO, build=_choice), "fresh_yesno": SimpleNamespace(INFO=YESNO_INFO, build=_yesno)}

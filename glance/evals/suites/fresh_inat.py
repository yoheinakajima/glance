"""Fresh real photos with CLEANER labels than fresh_commons (opt-in suites `inat_choice` and `inat_yesno`).

Source: iNaturalist research-grade observations made after the evaluated models' release, CC0 / CC BY / CC BY-SA.
"Research grade" means at least two identifiers agreed, so unlike Commons' "depicts" (which can mean "appears
somewhere in the frame"), the photo IS of the identified organism. Labels are the community identification's
ICONIC TAXON (bird, insect, plant, ...), not the species. Built by `tools/fetch_fresh_inat.py`; labels and
attributions are in `manifests/fresh_inat_source.jsonl`. The images are public, so the frontier baseline may see
them; they are not redistributed here.

- `inat_choice`: "What kind of organism is the main subject of `img0`?" with the 10 iconic taxa as options.
- `inat_yesno`: for every photo one question about its own class (answer yes) and one about a seeded other class
  (answer no), POPE-style.
"""

from __future__ import annotations

import random
import shutil
from types import SimpleNamespace

from ...config import PROJECT_ROOT, Config
from ...logging_utils import read_jsonl
from .base import MANIFEST_DIR, EvalItem, RawItem, SuiteInfo, SuiteSkipped, materialize

SOURCE = MANIFEST_DIR / "fresh_inat_source.jsonl"
IMAGES = PROJECT_ROOT / ".cache" / "datasets" / "fresh_inat" / "images"
CLASSES = {
    "bird": "A bird", "insect": "An insect (including butterflies, bees, beetles, flies)",
    "plant": "A plant (flower, tree, fern, moss, grass)", "mammal": "A mammal", "reptile": "A reptile (lizard, snake, turtle)",
    "amphibian": "An amphibian (frog, toad, salamander)", "fish": "A fish", "fungus": "A fungus or lichen (mushroom)",
    "arachnid": "A spider, scorpion, tick or other arachnid", "mollusc": "A snail, slug, clam, octopus or other mollusc",
}
ARTICLE = {
    "bird": "a bird", "insect": "an insect", "plant": "a plant", "mammal": "a mammal", "reptile": "a reptile",
    "amphibian": "an amphibian", "fish": "a fish", "fungus": "a fungus", "arachnid": "a spider or other arachnid",
    "mollusc": "a snail or other mollusc",
}


def _rows() -> list[dict]:
    rows = read_jsonl(SOURCE)
    if not rows:
        raise SuiteSkipped("no fresh iNaturalist manifest; run `uv run python tools/fetch_fresh_inat.py`")
    if not (IMAGES / rows[0]["file"]).is_file():
        raise SuiteSkipped("fresh iNaturalist images are missing; run `uv run python tools/fetch_fresh_inat.py` to fetch them again")
    return rows


def _raw(item_id: str, row: dict, question: dict, label) -> RawItem:
    src = IMAGES / row["file"]
    return RawItem(
        item_id=item_id, question=question, label=label, ext=".jpg", write_image=lambda dest, src=src: shutil.copyfile(src, dest),
        meta={"page": row["page"], "photo_license": row["photo_license"], "observed_on": row["observed_on"],
              "taxon_name": row["taxon_name"], "label": row["label"]},
    )


def _choice(cfg: Config, n: int) -> list[EvalItem]:
    question = {"type": "choice", "instructions": "What kind of organism is the main subject of `img0`?", "criteria": dict(CLASSES)}
    return materialize(cfg, CHOICE_INFO, [_raw(r["item_id"], r, question, r["label"]) for r in _rows()], n)


def _yesno(cfg: Config, n: int) -> list[EvalItem]:
    rng = random.Random(f"{cfg.eval.seed}:inat_yesno")
    items = []
    for r in _rows():
        other = rng.choice([k for k in CLASSES if k != r["label"]])
        for key, label in ((r["label"], True), (other, False)):
            question = {"type": "noul", "instructions": f"Is the main subject of `img0` {ARTICLE[key]}?"}
            items.append(_raw(f"{r['item_id']}__{key}", r, question, label))
    return materialize(cfg, YESNO_INFO, items, n)


_SOURCE = "iNaturalist research-grade observations made after the models' release; labels are the community identification's iconic taxon"
CHOICE_INFO = SuiteInfo(name="inat_choice", qtype="choice", source=_SOURCE, license="CC0 / CC BY / CC BY-SA per photo (see manifest)",
                        backends=("siglip", "vlm", "frontier"))
YESNO_INFO = SuiteInfo(name="inat_yesno", qtype="noul", source=_SOURCE, license="CC0 / CC BY / CC BY-SA per photo (see manifest)",
                       backends=("vlm", "frontier"))
MODULES = {"inat_choice": SimpleNamespace(INFO=CHOICE_INFO, build=_choice), "inat_yesno": SimpleNamespace(INFO=YESNO_INFO, build=_yesno)}

"""A HARDER fresh photo test than fresh_inat.py (lab/NOTES.md entry 47; opt-in suites `inat_orders_choice` and
`inat_orders_yesno`): the same iNaturalist research-grade rules, but classes are INSECT ORDERS instead of iconic
taxa, so every class looks like the others (a beetle vs a true bug vs a fly).

Source: iNaturalist research-grade observations made after the evaluated models' release, CC0 / CC BY / CC BY-SA.
Built by `tools/fetch_fresh_inat_orders.py`, which selects observations by iNaturalist's `taxon_id` filter (the
order id, resolved live from the API, matching the observation's taxon or any of its ancestors) instead of
`iconic_taxa`; labels and attributions are in `manifests/fresh_inat_orders_source.jsonl`. The images are public,
so the frontier baseline may see them; they are not redistributed here.

- `inat_orders_choice`: "What kind of insect is the main subject of `img0`?" with the seven orders as options.
- `inat_orders_yesno`: for every photo one question about its own order (answer yes) and one about a seeded other
  order from the same seven (answer no), so every negative is a look-alike, not a fish.
"""

from __future__ import annotations

import random
import shutil
from types import SimpleNamespace

from ...config import PROJECT_ROOT, Config
from ...logging_utils import read_jsonl
from .base import MANIFEST_DIR, EvalItem, RawItem, SuiteInfo, SuiteSkipped, materialize

SOURCE = MANIFEST_DIR / "fresh_inat_orders_source.jsonl"
IMAGES = PROJECT_ROOT / ".cache" / "datasets" / "fresh_inat_orders" / "images"
CLASSES = {
    "beetle": "A beetle (hard wing covers meeting in a straight line down the back)",
    "butterfly_or_moth": "A butterfly or moth (scaled wings), including caterpillars",
    "bee_wasp_or_ant": "A bee, wasp or ant",
    "fly": "A fly, mosquito or midge (one pair of wings)",
    "dragonfly_or_damselfly": "A dragonfly or damselfly",
    "true_bug": "A true bug (shield bug, cicada, aphid, leafhopper, water strider)",
    "grasshopper_or_cricket": "A grasshopper, cricket or katydid",
}
ARTICLE = {
    "beetle": "a beetle",
    "butterfly_or_moth": "a butterfly or moth",
    "bee_wasp_or_ant": "a bee, wasp or ant",
    "fly": "a fly",
    "dragonfly_or_damselfly": "a dragonfly or damselfly",
    "true_bug": "a true bug",
    "grasshopper_or_cricket": "a grasshopper or cricket",
}


def _rows() -> list[dict]:
    rows = read_jsonl(SOURCE)
    if not rows:
        raise SuiteSkipped("no fresh iNaturalist orders manifest; run `uv run python tools/fetch_fresh_inat_orders.py`")
    if not (IMAGES / rows[0]["file"]).is_file():
        raise SuiteSkipped("fresh iNaturalist orders images are missing; run `uv run python tools/fetch_fresh_inat_orders.py` to fetch them again")
    return rows


def _raw(item_id: str, row: dict, question: dict, label) -> RawItem:
    src = IMAGES / row["file"]
    return RawItem(
        item_id=item_id, question=question, label=label, ext=".jpg", write_image=lambda dest, src=src: shutil.copyfile(src, dest),
        meta={"page": row["page"], "photo_license": row["photo_license"], "observed_on": row["observed_on"],
              "taxon_name": row["taxon_name"], "label": row["label"], "order_name": row["order_name"]},
    )


def _choice(cfg: Config, n: int) -> list[EvalItem]:
    question = {"type": "choice", "instructions": "What kind of insect is the main subject of `img0`?", "criteria": dict(CLASSES)}
    return materialize(cfg, CHOICE_INFO, [_raw(r["item_id"], r, question, r["label"]) for r in _rows()], n)


def _yesno(cfg: Config, n: int) -> list[EvalItem]:
    rng = random.Random(f"{cfg.eval.seed}:inat_orders_yesno")
    items = []
    for r in _rows():
        other = rng.choice([k for k in CLASSES if k != r["label"]])
        for key, label in ((r["label"], True), (other, False)):
            question = {"type": "noul", "instructions": f"Is the main subject of `img0` {ARTICLE[key]}?"}
            items.append(_raw(f"{r['item_id']}__{key}", r, question, label))
    return materialize(cfg, YESNO_INFO, items, n)


_SOURCE = "iNaturalist research-grade observations made after the models' release; labels are the insect ORDER of the community identification"
CHOICE_INFO = SuiteInfo(name="inat_orders_choice", qtype="choice", source=_SOURCE, license="CC0 / CC BY / CC BY-SA per photo (see manifest)",
                        backends=("siglip", "vlm", "frontier"))
YESNO_INFO = SuiteInfo(name="inat_orders_yesno", qtype="noul", source=_SOURCE, license="CC0 / CC BY / CC BY-SA per photo (see manifest)",
                       backends=("vlm", "frontier"))
MODULES = {"inat_orders_choice": SimpleNamespace(INFO=CHOICE_INFO, build=_choice), "inat_orders_yesno": SimpleNamespace(INFO=YESNO_INFO, build=_yesno)}

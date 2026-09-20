"""Stretch: open-set check. Drop 7 breeds from the `pets37` options, add `other`, and see how often the held-out
breeds land on `other`. Same images and seeded order as `pets37`; not part of the default eval."""

from __future__ import annotations

import random

from ...config import Config
from . import pets37
from .base import EvalItem, SuiteInfo

INFO = SuiteInfo(
    name="pets37_openset", qtype="choice", source=pets37.INFO.source + ", 7 breeds held out", license=pets37.INFO.license,
)
HELD_OUT = 7


def build(cfg: Config, n: int) -> list[EvalItem]:
    items = pets37.build(cfg, n)
    breeds = list(items[0].question["criteria"])
    held_out = set(random.Random(f"{cfg.eval.seed}:openset").sample(breeds, HELD_OUT))
    criteria = {k: v for k, v in items[0].question["criteria"].items() if k not in held_out}
    criteria["other"] = "Some other breed that is not listed"
    out = []
    for item in items:
        was_held_out = item.label in held_out
        out.append(EvalItem(
            suite=INFO.name, item_id=item.item_id, split=item.split, image_path=item.image_path,
            image_sha256=item.image_sha256, question={**item.question, "criteria": criteria},
            label="other" if was_held_out else item.label, meta={"held_out": was_held_out, "true_breed": item.label},
        ))
    return out

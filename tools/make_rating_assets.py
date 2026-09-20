"""Writes glance/assets/ratings/*.json: the `ens4d` calibrations for the five lab rubrics, shipped with the package.

Fit on the CALIBRATION split of the score lab only (500 labeled images per rubric, `lab/data/main_stagesAB.jsonl.gz`),
with the same matrix scaling the lab reported on its test split. They are valid for exactly the rubric text in
`glance/lab/ladders.py`, Qwen3-VL-4B-Instruct at the pinned revision, and the 768 image-token budget; any other rubric
or configuration needs its own `glance fit`. Only our own photos and model logits went into them.

uv run python tools/make_rating_assets.py
"""
import json
import pathlib

import numpy as np

from glance import rating
from glance.config import load_config
from glance.lab.ladders import LADDERS
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

cfg = load_config()
spec = next(s for s in cfg.models.vlm_tiers.values() if s.id == MODEL_ID)
rows = [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if r["split"] == "calibration"]
members = rating.MEMBERS["ens4d"]
index = []
for scale, meta in LADDERS.items():
    per = {m: {r["item_id"]: r for r in rows if r["ladder"] == scale and r["method_key"] == m} for m in members}
    ids = sorted(set.intersection(*(set(d) for d in per.values())))
    x = np.array([sum((per[m][i]["logits"] for m in members), []) for i in ids])
    y = np.array([per[members[0]][i]["level"] for i in ids])
    key = rating.RatingKey(backend="vlm", model=f"{spec.id}@{spec.revision}", prompt_version=rating.RATING_PROMPT_VERSION,
                           score_method="ens4d", image_token_budget=spec.image_token_budget,
                           instructions=meta["instructions"], criteria=tuple(meta["levels"]))
    cal = rating.build_calibration(key, x, y, name=f"lab/{scale}",
                                   source="score lab calibration split (lab/data/main_stagesAB.jsonl.gz), prefix-cached logits")
    path = rating.save_calibration(rating.ASSETS_DIR, cal)
    index.append({"name": cal.name, "version": cal.version, "file": path.name, "n": cal.n, "cv_accuracy": round(cal.cv["accuracy"], 3),
                  "instructions": meta["instructions"], "criteria": meta["levels"]})
    print(f"{cal.name}: {cal.version} n={cal.n} cv acc {cal.cv['accuracy']:.3f} mae {cal.cv['mae']:.3f} ece {cal.cv['ece']:.3f} -> {path.relative_to(ROOT)}")
(rating.ASSETS_DIR / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")

"""Acceptance check for the promoted rating method: does `Engine.decide` with the shipped calibrations reproduce the
score lab's held-out predictions?

For the first N test items of each lab scale it sends the ordinary request (score question, `calibrated: true`, default
`score_method`), and compares the answer with the lab's own `ens4d` prediction for the same item (lab logits from
`lab/data/main_stagesAB.jsonl.gz`, calibration = the shipped asset). Also sends ONE request carrying all five rubrics
to check the packed path end to end.

uv run python tools/check_harness_rating.py --n 40 --out results/lab/harness_rating_check.json
"""
import argparse
import json
import pathlib

import numpy as np

from glance import rating
from glance.config import load_config
from glance.lab.ladders import LADDERS
from glance.logging_utils import read_jsonl
from glance.pipeline import Engine
from glance.schema import ScoreQuestion

ROOT = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=40)
parser.add_argument("--out", default="results/lab/harness_rating_check.json")
args = parser.parse_args()

engine = Engine(load_config(overrides={"vlm": {"prefix_cache": True}}), source="check")
lab_rows = [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if r["split"] == "test"]
members = rating.MEMBERS["ens4d"]
summary = {}
for scale, meta in LADDERS.items():
    question = {"type": "score", "instructions": meta["instructions"], "criteria": meta["levels"]}
    items = [i for i in read_jsonl(ROOT / "lab/manifests" / f"{scale}.jsonl") if i["split"] == "test"][: args.n]
    lab = {m: {r["item_id"]: r["logits"] for r in lab_rows if r["ladder"] == scale and r["method_key"] == m} for m in members}
    cal = engine.rating_calibration(engine.rating_key(engine.backend("vlm"), "ens4d", ScoreQuestion(**question)))
    same, correct, lab_correct, diffs, versions = [], [], [], [], set()
    packed_same = []
    for item in items:
        body = {"model": "vlm", "state": {"images": [{"id": "img0", "path": item["path"]}]}, "questions": {"q": question},
                "options": {"calibrated": True}}
        trace = engine.decide(body)
        answer = trace.response.answers["q"]
        versions.add(answer.calibration)
        pred = int(max(answer.probabilities, key=answer.probabilities.get))
        features = np.array(sum((lab[m][item["item_id"]] for m in members), []))
        lab_pred = int(rating.apply_matrix(cal, features).argmax())
        same.append(pred == lab_pred)
        correct.append(pred == item["level"])
        lab_correct.append(lab_pred == item["level"])
        diffs.append(float(np.abs(trace.scoring.scores["q"].features - features).max()))
        # all five rubrics in one request: the answer for this item's own scale must not change
        five = {name: {"type": "score", "instructions": m["instructions"], "criteria": m["levels"]} for name, m in LADDERS.items()}
        packed = engine.decide({**body, "questions": five})
        packed_pred = int(max(packed.response.answers[scale].probabilities, key=packed.response.answers[scale].probabilities.get))
        packed_same.append(packed_pred == pred)
    summary[scale] = {"n": len(items), "calibration": sorted(versions), "same_prediction_as_lab": float(np.mean(same)),
                      "accuracy_harness": float(np.mean(correct)), "accuracy_lab_same_items": float(np.mean(lab_correct)),
                      "max_abs_logit_diff_vs_lab": max(diffs), "five_rubric_request_same_prediction": float(np.mean(packed_same))}
    print(scale, summary[scale], flush=True)
out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2) + "\n")
n = sum(s["n"] for s in summary.values())
print(f"over {n} items: same prediction as the lab on {sum(s['same_prediction_as_lab'] * s['n'] for s in summary.values()) / n:.3f}; "
      f"accuracy harness {sum(s['accuracy_harness'] * s['n'] for s in summary.values()) / n:.3f} vs lab "
      f"{sum(s['accuracy_lab_same_items'] * s['n'] for s in summary.values()) / n:.3f}")

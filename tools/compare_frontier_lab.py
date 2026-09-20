"""Head-to-head on the lab scales: frontier model(s) vs the local 4B model, on exactly the same held-out images.

Frontier picks come from `glance baseline --run <run_id>` on a run of the `ladder_*` suites (only whether each pick was
right is stored). The local numbers are recomputed from the lab's saved logits for the SAME item ids:
- v0 readout as shipped (single temperature), 0 labels
- Glance `ens4d`, 0 labels (softmax of the mean member logits)
- Glance `ens4d`, 32 labels (8 per level; mean over 20 random draws from the calibration split)
- Glance `ens4d`, all 500 calibration labels

uv run python tools/compare_frontier_lab.py --run <run_id> [--run <run_id> ...] --out results/lab/frontier_head_to_head
"""
import argparse
import collections
import json
import pathlib

import numpy as np
import yaml

from glance import rating
from glance.lab import score_methods as sm
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = rating.MEMBERS["ens4d"]
parser = argparse.ArgumentParser()
parser.add_argument("--run", action="append", required=True)
parser.add_argument("--out", default="results/lab/frontier_head_to_head")
parser.add_argument("--local-only", type=int, metavar="N", help="no frontier rows needed: local systems on the first N test items per scale of the run")
args = parser.parse_args()

rows = [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if r["method_key"] in MEMBERS + ("independent",)]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["ladder"], r["item_id"])][r["method_key"]] = r


def features(ladder, ids):
    return np.array([sum((by[(ladder, i)][m]["logits"] for m in MEMBERS), []) for i in ids])


result = {"runs": {}, "local": {}}
frontier_items = collections.defaultdict(set)
for run_id in args.run:
    run_dir = ROOT / "runs" / run_id
    model = (yaml.safe_load((run_dir / "config.yaml").read_text()).get("models") or {}).get("frontier")
    picks = [r for r in read_jsonl(run_dir / "predictions.jsonl") if r["backend"] == "frontier"]
    if not picks:
        print(f"{run_id}: no frontier rows yet (run `uv run glance baseline --run {run_id}`)")
        if args.local_only:
            seen = collections.Counter()
            for r in read_jsonl(run_dir / "predictions.jsonl"):
                scale = r["suite"].removeprefix("ladder_")
                if r["split"] == "test" and seen[scale] < args.local_only:
                    seen[scale] += 1
                    frontier_items[scale].add(r["item_id"])
        continue
    per = collections.defaultdict(list)
    for r in picks:
        scale = r["suite"].removeprefix("ladder_")
        per[scale].append(bool(r["correct"]))
        frontier_items[scale].add(r["item_id"])
    result["runs"][run_id] = {"model": model, "n": {s: len(v) for s, v in per.items()},
                              "accuracy": {s: float(np.mean(v)) for s, v in per.items()},
                              "mean_accuracy": float(np.mean([np.mean(v) for v in per.values()]))}

if frontier_items:
    rng = np.random.default_rng(7)
    local = collections.defaultdict(dict)
    for scale, ids in sorted(frontier_items.items()):
        ids = sorted(i for i in ids if len(by[(scale, i)]) == len(MEMBERS) + 1)
        y = np.array([by[(scale, i)]["digits"]["level"] for i in ids])
        cal_ids = sorted(i for (s, i), d in by.items() if s == scale and d["digits"]["split"] == "calibration" and len(d) == len(MEMBERS) + 1)
        assert not set(cal_ids) & set(ids), "frontier items must be test items"
        xc, yc = features(scale, cal_ids), np.array([by[(scale, i)]["digits"]["level"] for i in cal_ids])
        xt = features(scale, ids)
        zi_c = np.array([by[(scale, i)]["independent"]["logits"] for i in cal_ids])
        zi_t = np.array([by[(scale, i)]["independent"]["logits"] for i in ids])
        k = int(yc.max()) + 1
        local["v0 readout as shipped (single temperature), 0 rubric labels"][scale] = float(np.mean(
            sm.apply_fit("independent", zi_t, sm.fit_temperature(zi_c, yc)).argmax(1) == y))
        local["+ Glance ens4d, 0 labels"][scale] = float(np.mean(xt.reshape(len(y), len(MEMBERS), k).mean(1).argmax(1) == y))
        draws = []
        for _ in range(20):
            idx = np.concatenate([rng.choice(np.flatnonzero(yc == lvl), 32 // k, replace=False) for lvl in range(k)])
            draws.append(np.mean(rating.apply_matrix(rating.fit_matrix(xc[idx], yc[idx], k, rescale="cv"), xt).argmax(1) == y))
        local["+ Glance ens4d, 32 labels"][scale] = float(np.mean(draws))
        local["+ Glance ens4d, 500 labels"][scale] = float(np.mean(rating.apply_matrix(rating.fit_matrix(xc, yc, k), xt).argmax(1) == y))
    result["local"] = {name: {"accuracy": per, "mean_accuracy": float(np.mean(list(per.values())))} for name, per in local.items()}
    scales = sorted(frontier_items)
    lines = ["# Frontier models vs Qwen3-VL-4B on the lab scales, same held-out images", "",
             "Frontier: zero-shot, constrained to the rubric's levels, uncalibrated (a hard pick has no probabilities to calibrate); only "
             "whether each pick was right is stored. Local rows are recomputed from the lab's saved logits for the same item ids. "
             "The temperature of the v0 row was fit on the calibration split (it cannot change a prediction).", "",
             "| System | labels used for this rubric | " + " | ".join(scales) + " | mean |", "| --- | --- | " + " | ".join("---" for _ in scales) + " | --- |"]
    for run_id, r in result["runs"].items():
        lines.append(f"| {r['model']} | 0 | " + " | ".join(f"{r['accuracy'].get(s, float('nan')):.3f}" for s in scales) + f" | **{r['mean_accuracy']:.3f}** |")
    for name, r in result["local"].items():
        labels = "500" if "500 labels" in name else ("32" if "32 labels" in name else "0")
        lines.append(f"| Qwen3-VL-4B, {name.split(',')[0]} | {labels} | " + " | ".join(f"{r['accuracy'][s]:.3f}" for s in scales) + f" | **{r['mean_accuracy']:.3f}** |")
    lines += ["", "n per scale: " + ", ".join(f"{s} {len(frontier_items[s])}" for s in scales) + "."]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("\n".join(lines))

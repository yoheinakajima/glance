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
frontier_correct = {}
local_correct = collections.defaultdict(dict)  # system -> (scale, item_id) -> correctness (a fraction for the 32-label row: mean over draws)
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
    model = str(picks[0].get("model", "")).removeprefix("frontier:") or model or run_id  # rows first: a copied run folder keeps the old config
    # A call that failed or returned no valid answer counts as a wrong answer (same rule as the written-output baseline).
    answered = {(r["suite"], r["item_id"]) for r in picks}
    failed = [e for e in read_jsonl(run_dir / "errors.jsonl") if e.get("backend") == "frontier" and (e["suite"], e["item_id"]) not in answered]
    picks += [{"suite": e["suite"], "item_id": e["item_id"], "correct": False} for e in {(e["suite"], e["item_id"]): e for e in failed}.values()]
    per = collections.defaultdict(list)
    for r in picks:
        scale = r["suite"].removeprefix("ladder_")
        per[scale].append(bool(r["correct"]))
        frontier_items[scale].add(r["item_id"])
        frontier_correct[(run_id, scale, r["item_id"])] = bool(r["correct"])
    result["runs"][run_id] = {"model": model, "failed_calls_counted_wrong": len({(e["suite"], e["item_id"]) for e in failed}), "n": {s: len(v) for s, v in per.items()},
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
        def record(name, hits):
            local[name][scale] = float(np.mean(hits))
            for i, h in zip(ids, np.asarray(hits, dtype=float)):
                local_correct[name][(scale, i)] = float(h)

        record("v0 readout as shipped (single temperature), 0 rubric labels", sm.apply_fit("independent", zi_t, sm.fit_temperature(zi_c, yc)).argmax(1) == y)
        record("+ Glance ens4d, 0 labels", xt.reshape(len(y), len(MEMBERS), k).mean(1).argmax(1) == y)
        zt = ((xt - xc.mean(0)) / (xc.std(0) + 1e-6)).reshape(len(y), len(MEMBERS), k).mean(1)  # unlabeled pool = calibration images, labels unused
        record("+ Glance ens4d, 0 labels + unlabeled images", zt.argmax(1) == y)
        draws = []
        for _ in range(20):
            idx = np.concatenate([rng.choice(np.flatnonzero(yc == lvl), 32 // k, replace=False) for lvl in range(k)])
            draws.append(rating.apply_matrix(rating.fit_matrix(xc[idx], yc[idx], k, rescale="cv"), xt).argmax(1) == y)
        record("+ Glance ens4d, 32 labels", np.mean(draws, axis=0))
        record("+ Glance ens4d, 500 labels", rating.apply_matrix(rating.fit_matrix(xc, yc, k), xt).argmax(1) == y)
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
        labels = "500" if "500 labels" in name else ("32" if "32 labels" in name else ("0 (plus unlabeled images)" if "unlabeled" in name else "0"))
        lines.append(f"| Qwen3-VL-4B, {name.split(',')[0]} | {labels} | " + " | ".join(f"{r['accuracy'][s]:.3f}" for s in scales) + f" | **{r['mean_accuracy']:.3f}** |")
    # paired bootstrap over items: local minus frontier, mean over scales
    boot_rng, paired = np.random.default_rng(7), {}
    for run_id, r in result["runs"].items():
        for name in result["local"]:
            diffs = []
            for sc in scales:
                keys = sorted(i for i in frontier_items[sc] if (run_id, sc, i) in frontier_correct and (sc, i) in local_correct[name])
                d = np.array([local_correct[name][(sc, i)] - float(frontier_correct[(run_id, sc, i)]) for i in keys])
                diffs.append(d[boot_rng.integers(0, len(d), size=(5000, len(d)))].mean(1))
            mean_boot = np.mean(diffs, axis=0)
            paired[f"{name} minus {r['model']}"] = {"points": float(100 * (result["local"][name]["mean_accuracy"] - r["mean_accuracy"])),
                                                   "ci95": [float(100 * np.percentile(mean_boot, 2.5)), float(100 * np.percentile(mean_boot, 97.5))]}
    result["paired_differences"] = paired
    lines += ["", "Paired differences in mean accuracy, same images (bootstrap 95% interval over items):", "", "| Difference | points | 95% interval |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v['points']:+.1f} | [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] |" for k, v in paired.items()]
    lines += ["", "n per scale: " + ", ".join(f"{s} {len(frontier_items[s])}" for s in scales) + "."]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print("\n".join(lines))

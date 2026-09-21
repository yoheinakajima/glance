"""E11 (lab/NOTES.md entry 32): accuracy of the same frozen model WRITING a JSON answer, next to READING it (Glance).

uv run python tools/gen_accuracy_report.py
"""
import collections
import json
import pathlib

import numpy as np

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7)
written = read_jsonl(ROOT / "lab/runs/gen_accuracy.jsonl")
INAT = ROOT / "lab/runs/gen_accuracy_inat.jsonl"  # written answers on the iNaturalist set (entry 35c), all items
written += read_jsonl(INAT) if INAT.exists() else []
M = rating.MEMBERS["ens4d"]
lab = collections.defaultdict(dict)
for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz"):
    if r["method_key"] in M:
        lab[(r["ladder"], r["item_id"])][r["method_key"]] = r
fresh = {(r["suite"], r["item_id"]): r for run in ("20260920T205633Z-99f822", "20260920T232332Z-80efa7")  # the Commons and the iNaturalist local runs
         for r in read_jsonl(ROOT / "runs" / run / "predictions.jsonl") if r["backend"] == "vlm" and r["method"] in ("statement", "independent")}


def read_correct(row):
    """The raw, zero-label read of the same item."""
    if row["suite"].startswith("ladder_"):
        d = lab[(row["suite"].removeprefix("ladder_"), row["item_id"])]
        z = np.mean([d[m]["logits"] for m in M], axis=0)
        return int(np.argmax(z)) == d["digits"]["level"]
    p = fresh[(row["suite"], row["item_id"])]
    return (p["raw"] >= 0.5) == bool(p["label_index"]) if p["type"] == "noul" else int(np.argmax(p["raw"])) == int(p["label_index"])


def boot(a):
    a = np.asarray(a, dtype=float)
    b = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1)
    return [float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


out = {}
for suite in dict.fromkeys(r["suite"] for r in written):
    g = [r for r in written if r["suite"] == suite]
    w, rd = np.array([r["correct"] for r in g], float), np.array([read_correct(r) for r in g], float)
    out[suite] = {"n": len(g), "written": boot(w), "read_0_labels": boot(rd), "read_minus_written_points": [100 * x for x in boot(rd - w)],
                  "same_outcome": float(np.mean(w == rd)),  # share of items where written and read are both right or both wrong (picks are not compared: prompts differ)
                  "invalid_written": int(sum(not r["valid"] for r in g)), "write_ms_p50": float(np.percentile([r["write_ms"] for r in g], 50))}
ladders = [s for s in out if s.startswith("ladder_")]
head = json.loads((ROOT / "results/lab/frontier_head_to_head.json").read_text())["local"]
out["lab_mean"] = {"written": float(np.mean([out[s]["written"][0] for s in ladders])), "read_0_labels": float(np.mean([out[s]["read_0_labels"][0] for s in ladders])),
                   "read_unlabeled_images": head["+ Glance ens4d, 0 labels + unlabeled images"]["mean_accuracy"],
                   "read_32_labels": head["+ Glance ens4d, 32 labels"]["mean_accuracy"]}
(ROOT / "results/lab/gen_accuracy.json").write_text(json.dumps(out, indent=2) + "\n")
f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
lines = ["# The same frozen Qwen3-VL-4B: WRITE a JSON answer, or READ it (Glance)? Accuracy on identical items", "",
         "Writing = greedy generation of one JSON field, invalid or unparsable counts as wrong. Reading = raw, zero-label Glance readout "
         "(`ens4d` mean logits for ratings; yes/no and pick-one as shipped).", "",
         "| Suite | n | written | read, 0 labels | read minus written, points | same right/wrong outcome | invalid written | write p50 ms (GPU was shared unless noted) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for s, e in out.items():
    if s != "lab_mean":
        d = e["read_minus_written_points"]
        lines.append(f"| {s} | {e['n']} | {f(e['written'])} | {f(e['read_0_labels'])} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] | {e['same_outcome']:.1%} | {e['invalid_written']} | {e['write_ms_p50']:.0f} |")
m = out["lab_mean"]
lines += ["", f"Lab scales, mean over five rubrics: written {m['written']:.3f}; read with 0 labels {m['read_0_labels']:.3f}; read with unlabeled images "
          f"{m['read_unlabeled_images']:.3f}; read with 32 labels {m['read_32_labels']:.3f} (the last two from `results/lab/frontier_head_to_head.json`, same items)."]
(ROOT / "results/lab/gen_accuracy.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

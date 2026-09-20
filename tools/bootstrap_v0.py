"""Bootstrap 95% intervals for the v0 out-of-the-box numbers and the paired gap to the frontier baseline, from the
committed snapshot of the full evaluation (results/v0/m5_full_eval/predictions.jsonl.gz). Uncalibrated decisions
(argmax of the raw probabilities; p >= 0.5 for yes/no), test split, the items the frontier model also answered.

uv run python tools/bootstrap_v0.py
"""
import collections
import json
import pathlib

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = read_jsonl(ROOT / "results/v0/m5_full_eval/predictions.jsonl.gz")
DRAWS, rng = 10000, np.random.default_rng(7)


def correct(r):
    if r["backend"] == "frontier":
        return bool(r["correct"])
    if r["type"] == "noul":
        return (r["raw"] >= 0.5) == bool(r["label_index"])
    return int(np.argmax(r["raw"])) == int(r["label_index"])


by = collections.defaultdict(dict)
for r in rows:
    if r["split"] != "test":
        continue
    key = "frontier" if r["backend"] == "frontier" else f"{r['backend']}:{r['method']}"
    by[r["suite"]].setdefault(key, {})[r["item_id"]] = correct(r)

PRIMARY = {"pope": "vlm:statement", "gqa_yesno": "vlm:statement", "pets37": "vlm:independent", "caltech101": "vlm:independent", "blur_ladder": "vlm:statement"}
out = {"draws": DRAWS, "suites": {}}
gaps = {}
for suite, systems in by.items():
    ids = sorted(set(systems[PRIMARY[suite]]) & set(systems.get("frontier", systems[PRIMARY[suite]])))
    entry = {"n": len(ids)}
    arrays = {k: np.array([v[i] for i in ids], dtype=float) for k, v in systems.items() if all(i in v for i in ids)}
    idx = rng.integers(0, len(ids), size=(DRAWS, len(ids)))
    for k, a in arrays.items():
        boot = a[idx].mean(1)
        entry[k] = {"accuracy": float(a.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    if "frontier" in arrays:
        d = arrays["frontier"] - arrays[PRIMARY[suite]]
        boot = d[idx].mean(1)
        entry["frontier_minus_local"] = {"points": float(100 * d.mean()), "ci95": [float(100 * np.percentile(boot, 2.5)), float(100 * np.percentile(boot, 97.5))]}
        gaps[suite] = boot
    out["suites"][suite] = entry
for name, keep in (("all five suites", list(gaps)), ("yes/no and choice suites only (no blur_ladder)", [s for s in gaps if s != "blur_ladder"])):
    mean_boot = np.mean([gaps[s] for s in keep], axis=0)
    point = float(np.mean([out["suites"][s]["frontier_minus_local"]["points"] for s in keep]))
    out[f"mean gap, {name}"] = {"points": point, "ci95": [float(100 * np.percentile(mean_boot, 2.5)), float(100 * np.percentile(mean_boot, 97.5))]}
(ROOT / "results/v0/analysis/bootstrap_v0.json").write_text(json.dumps(out, indent=2) + "\n")
lines = ["# v0 out of the box: bootstrap 95% intervals (test split, uncalibrated decisions, 10,000 resamples)", "",
         "| Suite | n | Qwen3-VL-4B (primary method) | Claude Opus 5 (zero-shot pick) | Opus minus local, points |", "| --- | --- | --- | --- | --- |"]
f = lambda e: f"{e['accuracy']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"  # noqa: E731
for suite, e in out["suites"].items():
    g = e.get("frontier_minus_local")
    lines.append(f"| {suite} | {e['n']} | {f(e[PRIMARY[suite]])} | {f(e['frontier']) if 'frontier' in e else '-'} | "
                 + (f"{g['points']:+.1f} [{g['ci95'][0]:+.1f}, {g['ci95'][1]:+.1f}]" if g else "-") + " |")
for k, v in out.items():
    if k.startswith("mean gap"):
        lines.append(f"\n{k}: {v['points']:+.1f} points [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] (Opus ahead when positive).")
for suite, e in out["suites"].items():
    extra = [k for k in e if k.startswith(("vlm:", "siglip:")) and k != PRIMARY[suite]]
    if extra:
        lines.append(f"\n{suite}, other local rows: " + "; ".join(f"{k} {f(e[k])}" for k in extra))
(ROOT / "results/v0/analysis/bootstrap_v0.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

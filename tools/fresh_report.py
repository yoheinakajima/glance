"""E10 (lab/NOTES.md entry 30): out-of-the-box accuracy on fresh real photos whose labels are Wikimedia Commons "depicts"
statements. Uncalibrated decisions, ALL items (nothing is fit, so both splits count), bootstrap 95% intervals.

uv run python tools/fresh_report.py --run <run_id>
"""
import argparse
import collections
import json
import pathlib

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
args = parser.parse_args()
rows = read_jsonl(ROOT / "runs" / args.run / "predictions.jsonl")
rng = np.random.default_rng(7)


def correct(r):
    if r["backend"] == "frontier":
        return bool(r["correct"])
    if r["type"] == "noul":
        return (r["raw"] >= 0.5) == bool(r["label_index"])
    return int(np.argmax(r["raw"])) == int(r["label_index"])


def ci(a):
    a = np.asarray(a, dtype=float)
    boot = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1)
    return {"accuracy": float(a.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": int(len(a))}


out = {"run": args.run, "suites": {}}
for suite in ("fresh_choice", "fresh_yesno"):
    by_system = collections.defaultdict(list)
    for r in rows:
        if r["suite"] == suite:
            name = str(r.get("model", "")).removeprefix("frontier:") if r["backend"] == "frontier" else f"{r['backend']}:{r['method']}"
            by_system[name].append(r)
    entry = {}
    for name, group in by_system.items():
        e = ci([correct(r) for r in group])
        if suite == "fresh_yesno":
            e["on_yes_questions"] = float(np.mean([correct(r) for r in group if r["label_index"] == 1]))
            e["on_no_questions"] = float(np.mean([correct(r) for r in group if r["label_index"] == 0]))
        else:
            per = collections.defaultdict(list)
            for r in group:
                per[str(r["label"])].append(correct(r))
            e["per_class"] = {k: [float(np.mean(v)), len(v)] for k, v in sorted(per.items())}
        entry[name] = e
    out["suites"][suite] = entry
(ROOT / "results/lab/fresh_commons.json").write_text(json.dumps(out, indent=2) + "\n")
f = lambda e: f"{e['accuracy']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"  # noqa: E731
lines = ["# Out of the box on fresh real photos (taken after the models were released; labels from Commons 'depicts' statements)", "",
         "Uncalibrated decisions, all items, bootstrap 95% intervals. Label noise, the same for every system: a 'depicts' tag means the thing "
         "APPEARS in the photo, not that it is the main subject, so the pick-one labels are noisier than the yes/no labels; a 'no' question can be "
         "wrong when the other object happens to be in frame.", "",
         "## Yes/no: \"Is there a <class> in the photo?\"", "", "| System | n | accuracy | on yes questions | on no questions |", "| --- | --- | --- | --- | --- |"]
lines += [f"| {k} | {e['n']} | {f(e)} | {e['on_yes_questions']:.3f} | {e['on_no_questions']:.3f} |" for k, e in out["suites"]["fresh_yesno"].items()]
lines += ["", "## Pick one of 13: \"What is the main subject?\"", "", "| System | n | accuracy |", "| --- | --- | --- |"]
lines += [f"| {k} | {e['n']} | {f(e)} |" for k, e in out["suites"]["fresh_choice"].items()]
first = next(iter(out["suites"]["fresh_choice"].values()), None)
if first:
    classes = list(first["per_class"])
    lines += ["", "Per class (accuracy, photos):", "", "| System | " + " | ".join(classes) + " |", "| --- | " + " | ".join("---" for _ in classes) + " |"]
    lines += [f"| {k} | " + " | ".join(f"{e['per_class'][c][0]:.2f} ({e['per_class'][c][1]})" for c in classes) + " |" for k, e in out["suites"]["fresh_choice"].items()]
(ROOT / "results/lab/fresh_commons.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

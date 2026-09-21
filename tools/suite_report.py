"""Accuracy per suite and per system for any set of suites, from one local eval run plus copies of it that hold hosted
models' right/wrong flags (tools/frontier_batch.py). Hosted models answer the test half, so every system is ALSO reported on
exactly those items ("same items"); the open model is reported on all items too. Uncalibrated decisions, bootstrap 95%.

uv run python tools/suite_report.py --name probes --prefix probe_ --run <local run> [--run <copy> ...]
"""
import argparse
import collections
import json
import pathlib

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="output name: results/lab/<name>.json and .md")
parser.add_argument("--prefix", required=True, help="suite name prefix, e.g. probe_ or ui_")
parser.add_argument("--run", action="append", required=True)
parser.add_argument("--by", help="also break the OPEN model's accuracy down by this key of the item's meta/params (e.g. ratio)")
args = parser.parse_args()
rng = np.random.default_rng(7)


def ci(hits):
    a = np.asarray(hits, dtype=float)
    b = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(1)
    return {"accuracy": float(a.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(len(a))}


def hit(r):
    if r["backend"] == "frontier":
        return bool(r["correct"])
    return ((r["raw"] >= 0.5) == bool(r["label_index"])) if r["type"] == "noul" else (int(np.argmax(r["raw"])) == int(r["label_index"]))


rows = [r for r in read_jsonl(ROOT / "runs" / args.run[0] / "predictions.jsonl") if r["suite"].startswith(args.prefix)]
for extra in args.run[1:]:
    rows += [r for r in read_jsonl(ROOT / "runs" / extra / "predictions.jsonl") if r["backend"] == "frontier" and r["suite"].startswith(args.prefix)]
by_suite = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    if r["backend"] == "vlm" and r["method"] not in ("statement", "independent"):
        continue
    name = str(r.get("model", "")).removeprefix("frontier:") if r["backend"] == "frontier" else ("open 4B model, read" if r["backend"] == "vlm" else f"{r['backend']}")
    by_suite[r["suite"]][name][r["item_id"]] = hit(r)
out = {"runs": args.run, "suites": {}}
for suite, systems in sorted(by_suite.items()):
    hosted = [n for n in systems if n not in ("open 4B model, read", "siglip")]
    shared = set.intersection(*(set(systems[n]) for n in hosted)) if hosted else set()
    entry = {}
    for name, got in systems.items():
        e = {"all_items": ci(list(got.values()))}
        if shared and shared <= set(got):
            e["same_items"] = ci([got[i] for i in sorted(shared)])
        entry[name] = e
    out["suites"][suite] = entry
(ROOT / f"results/lab/{args.name}.json").write_text(json.dumps(out, indent=1))
f = lambda c: f"{c['accuracy']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"  # noqa: E731
md = [f"# `{args.prefix}*` suites: accuracy per system (uncalibrated decisions, bootstrap 95% intervals)", ""]
for suite, entry in out["suites"].items():
    md += [f"## {suite}", "", "| System | same items as the hosted models | n | all items | n |", "| --- | --- | --- | --- | --- |"]
    for name, e in entry.items():
        s = e.get("same_items")
        md.append(f"| {name} | {f(s) if s else '-'} | {s['n'] if s else '-'} | {f(e['all_items'])} | {e['all_items']['n']} |")
    md.append("")
(ROOT / f"results/lab/{args.name}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

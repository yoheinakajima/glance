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
parser.add_argument("--by", action="append", default=[], metavar="SUITE=KEY",
                    help="also break every system's accuracy on SUITE down by KEY of the item's generator params in its source manifest, or by `label` (e.g. probe_largest=ratio)")
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
labels = {}
for r in rows:
    if r["backend"] == "vlm" and r["method"] not in ("statement", "independent"):
        continue
    name = str(r.get("model", "")).removeprefix("frontier:") if r["backend"] == "frontier" else ("open 4B model, read" if r["backend"] == "vlm" else f"{r['backend']}")
    by_suite[r["suite"]][name][r["item_id"]] = hit(r)
    labels[(r["suite"], r["item_id"])] = str(r.get("label"))
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


def order(value):
    """Numbers in numeric order, words alphabetically after them."""
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def params_of(suite):
    """item id -> {label, **generator params}, from the generator's source manifests (ground truth lives there, not in the run)."""
    found = {}
    for path in sorted((ROOT / "glance/evals/manifests").glob("*_source.jsonl")):
        if not path.name.startswith(("probes_", "ui_screens")):
            continue
        for row in read_jsonl(path):
            if "item_id" in row:  # the UI screens manifest is keyed by screen, not by item: no per-item params there
                found.setdefault(row["item_id"], {"label": row.get("label"), **(row.get("params") or {})})
    return found


out["breakdowns"] = {}
for spec in args.by:
    suite, key = spec.split("=", 1)
    params, groups = params_of(suite), collections.defaultdict(lambda: collections.defaultdict(list))
    for name, got in by_suite.get(suite, {}).items():
        for item_id, ok in got.items():
            meta = params.get(item_id) or params.get(item_id.split("/")[-1]) or params.get(item_id.split(":")[-1]) or {}
            # keys starting with @ come from the run itself: the item's label, the part of its id after `__`, or the second `_` token of its id
            special = {"@label": labels.get((suite, item_id)), "@id_suffix": item_id.split("__")[-1], "@id_kind": (item_id.split("_") + [""])[1]}
            groups[name][str(special[key] if key in special else meta.get(key, "unknown"))].append(ok)
    out["breakdowns"][suite] = {"by": key, "systems": {name: {value: {"accuracy": float(np.mean(hits)), "n": len(hits)} for value, hits in sorted(g.items(), key=lambda kv: order(kv[0]))}
                                                       for name, g in groups.items()}}
(ROOT / f"results/lab/{args.name}.json").write_text(json.dumps(out, indent=1))
f = lambda c: f"{c['accuracy']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"  # noqa: E731
md = [f"# `{args.prefix}*` suites: accuracy per system (uncalibrated decisions, bootstrap 95% intervals)", ""]
for suite, entry in out["suites"].items():
    md += [f"## {suite}", "", "| System | same items as the hosted models | n | all items | n |", "| --- | --- | --- | --- | --- |"]
    for name, e in entry.items():
        s = e.get("same_items")
        md.append(f"| {name} | {f(s) if s else '-'} | {s['n'] if s else '-'} | {f(e['all_items'])} | {e['all_items']['n']} |")
    md.append("")
for suite, b in out["breakdowns"].items():
    values = sorted({v for g in b["systems"].values() for v in g}, key=order)
    md += [f"## {suite}, by {b['by']} (accuracy, items)", "", "| System | " + " | ".join(values) + " |", "| --- | " + " | ".join("---" for _ in values) + " |"]
    md += [f"| {name} | " + " | ".join(f"{g[v]['accuracy']:.2f} ({g[v]['n']})" if v in g else "-" for v in values) + " |" for name, g in b["systems"].items()]
    md.append("")
(ROOT / f"results/lab/{args.name}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

"""E23 (`lab/NOTES.md` entry 54): the three fresh photo sets pooled, yes/no and pick-one separately, on the items the hosted
models were asked (the test halves), a failed or missing hosted call counted as wrong. Item-pooled accuracy per system, the
paired difference open-minus-hosted per hosted model, 95% intervals from a bootstrap stratified by photo set (10,000 draws,
seed 7), and the equal-weight-per-set mean as a sensitivity check. NOT a blind analysis: every per-set result was seen before
the rule was registered; the rule is mechanical and drops no set and no system.

uv run python tools/pooled_photos.py
"""
import collections
import json
import pathlib

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETS = {"Commons": ("20260920T205633Z-99f822", "fresh_yesno", "fresh_choice"),
        "iNaturalist, ten groups": ("20260920T232332Z-80efa7", "inat_yesno", "inat_choice"),
        "iNaturalist, insect orders": ("20260921T061001Z-da85d6", "inat_orders_yesno", "inat_orders_choice")}
SUFFIXES = ["", "-opus", "-gpt", "-gemini", "-haiku", "-gptsmall", "-flashlite"]
NAMES = {"anthropic/claude-opus-5": "Claude Opus 5", "openai/gpt-5.6": "GPT-5.6", "openrouter/google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
         "anthropic/claude-haiku-4-5": "Claude Haiku 4.5", "openai/gpt-5.6-luna": "GPT-5.6 Luna", "openrouter/google/gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite"}
OPEN = "Qwen3-VL-4B, read"
rng = np.random.default_rng(7)


def local_hit(r):
    return ((r["raw"] >= 0.5) == bool(r["label_index"])) if r["type"] == "noul" else (int(np.argmax(r["raw"])) == int(r["label_index"]))


# hits[kind][set name][system][item id] = bool
hits = {"yesno": collections.defaultdict(lambda: collections.defaultdict(dict)), "choice": collections.defaultdict(lambda: collections.defaultdict(dict))}
for set_name, (base, yes_suite, choice_suite) in SETS.items():
    kind_of = {yes_suite: "yesno", choice_suite: "choice"}
    for suffix in SUFFIXES:
        path = ROOT / "runs" / (base + suffix) / "predictions.jsonl"
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if r["suite"] not in kind_of:
                continue
            if r["backend"] == "frontier":
                name = NAMES.get(str(r.get("model", "")).removeprefix("frontier:"))
                if name:
                    hits[kind_of[r["suite"]]][set_name][name][r["item_id"]] = bool(r["correct"])
            elif r["backend"] == "vlm" and r["method"] in ("statement", "independent") and suffix == "":
                hits[kind_of[r["suite"]]][set_name][OPEN][r["item_id"]] = local_hit(r)

out = {"sets": {k: v[0] for k, v in SETS.items()}, "kinds": {}}
for kind, by_set in hits.items():
    systems = [OPEN] + [n for n in NAMES.values() if any(n in by_set[s] for s in by_set)]
    # the items the hosted models were asked: the union over hosted models per set; a hosted model without a row for one of them failed that call
    asked = {s: sorted(set().union(*(set(by_set[s][n]) for n in systems[1:] if n in by_set[s]))) for s in SETS}
    table = {s: {n: np.array([bool(by_set[s].get(n, {}).get(i, False)) for i in asked[s]], dtype=float) for n in systems} for s in SETS}
    missing = {n: int(sum(i not in by_set[s].get(n, {}) for s in SETS for i in asked[s])) for n in systems}
    draws = [[rng.integers(0, len(asked[s]), size=len(asked[s])) for s in SETS] for _ in range(10000)]

    def stat(f):
        point = f({s: np.arange(len(asked[s])) for s in SETS})
        boot = np.array([f(dict(zip(SETS, d))) for d in draws])
        return [float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    pooled = lambda n: (lambda idx: float(np.concatenate([table[s][n][idx[s]] for s in SETS]).mean()))  # noqa: E731
    macro = lambda n: (lambda idx: float(np.mean([table[s][n][idx[s]].mean() for s in SETS])))  # noqa: E731
    diff = lambda n: (lambda idx: 100 * float(np.concatenate([table[s][OPEN][idx[s]] - table[s][n][idx[s]] for s in SETS]).mean()))  # noqa: E731
    entry = {"n_items": {s: len(asked[s]) for s in SETS}, "n_total": int(sum(len(v) for v in asked.values())), "missing_hosted_rows_counted_wrong": missing, "systems": {}}
    for n in systems:
        e = {"pooled": stat(pooled(n)), "equal_weight_per_set": stat(macro(n)), "per_set": {s: float(table[s][n].mean()) for s in SETS}}
        if n != OPEN:
            d = stat(diff(n))
            e["open_minus_this_points"] = d
            e["verdict"] = "indistinguishable" if d[1] <= 0 <= d[2] else ("open model ahead" if d[1] > 0 else "open model behind")
            e["within_3_points"] = bool(-3 <= d[1] and d[2] <= 3)
        entry["systems"][n] = e
    out["kinds"][kind] = entry
(ROOT / "results/lab/pooled_photos.json").write_text(json.dumps(out, indent=1) + "\n")

f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
md = ["# Three fresh photo sets pooled (E23, `lab/NOTES.md` entry 54; NOT blind: the per-set results were seen before the rule was fixed)", "",
      "Items: the test halves the hosted models were asked, the open model on the same items; a failed hosted call counts as wrong. 95% intervals from a bootstrap stratified by photo set. "
      "The difference column is paired on the same items.", ""]
for kind, title in (("yesno", "Yes/no"), ("choice", "Pick-one")):
    e = out["kinds"][kind]
    md += [f"## {title}: {e['n_total']} items ({', '.join(f'{s} {n}' for s, n in e['n_items'].items())})", "",
           "| System | pooled accuracy | equal weight per set | " + " | ".join(SETS) + " | open model minus this system, points | reading |", "| --- | --- | --- | " + " | ".join("---" for _ in SETS) + " | --- | --- |"]
    for n, s in e["systems"].items():
        d = s.get("open_minus_this_points")
        md.append(f"| {n} | {f(s['pooled'])} | {s['equal_weight_per_set'][0]:.3f} | " + " | ".join(f"{s['per_set'][k]:.3f}" for k in SETS) + " | "
                  + (f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]" if d else "-") + " | " + (s.get("verdict", "-") + (", within 3 points" if s.get("within_3_points") else "")) + " |")
    md += ["", "Hosted rows missing and counted wrong: " + ", ".join(f"{n} {m}" for n, m in e["missing_hosted_rows_counted_wrong"].items() if n != OPEN and m) + ".", ""]
(ROOT / "results/lab/pooled_photos.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

"""The comparison matrix the owner asked for: five systems (three hosted frontier models, the open model WRITING its answer,
the open model READ with Glance) x three tests (yes/no, pick-one, rating) x accuracy, speed and cost, with every accuracy
computed on the SAME items for all five rows. Cells that are estimates or not measured yet say so.
Writes results/lab/matrix.{json,md}; tools/make_site.py renders it.

uv run python tools/make_matrix.py
"""
import collections
import json
import pathlib
import statistics

import numpy as np

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTIER = [("Gemini 3.1 Pro", "gemini", "20260920T205633Z-99f822-gemini", "20260920T170146Z-8ff72a"),
            ("Claude Opus 5", "opus", "20260920T205633Z-99f822", "20260920T165748Z-8ff72a"),
            ("GPT-5.6", "gpt", "20260920T205633Z-99f822-gpt", "20260920T165949Z-8ff72a")]
TESTS = {"yesno": ("fresh_yesno", "Yes/no, 131 fresh Commons questions"), "choice": ("fresh_choice", "Pick one of 13, 65 fresh Commons photos"),
         "rating": (None, "Rating, exact level of 4, 1,000 lab images")}
LIST_PRICE = json.loads((ROOT / "results/lab/cost_estimates.json").read_text())["est_usd_per_1000_ratings"]
CLOUD = json.loads((ROOT / "results/lab/cost_model.json").read_text())["assumptions"]["cloud_gpu_usd_per_hour"]
rng = np.random.default_rng(3)


def cell(hits):
    a = np.asarray(hits, dtype=float)
    b = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(1)
    return {"accuracy": float(a.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(len(a))}


def frontier_rows(run):
    return [r for r in read_jsonl(ROOT / "runs" / run / "predictions.jsonl") if r["backend"] == "frontier"]


def local_usd(seconds):
    return [seconds / 3600 * 1000 * CLOUD[0], seconds / 3600 * 1000 * CLOUD[1]] if seconds else None


# ---- accuracy on identical items -----------------------------------------------------------------------
fresh = {name: frontier_rows(run) for name, _, run, _ in FRONTIER}
lab = {name: frontier_rows(run) for name, _, _, run in FRONTIER}
written = {(r["suite"], r["item_id"]): bool(r["correct"]) for r in read_jsonl(ROOT / "lab/runs/gen_accuracy.jsonl")}
local = {(r["suite"], r["item_id"]): r for r in read_jsonl(ROOT / "runs/20260920T205633Z-99f822/predictions.jsonl")
         if r["backend"] == "vlm" and r["method"] in ("statement", "independent")}


def read_hit(r):
    return ((r["raw"] >= 0.5) == bool(r["label_index"])) if r["type"] == "noul" else (int(np.argmax(r["raw"])) == int(r["label_index"]))


members = rating.MEMBERS["ens4d"]
by = collections.defaultdict(dict)
for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz"):
    if r["method_key"] in members:
        by[(f"ladder_{r['ladder']}", r["item_id"])][r["method_key"]] = r

systems = {name: {"kind": "hosted frontier model, zero-shot, answer written"} for name, *_ in FRONTIER}
systems["Qwen3-VL-4B, written"] = {"kind": "open 4B model on a laptop, zero-shot, writes one JSON answer (no Glance)"}
systems["Qwen3-VL-4B, read (Glance)"] = {"kind": "the same open model, zero-shot, answer read from one forward pass (four for a rating)"}
for test, (suite, _) in TESTS.items():
    pick = (lambda r: r["suite"] == suite) if suite else (lambda r: r["suite"].startswith("ladder_"))
    shared = None
    for name in fresh:
        ids = {(r["suite"], r["item_id"]) for r in (fresh if suite else lab)[name] if pick(r)}
        shared = ids if shared is None else shared | ids  # a failed call is a wrong answer, so the union is the item set
    for name in fresh:
        got = {(r["suite"], r["item_id"]): bool(r["correct"]) for r in (fresh if suite else lab)[name] if pick(r)}
        systems[name].setdefault("accuracy", {})[test] = cell([got.get(k, False) for k in sorted(shared)])
        rows = [r for r in (fresh if suite else lab)[name] if pick(r)]
        systems[name].setdefault("seconds", {})[test] = {"value": statistics.median(r["latency_ms"] for r in rows) / 1000, "how": "measured, one call at a time from this laptop"}
        costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
        key = next(k for k in LIST_PRICE if name.split()[0].lower() in k or name.split()[-1].lower() in k or ("opus" in k and "Opus" in name))
        systems[name].setdefault("usd_per_1000", {})[test] = ({"value": [1000 * sum(costs) / len(costs)] * 2, "how": "measured"} if costs
                                                              else {"value": [LIST_PRICE[key]] * 2, "how": "list-price upper estimate (this run predates cost logging)"})
    systems["Qwen3-VL-4B, written"].setdefault("accuracy", {})[test] = cell([written[k] for k in sorted(shared)])
    if suite:
        read_hits = [read_hit(local[k]) for k in sorted(shared)]
    else:
        read_hits = [int(np.mean([by[k][m]["logits"] for m in members], axis=0).argmax()) == by[k]["digits"]["level"] for k in sorted(shared)]
    systems["Qwen3-VL-4B, read (Glance)"].setdefault("accuracy", {})[test] = cell(read_hits)

# ---- speed and cost of the open model: only clean (idle GPU) timings are used --------------------------------
gen = json.loads((ROOT / "lab/GENBENCH.json").read_text())
single = ROOT / "lab/GENBENCH_SINGLE.json"
single = json.loads(single.read_text()) if single.exists() else {}
pack = json.loads((ROOT / "lab/PACKING.json").read_text())
clean = {"written": {"yesno": gen["1 yes/no"]["write_p50_ms"]}, "read": {"yesno": gen["1 yes/no"]["read_fast2_p50_ms"], "rating": pack["ens4d packed, 1 question"]["p50_ms_per_question"]}}
for test, shape in (("yesno", "1 yes/no"), ("choice", "1 pick-one"), ("rating", "1 rating")):
    if shape in single:
        clean["written"][test] = single[shape]["write_p50_ms"]
        clean["read"][test] = single[shape].get("read_ens4d_p50_ms", single[shape]["read_fast2_p50_ms"]) if test == "rating" else single[shape]["read_fast2_p50_ms"]
for label, key in (("Qwen3-VL-4B, written", "written"), ("Qwen3-VL-4B, read (Glance)", "read")):
    for test in TESTS:
        ms = clean[key].get(test)
        systems[label].setdefault("seconds", {})[test] = {"value": ms / 1000 if ms else None, "how": "measured on the laptop, GPU otherwise idle" if ms else "clean timing scheduled (idle-GPU window tonight)"}
        systems[label].setdefault("usd_per_1000", {})[test] = {"value": local_usd(ms / 1000) if ms else None, "how": "arithmetic: measured seconds x rented-GPU price, GPU assumed no faster than the laptop" if ms else "pending the timing"}

# ---- what only the read row can add ------------------------------------------------------------------------
h2h = json.loads((ROOT / "results/lab/frontier_head_to_head.json").read_text())["local"]
LF16 = json.loads((ROOT / "results/lab/label_free_test.json").read_text())["BCz, pool = 16 unlabeled (20 draws)"]["accuracy"]
ladder = json.loads((ROOT / "lab/READOUT_LADDER.json").read_text())["summary"]
extras = [
    {"what": "UNLABELED images of the rubric (`glance fit --unlabeled`)", "rating_accuracy": h2h["+ Glance ens4d, 0 labels + unlabeled images"]["mean_accuracy"],
     "note": f"zero labels, but not zero-shot; 500 unlabeled images here, and 16 already give {LF16:.3f} on the full test split; probabilities improve but are not calibrated"},
    {"what": "32 labeled images of the rubric (`glance fit`)", "rating_accuracy": h2h["+ Glance ens4d, 32 labels"]["mean_accuracy"], "note": "same 1,000 images; calibrated probabilities (ECE about 0.03)"},
    {"what": "500 labels, readout fitted on the hidden state (research result, not shipped)", "rating_accuracy": ladder["R4a"]["accuracy"], "note": "full test split, ONE forward pass; needs on the order of a hundred labels to beat the row above"},
]
out = {"tests": {k: v[1] for k, v in TESTS.items()}, "systems": systems, "only_the_read_row_can_add": extras,
       "caveats": ["Frontier models were run once, zero-shot, with a constrained written pick; a failed call counts as wrong. No few-shot prompt was tried for any written row.",
                   "Yes/no and pick-one: photographs taken after every model's release; labels are uploaders' structured 'depicts' statements (pick-one labels are noisy). Ratings: five synthetic 4-level scales.",
                   "Hosted latency includes the network from one laptop; hosted cost is the provider's bill per call where logged, otherwise a list-price upper estimate.",
                   "Open-model cost is arithmetic on measured seconds and an on-demand cloud GPU price; several questions about one image share its cost and get cheaper per answer (see the cost model)."]}
(ROOT / "results/lab/matrix.json").write_text(json.dumps(out, indent=1))

f_acc = lambda c: f"{c['accuracy']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"  # noqa: E731
f_sec = lambda c: "pending" if c["value"] is None else f"{c['value']:.2f} s"  # noqa: E731
f_usd = lambda c: "pending" if c["value"] is None else (f"${c['value'][0]:.2f}" + ("" if c["value"][0] == c["value"][1] else f" to ${c['value'][1]:.2f}") + (" (est.)" if "estimate" in c["how"] else ""))  # noqa: E731
md = ["# Five systems, three tests: accuracy, speed, cost (same items in every row)", ""]
for title, field, fmt in (("Accuracy, zero-shot [95% interval]", "accuracy", f_acc), ("Median seconds per answer", "seconds", f_sec), ("US dollars per 1,000 answers", "usd_per_1000", f_usd)):
    md += [f"## {title}", "", "| System | " + " | ".join(TESTS[t][1] for t in TESTS) + " |", "| --- | --- | --- | --- |"]
    md += [f"| {name} | " + " | ".join(fmt(s[field][t]) for t in TESTS) + " |" for name, s in systems.items()] + [""]
md += ["## What only the read row can add (rating accuracy)", "", "| Give it | exact-level accuracy | note |", "| --- | --- | --- |"]
md += [f"| {e['what']} | {e['rating_accuracy']:.3f} | {e['note']} |" for e in extras]
md += ["", "Caveats:"] + [f"- {c}" for c in out["caveats"]]
(ROOT / "results/lab/matrix.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

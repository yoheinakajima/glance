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
INAT_RUN = {"Gemini 3.1 Pro": "20260920T232332Z-80efa7-gemini", "Claude Opus 5": "20260920T232332Z-80efa7", "GPT-5.6": "20260920T232332Z-80efa7-gpt"}
PRETTY = {"anthropic/claude-haiku-4-5": "Claude Haiku 4.5", "openai/gpt-5.6-luna": "GPT-5.6 Luna", "openai/gpt-5-nano": "GPT-5 nano",
          "openrouter/google/gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "openrouter/google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite"}
for suffix in ("haiku", "gptsmall", "flashlite"):  # the providers' cheapest models (tools/frontier_batch.py --set cheap), when their runs exist
    commons, labrun, inat = (f"{base}-{suffix}" for base in ("20260920T205633Z-99f822", "20260920T165748Z-8ff72a", "20260920T232332Z-80efa7"))
    if all((ROOT / "runs" / r / "predictions.jsonl").exists() for r in (commons, labrun)):
        got = [r for r in read_jsonl(ROOT / "runs" / commons / "predictions.jsonl") if r["backend"] == "frontier"]
        if got and any(r["backend"] == "frontier" for r in read_jsonl(ROOT / "runs" / labrun / "predictions.jsonl")):
            model_id = str(got[0]["model"]).removeprefix("frontier:")
            name = PRETTY.get(model_id, model_id.split("/")[-1])
            FRONTIER.append((name, suffix, commons, labrun))
            INAT_RUN[name] = inat
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

JSON_ROWS = {(r["ladder"], r["item_id"]): r["logits"] for r in read_jsonl(ROOT / "lab/runs/lab_jsondigits.jsonl") if r["method_key"] == "jsondigits"}
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
        key = next((k for k in LIST_PRICE if (("opus" in k and "Opus" in name) or ("gpt-5.6" in k and name == "GPT-5.6") or ("gemini-3.1-pro" in k and name == "Gemini 3.1 Pro"))), None)
        if not costs and suite:  # this run predates cost logging: use the same model's measured bill on the iNaturalist photos, same question type
            twin = {"fresh_yesno": "inat_yesno", "fresh_choice": "inat_choice"}[suite]
            inat_run = INAT_RUN[name]
            inat = [r["cost_usd"] for r in frontier_rows(inat_run) if r["suite"] == twin and r.get("cost_usd") is not None] if (ROOT / "runs" / inat_run).exists() else []
            if inat:
                systems[name].setdefault("usd_per_1000", {})[test] = {"value": [1000 * sum(inat) / len(inat)] * 2, "how": "measured on the iNaturalist photos (smaller images), same question type"}
                continue
        systems[name].setdefault("usd_per_1000", {})[test] = ({"value": [1000 * sum(costs) / len(costs)] * 2, "how": "measured"} if costs
                                                              else {"value": [LIST_PRICE[key]] * 2, "how": "list-price upper estimate (this run predates cost logging)"} if key
                                                              else {"value": None, "how": "not logged"})
    systems["Qwen3-VL-4B, written"].setdefault("accuracy", {})[test] = cell([written[k] for k in sorted(shared)])
    if suite:
        read_hits = [read_hit(local[k]) for k in sorted(shared)]
    else:
        read_hits = [int(np.mean([by[k][m]["logits"] for m in members], axis=0).argmax()) == by[k]["digits"]["level"] for k in sorted(shared)]
    if not suite and JSON_ROWS:  # E16 (entry 42b): the one-pass read at the JSON answer position is the zero-shot rating read
        read_hits = [int(np.argmax(JSON_ROWS[(k[0].removeprefix("ladder_"), k[1])])) == by[k]["digits"]["level"] for k in sorted(shared)]
    systems["Qwen3-VL-4B, read (Glance)"].setdefault("accuracy", {})[test] = cell(read_hits)

# ---- speed and cost of the open model: only clean (idle GPU) timings are used --------------------------------
gen = json.loads((ROOT / "lab/GENBENCH.json").read_text())
single = ROOT / "lab/GENBENCH_SINGLE.json"
single = json.loads(single.read_text()) if single.exists() else {}
pack = json.loads((ROOT / "lab/PACKING.json").read_text())
timing = [r["latency_ms"] for r in read_jsonl(ROOT / "lab/runs/jsondigits_timing.jsonl") if r["method_key"] == "jsondigits"]  # idle-GPU window
clean = {"written": {"yesno": gen["1 yes/no"]["write_p50_ms"]}, "read": {"yesno": gen["1 yes/no"]["read_fast2_p50_ms"]}}
if timing:
    clean["read"]["rating"] = statistics.median(timing)
elif not JSON_ROWS:
    clean["read"]["rating"] = pack["ens4d packed, 1 question"]["p50_ms_per_question"]
for test, shape in (("yesno", "1 yes/no"), ("choice", "1 pick-one"), ("rating", "1 rating")):
    if shape in single:
        clean["written"][test] = single[shape]["write_p50_ms"]
        if test != "rating":
            clean["read"][test] = single[shape]["read_fast2_p50_ms"]
photo = ROOT / "lab/PHOTO_TIMING.json"  # clean timing ON the matrix's own photographs (glance.lab.photo_timing); hosted models were timed on them too
photo = json.loads(photo.read_text()) if photo.exists() else None
if photo:
    for test in ("yesno", "choice"):
        clean["written"][test], clean["read"][test] = photo[test]["write_p50_ms"], photo[test]["read_p50_ms"]
for label, key in (("Qwen3-VL-4B, written", "written"), ("Qwen3-VL-4B, read (Glance)", "read")):
    for test in TESTS:
        ms = clean[key].get(test)
        systems[label].setdefault("seconds", {})[test] = {"value": ms / 1000 if ms else None, "how": ("measured on the laptop, GPU otherwise idle, " + ("on these photographs" if (photo and test != "rating") else "on the 448 px lab images" if test == "rating"
                                                                        else "on 448 px test images and a 3-option pick-one: SMALLER inputs than the hosted rows saw; photo timing pending"))
                                                               if ms else "clean timing scheduled (idle-GPU window tonight)"}
        systems[label].setdefault("usd_per_1000", {})[test] = {"value": local_usd(ms / 1000) if ms else None, "how": "arithmetic: measured seconds x rented-GPU price, GPU assumed no faster than the laptop" if ms else "pending the timing"}

# ---- what only the read row can add ------------------------------------------------------------------------
h2h = json.loads((ROOT / "results/lab/frontier_head_to_head.json").read_text())["local"]
LF16 = json.loads((ROOT / "results/lab/label_free_test.json").read_text())["BCz, pool = 16 unlabeled (20 draws)"]["accuracy"]
ladder = json.loads((ROOT / "lab/READOUT_LADDER.json").read_text())["summary"]
JD = json.loads((ROOT / "results/lab/jsondigits.json").read_text())["mean"] if (ROOT / "results/lab/jsondigits.json").exists() else None
extras = [
    ({"what": "16 UNLABELED images of the rubric (`glance fit --unlabeled`)", "rating_accuracy": JD["json_u16"],
      "note": "zero labels, but not zero-shot; same 1,000 images, one pass; probabilities improve but are not calibrated"} if JD else
     {"what": "UNLABELED images of the rubric (`glance fit --unlabeled`)", "rating_accuracy": h2h["+ Glance ens4d, 0 labels + unlabeled images"]["mean_accuracy"],
      "note": f"zero labels, but not zero-shot; 500 unlabeled images here, and 16 already give {LF16:.3f} on the full test split; probabilities improve but are not calibrated"}),
    {"what": "32 labeled images of the rubric (`glance fit`)", "rating_accuracy": h2h["+ Glance ens4d, 32 labels"]["mean_accuracy"], "note": "same 1,000 images, the four-pass readout; calibrated probabilities (ECE about 0.03)"},
    {"what": "500 labels, readout fitted on the hidden state (research result, not shipped)", "rating_accuracy": ladder["R4a"]["accuracy"], "note": "full test split, ONE forward pass; needs on the order of a hundred labels to beat the row above"},
]
# ---- accuracy basis for yes/no and pick-one: the three fresh photo sets pooled (E23, notebook entries 54 and 55) ----------------
# Used only when tools/pooled_photos.py covers every row, the open model's written answers included; otherwise the Commons
# numbers above stand. Seconds and dollars are NOT pooled (hosted cost depends on image size): they stay as measured on Commons.
tests = {k: v[1] for k, v in TESTS.items()}
basis = "Commons test half"
pooled_path = ROOT / "results/lab/pooled_photos.json"
pooled = json.loads(pooled_path.read_text()) if pooled_path.exists() else None
if pooled and all(pooled["kinds"][k].get("written_row_complete") for k in ("yesno", "choice")):
    alias = {"Qwen3-VL-4B, read (Glance)": "Qwen3-VL-4B, read", "Qwen3-VL-4B, written": "Qwen3-VL-4B, written"}
    # Other open models read the same way (E24, notebook entry 58) join only on the same basis as every other row: the pooled
    # items for yes/no and pick-one, the same 1,000 lab images for ratings, and seconds measured the same way (idle GPU).
    for size, json_rows_path in (("2B", "lab/runs/scaling_2b.jsonl"), ("8B", "lab/runs/scaling_8b.jsonl")):
        pooled_name, row_name = f"Qwen3-VL-{size}, read", f"Qwen3-VL-{size}, read (Glance)"
        photo_t, rating_t = ROOT / f"lab/PHOTO_TIMING_{size}.json", ROOT / f"lab/runs/jsondigits_timing_{size.lower()}.jsonl"
        if not all(pooled_name in pooled["kinds"][k]["systems"] for k in ("yesno", "choice")) or not photo_t.exists() or not rating_t.exists():
            continue
        other_json = {(r["ladder"], r["item_id"]): r["logits"] for r in read_jsonl(ROOT / json_rows_path) if r["method_key"] == "jsondigits"}
        keys = sorted({(r["suite"], r["item_id"]) for name in lab for r in lab[name] if r["suite"].startswith("ladder_")})
        if not all((k[0].removeprefix("ladder_"), k[1]) in other_json for k in keys):
            continue
        pt = json.loads(photo_t.read_text())
        ms = {"yesno": pt["yesno"]["read_p50_ms"], "choice": pt["choice"]["read_p50_ms"],
              "rating": statistics.median(r["latency_ms"] for r in read_jsonl(rating_t) if r["method_key"] == "jsondigits")}
        systems[row_name] = {"kind": f"a {size} open model of the same family, zero-shot, answer read from one forward pass",
                             "accuracy": {"rating": cell([int(np.argmax(other_json[(k[0].removeprefix("ladder_"), k[1])])) == by[k]["digits"]["level"] for k in keys])},
                             "seconds": {t: {"value": v / 1000, "how": "measured on the laptop, GPU otherwise idle, " + ("on these photographs" if t != "rating" else "on the 448 px lab images")} for t, v in ms.items()},
                             "usd_per_1000": {t: {"value": local_usd(v / 1000), "how": "arithmetic: measured seconds x the same rented-GPU price as the 4B rows"} for t, v in ms.items()}}
        alias[row_name] = pooled_name
        # the same size WRITING its answer (E25, notebook entry 59): strict scoring in the row, as for every written row; the lenient
        # score (first allowed answer found in the raw text) is carried beside it so the page can say how much is formatting
        w_name, w_path, single_t = f"Qwen3-VL-{size}, written", ROOT / f"lab/runs/gen_accuracy_{size.lower()}.jsonl", ROOT / f"lab/GENBENCH_SINGLE_{size}.json"
        if all(w_name in pooled["kinds"][k]["systems"] for k in ("yesno", "choice")) and w_path.exists() and single_t.exists():
            import written_scoring
            w_rows = {(r["suite"], r["item_id"]): r for r in read_jsonl(w_path) if r["suite"].startswith("ladder_")}
            if all(k in w_rows for k in keys):
                w_ms = {"yesno": pt["yesno"]["write_p50_ms"], "choice": pt["choice"]["write_p50_ms"], "rating": json.loads(single_t.read_text())["1 rating"]["write_p50_ms"]}
                systems[w_name] = {"kind": f"the {size} open model, zero-shot, writes one JSON answer (no Glance); an unparsable or invalid answer counts as wrong",
                                   "accuracy": {"rating": cell([written_scoring.strict(w_rows[k]) for k in keys])},
                                   "lenient_accuracy": {"rating": cell([written_scoring.lenient(w_rows[k]) for k in keys])["accuracy"],
                                                        **{k: pooled["kinds"][k]["systems"][w_name]["lenient_pooled"][0] for k in ("yesno", "choice")}},
                                   "invalid_share": {"rating": float(np.mean([not w_rows[k]["valid"] for k in keys])), **{k: pooled["kinds"][k]["systems"][w_name]["invalid_share"] for k in ("yesno", "choice")}},
                                   "seconds": {t: {"value": v / 1000, "how": "measured on the laptop, GPU otherwise idle, " + ("on these photographs" if t != "rating" else "on the 448 px lab images")} for t, v in w_ms.items()},
                                   "usd_per_1000": {t: {"value": local_usd(v / 1000), "how": "arithmetic: measured seconds x the same rented-GPU price as the 4B rows"} for t, v in w_ms.items()}}
                alias[w_name] = w_name
    order = [n for n in systems if not n.startswith("Qwen")] + [n for n in ("Qwen3-VL-2B, written", "Qwen3-VL-2B, read (Glance)", "Qwen3-VL-4B, written", "Qwen3-VL-4B, read (Glance)", "Qwen3-VL-8B, written", "Qwen3-VL-8B, read (Glance)") if n in systems]
    systems = {n: systems[n] for n in order}
    for kind in ("yesno", "choice"):
        e = pooled["kinds"][kind]
        for name in systems:
            row = e["systems"][alias.get(name, name)]
            systems[name]["accuracy"][kind] = {"accuracy": row["pooled"][0], "ci95": row["pooled"][1:], "n": e["n_total"]}
    tests["yesno"] = f"Yes/no, {pooled['kinds']['yesno']['n_total']} questions about fresh photographs (three sets pooled)"
    tests["choice"] = f"Pick one, {pooled['kinds']['choice']['n_total']} fresh photographs (three sets pooled)"
    basis = "three fresh photo sets pooled"
out = {"tests": tests, "accuracy_basis": basis, "systems": systems, "only_the_read_row_can_add": extras,
       "caveats": ["The rating read is one forward pass at the JSON answer position (registered and scored once, notebook entry 42b). The four-pass readout shipped earlier scores 0.570 zero-shot on these images and remains the better option once labels exist.",
                   "Frontier models were run once, zero-shot, with a constrained written pick; a failed call counts as wrong. No few-shot prompt was tried for any written row.",
                   "Yes/no and pick-one: photographs taken after every model's release; labels are uploaders' structured 'depicts' statements (pick-one labels are noisy). Ratings: five synthetic 4-level scales.",
                   "Hosted latency includes the network from one laptop; hosted cost is the provider's bill per call where logged, otherwise a list-price upper estimate.",
                   "Open-model cost is arithmetic on measured seconds and an on-demand cloud GPU price; several questions about one image share its cost and get cheaper per answer (see the cost model)."]}
(ROOT / "results/lab/matrix.json").write_text(json.dumps(out, indent=1))

f_acc = lambda c: f"{c['accuracy']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"  # noqa: E731
f_sec = lambda c: "pending" if c["value"] is None else f"{c['value']:.2f} s"  # noqa: E731
f_usd = lambda c: "pending" if c["value"] is None else (f"${c['value'][0]:.2f}" + ("" if c["value"][0] == c["value"][1] else f" to ${c['value'][1]:.2f}") + (" (est.)" if "estimate" in c["how"] else ""))  # noqa: E731
md = ["# Five systems, three tests: accuracy, speed, cost (same items in every row)", ""]
for title, field, fmt in (("Accuracy, zero-shot [95% interval]", "accuracy", f_acc), ("Median seconds per answer", "seconds", f_sec), ("US dollars per 1,000 answers", "usd_per_1000", f_usd)):
    md += [f"## {title}", "", "| System | " + " | ".join(tests[t] for t in TESTS) + " |", "| --- | --- | --- | --- |"]
    md += [f"| {name} | " + " | ".join(fmt(s[field][t]) for t in TESTS) + " |" for name, s in systems.items()] + [""]
md += ["## What only the read row can add (rating accuracy)", "", "| Give it | exact-level accuracy | note |", "| --- | --- | --- |"]
md += [f"| {e['what']} | {e['rating_accuracy']:.3f} | {e['note']} |" for e in extras]
md += ["", "Caveats:"] + [f"- {c}" for c in out["caveats"]]
(ROOT / "results/lab/matrix.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

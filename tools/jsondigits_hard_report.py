"""E17 of lab/NOTES.md entry 43: the one-pass JSON-position read (`jsondigits`) against the raw four-pass `ens4d` read on the
harder rating benchmarks: the creative-QA rubrics (test split) and KADID-10k (first 200 test items per severity distortion),
zero-shot, and with 16 unlabeled images where calibration-split rows exist. Applies the registered decision rule.

uv run python tools/jsondigits_hard_report.py
"""
import collections
import gzip
import json
import pathlib

import numpy as np

from glance import rating

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = rating.MEMBERS["ens4d"]
BENCHES = {"creative-QA rubrics": ("lab/runs/semantic_jsondigits.jsonl", ["lab/runs/semantic.jsonl"], None),
           "KADID-10k (23 severity distortions)": ("lab/runs/kadid_jsondigits.jsonl", ["lab/data/kadid_ens4d.jsonl.gz", "lab/runs/kadid.jsonl"], {"mean_shift", "contrast_change"})}


def rows_of(path):
    f = ROOT / path
    if not f.exists():
        return []
    with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


out, rng = {}, np.random.default_rng(7)
for bench, (json_path, ens_paths, skip) in BENCHES.items():
    js = {(r["ladder"], r["item_id"]): r for r in rows_of(json_path) if r["method_key"] == "jsondigits"}
    ens = collections.defaultdict(dict)
    for path in ens_paths:
        for r in rows_of(path):
            if r["method_key"] in MEMBERS:
                ens[(r["ladder"], r["item_id"])][r["method_key"]] = r["logits"]
    per = {}
    for scale in sorted({k[0] for k in js} - (skip or set())):
        test = [r for (s, _), r in js.items() if s == scale and r["split"] == "test" and len(ens.get((s, r["item_id"]), {})) == len(MEMBERS)]
        cal = [r for (s, _), r in js.items() if s == scale and r["split"] == "calibration"]
        if len(test) < 50:
            continue
        y = np.array([r["level"] for r in test])
        xj = np.array([r["logits"] for r in test])
        xe = np.array([np.mean([ens[(scale, r["item_id"])][m] for m in MEMBERS], axis=0) for r in test])
        e = {"n": len(test), "json_zero": float(np.mean(xj.argmax(1) == y)), "ens_zero": float(np.mean(xe.argmax(1) == y)),
             "json_within_1": float(np.mean(np.abs(xj.argmax(1) - y) <= 1))}
        if len(cal) >= 32:
            xc = np.array([r["logits"] for r in cal])
            pools = [rng.choice(len(cal), 16, replace=False) for _ in range(20)]
            e["json_u16"] = float(np.mean([np.mean(((xj - xc[i].mean(0)) / (xc[i].std(0) + 1e-6)).argmax(1) == y) for i in pools]))
        per[scale] = e
    if per:
        mean = lambda k: float(np.mean([v[k] for v in per.values() if k in v]))  # noqa: E731
        out[bench] = {"per_scale": per, "mean": {k: mean(k) for k in ("json_zero", "ens_zero", "json_within_1", "json_u16") if any(k in v for v in per.values())},
                      "scales_where_json_is_at_least_ens": sum(v["json_zero"] >= v["ens_zero"] for v in per.values()), "scales": len(per)}
verdict = {}
if "creative-QA rubrics" in out:
    c = out["creative-QA rubrics"]
    verdict["H42: creative-QA, jsondigits >= raw ens4d on the mean and on at least 3 of 5 rubrics"] = bool(c["mean"]["json_zero"] >= c["mean"]["ens_zero"] and c["scales_where_json_is_at_least_ens"] >= 3)
if "KADID-10k (23 severity distortions)" in out:
    k = out["KADID-10k (23 severity distortions)"]
    verdict["H43a: KADID, jsondigits beats raw ens4d by >= 5 points"] = bool(k["mean"]["json_zero"] - k["mean"]["ens_zero"] >= 0.05)
    verdict["H43b: KADID, jsondigits stays below 0.50"] = bool(k["mean"]["json_zero"] < 0.50)
if len(out) == 2:
    verdict["DECISION (registered rule): jsondigits becomes the zero-shot default for `score`"] = bool(
        verdict["H42: creative-QA, jsondigits >= raw ens4d on the mean and on at least 3 of 5 rubrics"]
        and out["KADID-10k (23 severity distortions)"]["mean"]["json_zero"] >= out["KADID-10k (23 severity distortions)"]["mean"]["ens_zero"])
(ROOT / "results/lab/jsondigits_hard.json").write_text(json.dumps({"benches": out, "verdict": verdict}, indent=1))
md = ["# The one-pass JSON-position read on the harder rating benchmarks (E17, zero-shot)", ""]
for bench, o in out.items():
    md += [f"## {bench}", "", "| Scale | n | `jsondigits`, one pass | raw `ens4d`, four passes | `jsondigits` within one | `jsondigits` + 16 unlabeled |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {s} | {v['n']} | {v['json_zero']:.3f} | {v['ens_zero']:.3f} | {v['json_within_1']:.3f} | {v.get('json_u16', float('nan')):.3f} |" for s, v in o["per_scale"].items()]
    md += [f"| **mean** | | **{o['mean']['json_zero']:.3f}** | **{o['mean']['ens_zero']:.3f}** | {o['mean']['json_within_1']:.3f} | {o['mean'].get('json_u16', float('nan')):.3f} |", ""]
md += ["Registered predictions and decision (`lab/NOTES.md` entry 43):", ""] + [f"- {k}: {'YES' if v else 'NO'}" for k, v in verdict.items()]
(ROOT / "results/lab/jsondigits_hard.md").write_text("\n".join(md) + "\n")
print("\n".join(md) if out else "not collected yet")

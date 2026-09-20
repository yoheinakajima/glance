"""E9 (lab/NOTES.md entry 29): reference-anchored ratings on KADID-10k against the no-reference readouts on the SAME items.

uv run python tools/kadid_ref_report.py
"""
import collections
import json
import pathlib

import numpy as np

from glance.lab.analyze import combine_rows
from glance.lab.bench_report import evaluate
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEPARATE = ("mean_shift", "contrast_change")
ref_rows = read_jsonl(ROOT / "lab/runs/kadid_ref.jsonl")
keep = {(r["ladder"], r["item_id"]) for r in ref_rows}
base_rows = [r for r in read_jsonl(ROOT / "lab/runs/kadid.jsonl") if (r["ladder"], r["item_id"]) in keep]
dmos = {}
for path in (ROOT / "lab/manifests_kadid").glob("*.jsonl"):
    for r in read_jsonl(path):
        dmos[r["item_id"]] = r["dmos"]
systems = {
    "fast2_ref (reference + image, 2 passes)": combine_rows(ref_rows, "fast2_ref=ref_digits+ref_digitsrev:concat"),
    "fast2 (image only, 2 passes)": combine_rows(base_rows, "fast2=digits+digitsrev:concat"),
    "ens4d (image only, 4 passes)": combine_rows(base_rows, "ens4d=digits+zoom_digits+digitsrev+zoom_digitsrev:concat"),
}
result = {}
for name, rows in systems.items():
    per = collections.defaultdict(list)
    for r in rows:
        per[r["ladder"]].append(r)
    scales = {s: evaluate(g, "matrix", dmos) for s, g in per.items()}
    scales = {s: {k: v for k, v in m.items() if not k.startswith("_")} for s, m in scales.items() if m}
    main = [s for s in scales if s not in SEPARATE]
    result[name] = {"n_distortions": len(main), **{f: float(np.mean([scales[s][f] for s in main])) for f in ("accuracy", "within_1", "mae", "srcc_vs_dmos")},
                    "per_scale": scales}
ref, f2, e4 = (result[k] for k in systems)
verdicts = {"H23 accuracy gain over fast2 >= 10 points": ref["accuracy"] - f2["accuracy"] >= 0.10, "H23 within one >= 0.93": ref["within_1"] >= 0.93,
            "H23 Spearman with DMOS >= 0.80": ref["srcc_vs_dmos"] >= 0.80, "H24 beats no-reference ens4d": ref["accuracy"] > e4["accuracy"]}
(ROOT / "results/lab/kadid_ref.json").write_text(json.dumps({"systems": result, "verdicts": verdicts}, indent=2) + "\n")
lines = ["# KADID-10k with the pristine reference in the request (E9): same items, fixed method, matrix calibration per distortion", "",
         "First 200 manifest items of each of the 23 severity distortions (85 calibration / 115 test). Full-reference grading is a different task "
         "from no-reference grading; never compare these numbers with blind image-quality results.", "",
         "| System | exact accuracy | within one level | MAE | per-type Spearman with DMOS |", "| --- | --- | --- | --- | --- |"]
lines += [f"| {k} | {v['accuracy']:.3f} | {v['within_1']:.3f} | {v['mae']:.3f} | {v['srcc_vs_dmos']:.3f} |" for k, v in result.items()]
lines += ["", "Registered verdicts: " + "; ".join(f"{k}: {'MET' if ok else 'NOT met'}" for k, ok in verdicts.items()) + "."]
(ROOT / "results/lab/kadid_ref.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

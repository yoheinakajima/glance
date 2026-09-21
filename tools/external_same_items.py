"""Outside systems against this lab's four-pass read on EXACTLY the same items with the same fit (E13, `lab/NOTES.md` entries
37 and 37c). `tools/external_report.py` compares an outside system fitted on 300 items per scale with a number (`ens4d`
0.867) that came from 500 labels and the full test split; this tool removes that difference: both systems get the same
matrix scaling (`glance.rating.fit_matrix`), the same fit items and the same test items, with all 300 labels per scale and
with 32 (balanced, 20 draws). Added AFTER the outside system's result was seen, to make the registered comparison fair in
both directions; it is reported as such.

uv run python tools/external_same_items.py
"""
import collections
import json
import pathlib

import numpy as np

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SYSTEMS = {"q-sit-mini (0.9B, trained for image quality)": "lab/runs/external_qsit.jsonl", "openjev v2 (4B, trained claim scorer)": "lab/runs/external_openjev.jsonl"}
SCALES = ["blur", "noise", "jpeg", "exposure", "resolution"]
M = rating.MEMBERS["ens4d"]
rng = np.random.default_rng(7)

ours = collections.defaultdict(dict)
for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz"):
    if r.get("bench", "ladders") == "ladders" and r["method_key"] in M:
        ours[(r["ladder"], r["item_id"])][r["method_key"]] = r["logits"]


def hits(xf, yf, xt, yt, k, n_labels):
    """Per-test-item correctness, averaged over draws when the fit uses a subset of the labels."""
    if n_labels is None:
        return (rating.apply_matrix(rating.fit_matrix(xf, yf, k, rescale="train"), xt).argmax(1) == yt).astype(float)
    draws = []
    for _ in range(20):
        idx = np.concatenate([rng.choice(np.flatnonzero(yf == lvl), n_labels // k, replace=False) for lvl in range(k)])
        draws.append(rating.apply_matrix(rating.fit_matrix(xf[idx], yf[idx], k, rescale="cv"), xt).argmax(1) == yt)
    return np.mean(draws, axis=0)


def boot(a):
    a = np.asarray(a, dtype=float)
    b = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1)
    return [float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


out = {}
for name, path in SYSTEMS.items():
    rows = collections.defaultdict(dict)
    for r in (read_jsonl(ROOT / path) if (ROOT / path).exists() else []):
        rows[r["ladder"]][r["item_id"]] = r
    if not all(len(rows.get(s, {})) >= 600 for s in SCALES):
        continue
    entry = {"per_scale": {}}
    for label, n_labels in (("all_300_labels", None), ("32_labels", 32)):
        theirs_all, ours_all = [], []
        for scale in SCALES:
            items = rows[scale]
            fit_ids = [i for i, r in items.items() if r["split"] == "calibration" and len(ours.get((scale, i), {})) == len(M)]
            test_ids = [i for i, r in items.items() if r["split"] == "test" and len(ours.get((scale, i), {})) == len(M)]
            yf, yt = np.array([items[i]["level"] for i in fit_ids]), np.array([items[i]["level"] for i in test_ids])
            k = int(max(yf.max(), yt.max())) + 1
            feats = lambda ids, theirs: np.array([items[i]["logits"] if theirs else np.concatenate([ours[(scale, i)][m] for m in M]) for i in ids])  # noqa: E731
            a, b = hits(feats(fit_ids, True), yf, feats(test_ids, True), yt, k, n_labels), hits(feats(fit_ids, False), yf, feats(test_ids, False), yt, k, n_labels)
            entry["per_scale"].setdefault(scale, {})[label] = {"outside": float(a.mean()), "ens4d": float(b.mean()), "n_fit": len(fit_ids), "n_test": len(test_ids)}
            theirs_all.append(a)
            ours_all.append(b)
        a, b = np.concatenate(theirs_all), np.concatenate(ours_all)
        entry[label] = {"outside": boot(a), "ens4d": boot(b), "outside_minus_ens4d_points": [100 * x for x in boot(a - b)]}
    out[name] = entry
(ROOT / "results/lab/external_same_items.json").write_text(json.dumps(out, indent=1) + "\n")
f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
md = ["# Outside systems against the four-pass read (`ens4d`) on the same items with the same fit", "",
      "Both systems: matrix scaling on their own level logits, the same 300 calibration items per scale to fit (or 32 of them, balanced, 20 draws), the same 300 test items per scale, "
      "five lab scales, exact-level accuracy, paired bootstrap 95% intervals. Added after the registered comparison (`results/lab/external_systems.md`) was seen; see `lab/NOTES.md` entry 37c.", ""]
for name, e in out.items():
    md += [f"## {name}", "", "| Labels per scale | outside system | Qwen3-VL-4B + Glance `ens4d` | outside minus `ens4d`, points |", "| --- | --- | --- | --- |"]
    for label, title in (("all_300_labels", "300"), ("32_labels", "32")):
        d = e[label]["outside_minus_ens4d_points"]
        md.append(f"| {title} | {f(e[label]['outside'])} | {f(e[label]['ens4d'])} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] |")
    md += ["", "| Scale | outside, 300 labels | `ens4d`, 300 labels | outside, 32 labels | `ens4d`, 32 labels |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {s} | {v['all_300_labels']['outside']:.3f} | {v['all_300_labels']['ens4d']:.3f} | {v['32_labels']['outside']:.3f} | {v['32_labels']['ens4d']:.3f} |" for s, v in e["per_scale"].items()]
    md.append("")
if not out:
    md.append("No outside system has a complete run yet.")
(ROOT / "results/lab/external_same_items.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

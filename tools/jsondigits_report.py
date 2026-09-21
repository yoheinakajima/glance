"""E16 of lab/NOTES.md entry 42: the rating read at the forced JSON answer position (`jsondigits`, ONE pass) against the
same model's written answer, the raw four-pass `ens4d` read and the fitted options, on the E15 item subset (first 200
test and first 100 calibration items per lab scale). Verdicts on H38 to H41.

uv run python tools/jsondigits_report.py
"""
import collections
import json
import pathlib

import numpy as np
from scipy.special import softmax

from glance import rating
from glance.calibration import ece_equal_mass
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCALES, MEMBERS = ["blur", "exposure", "jpeg", "noise", "resolution"], rating.MEMBERS["ens4d"]
js = {(r["ladder"], r["item_id"]): r for r in read_jsonl(ROOT / "lab/runs/lab_jsondigits.jsonl") if r["method_key"] == "jsondigits"}
ens = collections.defaultdict(dict)
for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz"):
    if r["method_key"] in MEMBERS:
        ens[(r["ladder"], r["item_id"])][r["method_key"]] = r["logits"]
written = {(r["suite"].removeprefix("ladder_"), r["item_id"]): r for r in read_jsonl(ROOT / "lab/runs/gen_accuracy.jsonl") if r["suite"].startswith("ladder_")}
rng, per = np.random.default_rng(7), {}
for s in SCALES:
    items = [json.loads(line) for line in (ROOT / "lab/manifests" / f"{s}.jsonl").read_text().splitlines() if line.strip()]
    test, cal = [i for i in items if i["split"] == "test"][:200], [i for i in items if i["split"] == "calibration"][:100]
    yt, yc = np.array([i["level"] for i in test]), np.array([i["level"] for i in cal])
    xt, xc = np.array([js[(s, i["item_id"])]["logits"] for i in test]), np.array([js[(s, i["item_id"])]["logits"] for i in cal])
    et = np.array([[ens[(s, i["item_id"])][m] for m in MEMBERS] for i in test])
    ec = np.array([[ens[(s, i["item_id"])][m] for m in MEMBERS] for i in cal])
    p = softmax(xt, axis=1)
    w = [written[(s, i["item_id"])] for i in test]
    e = {"json_zero": p.argmax(1) == yt, "ens_zero": et.mean(1).argmax(1) == yt, "written": np.array([bool(r["correct"]) for r in w]),
         "agree_with_written": float(np.mean([r["written"] == int(a) for r, a in zip(w, p.argmax(1))])),
         "within_1": float(np.mean(np.abs(p.argmax(1) - yt) <= 1)), "ece": ece_equal_mass(p.max(1), p.argmax(1) == yt, 15),
         "off_mass": float(np.mean([js[(s, i["item_id"])]["off_mass_max"] for i in test]))}
    pools = [rng.choice(100, 16, replace=False) for _ in range(20)]
    e["json_u16"] = np.mean([(((xt - xc[i].mean(0)) / (xc[i].std(0) + 1e-6)).argmax(1) == yt) for i in pools], axis=0)
    e["ens_u16"] = np.mean([(((et - ec[i].mean(0)) / (ec[i].std(0) + 1e-6)).mean(1).argmax(1) == yt) for i in pools], axis=0)
    dj, de = [], []
    for _ in range(20):
        idx = np.concatenate([rng.choice(np.flatnonzero(yc == lvl), 8, replace=False) for lvl in range(4)])
        dj.append(rating.apply_matrix(rating.fit_matrix(xc[idx], yc[idx], 4, rescale="cv"), xt).argmax(1) == yt)
        de.append(rating.apply_matrix(rating.fit_matrix(ec.reshape(100, -1)[idx], yc[idx], 4, rescale="cv"), et.reshape(200, -1)).argmax(1) == yt)
    e["json_l32"], e["ens_l32"] = np.mean(dj, axis=0), np.mean(de, axis=0)
    per[s] = e
mean = lambda k: float(np.mean([np.mean(per[s][k]) for s in SCALES]))  # noqa: E731
rows = {k: mean(k) for k in ("written", "json_zero", "ens_zero", "json_u16", "ens_u16", "json_l32", "ens_l32", "agree_with_written", "within_1", "ece", "off_mass")}
verdicts = {"H38a: argmax agrees with the written answer on >= 97% of items": rows["agree_with_written"] >= 0.97,
            "H38b: zero-shot accuracy within 1.5 points of the written answer": abs(rows["json_zero"] - rows["written"]) <= 0.015,
            "H39: beats raw ens4d zero-shot by >= 8 points": rows["json_zero"] - rows["ens_zero"] >= 0.08,
            "H40a: 16 unlabeled images add less than they add to ens4d": (rows["json_u16"] - rows["json_zero"]) < (rows["ens_u16"] - rows["ens_zero"]),
            "H40b: with 16 unlabeled images it reaches >= 0.70": rows["json_u16"] >= 0.70,
            "H41: with 32 labels within 3 points of ens4d + matrix": rows["ens_l32"] - rows["json_l32"] <= 0.03}
out = {"mean": rows, "per_scale_zero_shot": {s: {k: float(np.mean(per[s][k])) for k in ("written", "json_zero", "ens_zero")} for s in SCALES}, "verdicts": {k: bool(v) for k, v in verdicts.items()}}
(ROOT / "results/lab/jsondigits.json").write_text(json.dumps(out, indent=1))
md = ["# Reading the rating at the JSON answer position (E16): one pass, same 1,000 images as the frontier models", "",
      "| Setting | passes | written answer | `jsondigits` read | raw `ens4d` read |", "| --- | --- | --- | --- | --- |",
      f"| zero-shot | 1 / 1 / 4 | {rows['written']:.3f} | {rows['json_zero']:.3f} | {rows['ens_zero']:.3f} |",
      f"| + 16 unlabeled images | | not possible | {rows['json_u16']:.3f} | {rows['ens_u16']:.3f} |",
      f"| + 32 labels | | not possible | {rows['json_l32']:.3f} | {rows['ens_l32']:.3f} |", "",
      f"`jsondigits` zero-shot: agrees with the written answer on {rows['agree_with_written']:.1%} of items, within one level {rows['within_1']:.3f}, ECE {rows['ece']:.3f}, "
      f"probability mass outside the digits {rows['off_mass']:.3f}.", "", "| Scale | written | `jsondigits` | raw `ens4d` |", "| --- | --- | --- | --- |"]
md += [f"| {s} | {v['written']:.3f} | {v['json_zero']:.3f} | {v['ens_zero']:.3f} |" for s, v in out["per_scale_zero_shot"].items()]
md += ["", "Registered predictions (`lab/NOTES.md` entry 42):", ""] + [f"- {k}: {'SUPPORTED' if v else 'NOT SUPPORTED'}" for k, v in verdicts.items()]
(ROOT / "results/lab/jsondigits.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

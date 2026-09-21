"""Calibration quality of the zero-shot rating read before and after self-calibration from unlabeled images (the
"ECE before / after" row reviewers asked for; METHODS 14.1). Lab test split, `ens4d`, 15 equal-mass bins with the
sampling floor; pools of 16 and 64 are 20 seeded draws from the calibration split, labels unused. Nothing is selected.

uv run python tools/label_free_ece.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from label_free_calibration import ROOT, load, predict  # noqa: E402

from glance.calibration import ece_equal_mass  # noqa: E402
from glance.evals.metrics import ece_noise_floor  # noqa: E402

data, rng, out = load(), np.random.default_rng(7), {}
for name, pool_n in (("zero-shot (no pool)", 0), ("16 unlabeled images", 16), ("64 unlabeled images", 64), ("500 unlabeled images", 500)):
    acc, ece, floor, nll = [], [], [], []
    for d in data.values():
        t, c = d["split"] == "test", np.flatnonzero(d["split"] == "calibration")
        x, y = d["x"][t], d["y"][t]
        pools = [None] if pool_n == 0 else ([c] if pool_n == 500 else [rng.choice(c, pool_n, replace=False) for _ in range(20)])
        for pool in pools:
            p = predict(x) if pool is None else predict(x, d["x"][pool], zscore=True)
            conf, hit = p.max(1), p.argmax(1) == y
            acc.append(hit.mean()); ece.append(ece_equal_mass(conf, hit, 15)); floor.append(ece_noise_floor(conf, 15))
            nll.append(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))
    out[name] = {"accuracy": float(np.mean(acc)), "ece": float(np.mean(ece)), "ece_floor": float(np.mean(floor)), "nll": float(np.mean(nll))}
(ROOT / "results/lab/label_free_ece.json").write_text(json.dumps(out, indent=1))
md = ["# Zero-shot ratings: calibration before and after self-calibration from unlabeled images (lab test split)", "",
      "| Setting | exact accuracy | ECE (sampling floor) | NLL |", "| --- | --- | --- | --- |"]
md += [f"| {k} | {e['accuracy']:.3f} | {e['ece']:.3f} ({e['ece_floor']:.3f}) | {e['nll']:.2f} |" for k, e in out.items()]
md += ["", "A labeled fit (`ens4d` + matrix, 500 labels) has ECE about 0.03 on the same split (`docs/paper/RESULTS_LAB.md`). Pool draws differ from "
       "`label_free_test.json`, so the 16-image accuracy differs from it in the third decimal."]
(ROOT / "results/lab/label_free_ece.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

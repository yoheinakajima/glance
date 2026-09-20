"""E14 of lab/NOTES.md entry 39: a content-free prior for zero-shot ratings. The model's level logits for null images
(flat grey, black, white, three noise images) are subtracted, per rubric and per `ens4d` member, from its logits for the
real image; members are then averaged. No labels, no images of the task, no fitted number. Scored on the lab TEST split.

uv run python tools/make_null_images.py
uv run python -m glance.lab.collect --bench ladders_null --out lab/runs/lab_null.jsonl --methods "digits,zoom_digits,digitsrev,zoom_digitsrev" --prefix-cache
uv run python tools/null_prior.py
"""
import collections
import json
import pathlib
import sys

import numpy as np
from scipy.special import softmax

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from label_free_calibration import K, MEMBERS, ROOT, acc, load, predict  # noqa: E402

from glance.logging_utils import read_jsonl  # noqa: E402

VARIANTS = {"all six null images (registered)": None, "flat grey only": ("null_grey",), "three noise images only": ("null_noise1", "null_noise2", "null_noise3")}


def main():
    null = collections.defaultdict(dict)
    for r in read_jsonl(ROOT / "lab/runs/lab_null.jsonl"):
        if r["method_key"] in MEMBERS:
            null[r["ladder"]][(r["item_id"], r["method_key"])] = np.array(r["logits"])
    data, out = load(), collections.defaultdict(dict)
    for scale, d in data.items():
        test, cal = d["split"] == "test", d["split"] == "calibration"
        x, y = d["x"][test], d["y"][test]
        out["raw zero-shot"][scale] = acc(predict(x), y)
        for name, keep in VARIANTS.items():
            ids = sorted({i for i, _ in null[scale] if keep is None or i in keep})
            prior = np.stack([np.mean([null[scale][(i, m)] for i in ids], axis=0) for m in MEMBERS])  # [members, K]
            out[f"content-free prior, {name}"][scale] = acc(softmax((x - prior[None]).mean(axis=1), axis=1), y)
        rng = np.random.default_rng(7)
        pools = [rng.choice(np.flatnonzero(cal), 16, replace=False) for _ in range(20)]
        out["self-calibration from 16 unlabeled images (reference, entry 26)"][scale] = tuple(
            np.mean([acc(predict(x, d["x"][p], zscore=True), y) for p in pools], axis=0))
        out["_prior_argmax"][scale] = {m: int(prior[j].argmax()) for j, m in enumerate(MEMBERS)}
    scales = sorted(data)
    table = {k: {"accuracy": float(np.mean([v[s][0] for s in scales])), "mae": float(np.mean([v[s][1] for s in scales])),
                 "within_1": float(np.mean([v[s][2] for s in scales])), "per_scale_accuracy": {s: float(v[s][0]) for s in scales}}
             for k, v in out.items() if not k.startswith("_")}
    raw, reg = table["raw zero-shot"]["accuracy"], table["content-free prior, all six null images (registered)"]["accuracy"]
    ref = table["self-calibration from 16 unlabeled images (reference, entry 26)"]["accuracy"]
    verdict = {"H37: gain over raw >= 3 points": bool(reg - raw >= 0.03), "H37: stays below self-calibration from 16 unlabeled images": bool(reg < ref),
               "gain_points": round(100 * (reg - raw), 1), "scales_where_it_hurts": [s for s in scales if out["content-free prior, all six null images (registered)"][s][0] < out["raw zero-shot"][s][0]]}
    (ROOT / "results/lab/null_prior.json").write_text(json.dumps({"table": table, "verdict": verdict, "prior_argmax_level": out["_prior_argmax"]}, indent=1))
    md = ["# Content-free prior for zero-shot ratings (E14, lab test split, no labels and no images of the task)", "",
          "| Setting | " + " | ".join(scales) + " | mean exact accuracy | within one level | MAE |", "| --- | " + " | ".join("---" for _ in scales) + " | --- | --- | --- |"]
    for k, e in table.items():
        md.append(f"| {k} | " + " | ".join(f"{e['per_scale_accuracy'][s]:.3f}" for s in scales) + f" | **{e['accuracy']:.3f}** | {e['within_1']:.3f} | {e['mae']:.3f} |")
    md += ["", "Level the null images are read as (argmax of the prior, per member): " + "; ".join(f"{s}: {sorted(set(v.values()))}" for s, v in out["_prior_argmax"].items()) + ".", "",
           f"Verdict: gain over raw {verdict['gain_points']:+.1f} points; H37 gain >= 3 points: {'SUPPORTED' if verdict['H37: gain over raw >= 3 points'] else 'NOT SUPPORTED'}; "
           f"below self-calibration: {'yes' if verdict['H37: stays below self-calibration from 16 unlabeled images'] else 'no'}; hurts on: {verdict['scales_where_it_hurts'] or 'no scale'}."]
    (ROOT / "results/lab/null_prior.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    assert K == 4
    main()

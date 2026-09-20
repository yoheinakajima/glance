"""Entry 26 of lab/NOTES.md, second half: leave one rubric out. Same question template and level words for every
distortion, so a map fit on the OTHER distortions can be applied unchanged to a distortion it has never seen (zero labels
and zero images from the target for LORO; unlabeled target images only for the BC variants).

uv run python tools/loro.py --bench kadid --in lab/runs/kadid.jsonl --exclude mean_shift,contrast_change
"""
import argparse
import collections
import json
import pathlib

import numpy as np
from scipy.special import softmax

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = rating.MEMBERS["ens4d"]
parser = argparse.ArgumentParser()
parser.add_argument("--bench", required=True)
parser.add_argument("--in", dest="inp", required=True)
parser.add_argument("--exclude", default="")
args = parser.parse_args()

by = collections.defaultdict(dict)
for r in read_jsonl(ROOT / args.inp):
    if r["method_key"] in MEMBERS:
        by[(r["ladder"], r["item_id"])][r["method_key"]] = r
exclude = {s for s in args.exclude.split(",") if s}
scales = sorted({k[0] for k in by} - exclude)
data = {}
for s in scales:
    items = sorted((i, d) for (sc, i), d in by.items() if sc == s and len(d) == len(MEMBERS))
    data[s] = {"x": np.array([sum((d[m]["logits"] for m in MEMBERS), []) for _, d in items]),
               "y": np.array([d[MEMBERS[0]]["level"] for _, d in items]), "cal": np.array([d[MEMBERS[0]]["split"] == "calibration" for _, d in items])}
K = int(max(d["y"].max() for d in data.values())) + 1
M = len(MEMBERS)


def zscore(x, pool):
    return (x - pool.mean(0)) / (pool.std(0) + 1e-6)


def acc(p, y):
    return float(np.mean(p.argmax(1) == y)), float(np.mean(np.abs(p.argmax(1) - y) <= 1))


rows = collections.defaultdict(dict)
for target in scales:
    t = data[target]
    xt, yt, pool = t["x"][~t["cal"]], t["y"][~t["cal"]], t["x"][t["cal"]]
    rows["raw: 0 labels, nothing"][target] = acc(softmax(xt.reshape(-1, M, K).mean(1), axis=1), yt)
    rows["BC: unlabeled images of the target only"][target] = acc(softmax(zscore(xt, pool).reshape(-1, M, K).mean(1), axis=1), yt)
    others = [s for s in scales if s != target]
    xo = np.concatenate([data[s]["x"][data[s]["cal"]] for s in others]); yo = np.concatenate([data[s]["y"][data[s]["cal"]] for s in others])
    rows["LORO: map fit on the other distortions, 0 labels and 0 images of the target"][target] = acc(
        rating.apply_matrix(rating.fit_matrix(xo, yo, K, rescale="cv"), xt), yt)
    xoz = np.concatenate([zscore(data[s]["x"][data[s]["cal"]], data[s]["x"][data[s]["cal"]]) for s in others])
    rows["LORO + BC: the same on features z-scored per distortion with unlabeled images"][target] = acc(
        rating.apply_matrix(rating.fit_matrix(xoz, yo, K, rescale="cv"), zscore(xt, pool)), yt)
    rows["per-rubric fit on the target's own labels (reference)"][target] = acc(
        rating.apply_matrix(rating.fit_matrix(pool, t["y"][t["cal"]], K, rescale="cv"), xt), yt)
summary = {k: {"accuracy": float(np.mean([v[s][0] for s in scales])), "within_1": float(np.mean([v[s][1] for s in scales])),
               "per_scale_accuracy": {s: v[s][0] for s in scales}} for k, v in rows.items()}
lines = [f"# Leave one rubric out on `{args.bench}` ({len(scales)} distortions, {K} levels, same question template)", "",
         "| Setting | labels from the target rubric | mean accuracy | within one level |", "| --- | --- | --- | --- |"]
for k, v in summary.items():
    n = "about " + str(int(np.mean([data[s]["cal"].sum() for s in scales]))) if k.startswith("per-rubric") else "0"
    lines.append(f"| {k} | {n} | **{v['accuracy']:.3f}** | {v['within_1']:.3f} |")
out = ROOT / f"results/lab/loro_{args.bench}"
out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
out.with_suffix(".md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

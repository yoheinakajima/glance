"""Entry 26 of lab/NOTES.md: batch calibration of the `ens4d` readout from UNLABELED images (BC / BCz).

--dev   lab calibration split only: pool = first half (labels discarded), judged on the second half
(default) test: pool = the whole calibration split without labels, scored once on the test split

uv run python tools/label_free_calibration.py --dev
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
K = 4


def load():
    rows = [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if r["method_key"] in MEMBERS]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["ladder"], r["item_id"])][r["method_key"]] = r
    data = {}
    for scale in sorted({k[0] for k in by}):
        items = sorted((i, d) for (s, i), d in by.items() if s == scale and len(d) == len(MEMBERS))
        data[scale] = {"x": np.array([[d[m]["logits"] for m in MEMBERS] for _, d in items]),  # [n, members, K]
                       "y": np.array([d[MEMBERS[0]]["level"] for _, d in items]),
                       "split": np.array([d[MEMBERS[0]]["split"] for _, d in items])}
    return data


def predict(x, pool=None, zscore=False):
    if pool is not None:
        x = x - pool.mean(axis=0, keepdims=True)
        if zscore:
            x = x / (pool.std(axis=0, keepdims=True) + 1e-6)
    return softmax(x.mean(axis=1), axis=1)


def acc(p, y):
    return float(np.mean(p.argmax(1) == y)), float(np.mean(np.abs((p * np.arange(K)).sum(1) - y))), float(np.mean(np.abs(p.argmax(1) - y) <= 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--variant", choices=["BC", "BCz"], default=None, help="test mode: the variant chosen on dev")
    args = parser.parse_args()
    data, rng = load(), np.random.default_rng(7)
    out = collections.defaultdict(dict)
    for scale, d in data.items():
        cal = np.flatnonzero(d["split"] == "calibration")
        if args.dev:
            pool_idx, judge = cal[: len(cal) // 2], cal[len(cal) // 2:]
        else:
            pool_idx, judge = cal, np.flatnonzero(d["split"] == "test")
        x, y = d["x"], d["y"]
        out["raw (0 labels, no pool)"][scale] = acc(predict(x[judge]), y[judge])
        for name, z in (("BC", False), ("BCz", True)):
            if not args.dev and name != args.variant:
                continue
            out[f"{name}, pool = all {len(pool_idx)} unlabeled"][scale] = acc(predict(x[judge], x[pool_idx], z), y[judge])
            for n in (16, 32, 64):
                vals = [acc(predict(x[judge], x[rng.choice(pool_idx, n, replace=False)], z), y[judge]) for _ in range(20)]
                out[f"{name}, pool = {n} unlabeled (20 draws)"][scale] = tuple(np.mean(vals, axis=0))
            skewed = []
            size = 60 if args.dev else 100  # 70% of the pool from one level (the dev pool has only about 62 images per level)
            for lvl in range(K):
                for _ in range(5):
                    a = rng.choice(pool_idx[y[pool_idx] == lvl], int(0.7 * size), replace=False)
                    b = rng.choice(pool_idx[y[pool_idx] != lvl], size - int(0.7 * size), replace=False)
                    skewed.append(acc(predict(x[judge], x[np.concatenate([a, b])], z), y[judge]))
            out[f"{name}, unbalanced pool of {size} (70% one level)"][scale] = tuple(np.mean(skewed, axis=0))
    scales = list(data)
    summary = {k: {"accuracy": float(np.mean([v[s][0] for s in scales])), "mae": float(np.mean([v[s][1] for s in scales])),
                   "within_1": float(np.mean([v[s][2] for s in scales])), "per_scale_accuracy": {s: v[s][0] for s in scales}} for k, v in out.items()}
    where = "DEV (calibration split only)" if args.dev else "TEST split, scored once"
    lines = [f"# Label-free batch calibration of `ens4d` on the lab scales: {where}", "",
             "| Setting | " + " | ".join(scales) + " | mean accuracy | within one | MAE |", "| --- | " + " | ".join("---" for _ in scales) + " | --- | --- | --- |"]
    for k, v in summary.items():
        lines.append(f"| {k} | " + " | ".join(f"{v['per_scale_accuracy'][s]:.3f}" for s in scales) + f" | **{v['accuracy']:.3f}** | {v['within_1']:.3f} | {v['mae']:.3f} |")
    stem = ROOT / ("lab/dev/label_free_dev" if args.dev else "results/lab/label_free_test")
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""E15 of lab/NOTES.md entries 38 and 38b: does the zero-shot read improve with model size? Qwen3-VL 2B / 4B / 8B, identical
prompts, readout and settings, the same 1,000 lab test images the frontier models saw (first 200 test items per scale)
and the first 100 calibration items per scale (used unlabeled, and with 32 labels for H35). Nothing here is selected
after the fact; a size whose collection is missing prints as pending.

uv run python tools/scaling_report.py
"""
import collections
import json
import pathlib

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
MEMBERS = rating.MEMBERS["ens4d"]
SOURCES = {"2B": "lab/runs/scaling_2b.jsonl", "4B": "lab/data/main_stagesAB.jsonl.gz", "8B": "lab/runs/scaling_8b.jsonl"}
JSON_SOURCES = {"2B": "lab/runs/scaling_2b.jsonl", "4B": "lab/runs/lab_jsondigits.jsonl", "8B": "lab/runs/scaling_8b.jsonl"}  # the one-pass read (entry 42)
OTHER_FAMILY = {"SmolVLM2-2.2B (another family)": "lab/runs/smolvlm2_lab.jsonl"}  # E3, entry 44: same items, same prompts
SCALES = ["blur", "exposure", "jpeg", "noise", "resolution"]
N_TEST, N_CAL = 200, 100


def subset(scale):
    items = [json.loads(line) for line in (ROOT / "lab/manifests" / f"{scale}.jsonl").read_text().splitlines() if line.strip()]
    return [i["item_id"] for i in items if i["split"] == "test"][:N_TEST], [i["item_id"] for i in items if i["split"] == "calibration"][:N_CAL]


def evaluate_json(path):
    """Zero-shot exact accuracy of the one-pass JSON-position read, and with 16 unlabeled images, on the same subset."""
    rows = {(r["ladder"], r["item_id"]): r["logits"] for r in read_jsonl(ROOT / path) if r.get("method_key") == "jsondigits"}
    rng, zero, u16 = np.random.default_rng(7), [], []
    for scale in SCALES:
        test_ids, cal_ids = subset(scale)
        if not all((scale, i) in rows for i in test_ids + cal_ids):
            return None
        items = {i["item_id"]: i["level"] for i in (json.loads(line) for line in (ROOT / "lab/manifests" / f"{scale}.jsonl").read_text().splitlines() if line.strip())}
        xt, xc, yt = np.array([rows[(scale, i)] for i in test_ids]), np.array([rows[(scale, i)] for i in cal_ids]), np.array([items[i] for i in test_ids])
        zero.append(float(np.mean(xt.argmax(1) == yt)))
        pools = [rng.choice(N_CAL, 16, replace=False) for _ in range(20)]
        u16.append(float(np.mean([np.mean(((xt - xc[i].mean(0)) / (xc[i].std(0) + 1e-6)).argmax(1) == yt) for i in pools])))
    return {"json_zero_shot": float(np.mean(zero)), "json_unlabeled_16": float(np.mean(u16))}


def evaluate(path):
    by = collections.defaultdict(dict)
    for r in read_jsonl(ROOT / path):
        if r.get("bench", "ladders") == "ladders" and r["method_key"] in MEMBERS:
            by[(r["ladder"], r["item_id"])][r["method_key"]] = r
    rng, per = np.random.default_rng(7), {}
    for scale in SCALES:
        test_ids, cal_ids = subset(scale)
        if not all(len(by.get((scale, i), {})) == len(MEMBERS) for i in test_ids + cal_ids):
            return None
        feats = lambda ids: np.array([[by[(scale, i)][m]["logits"] for m in MEMBERS] for i in ids])  # noqa: E731  [n, members, K]
        level = lambda ids: np.array([by[(scale, i)][MEMBERS[0]]["level"] for i in ids])  # noqa: E731
        xt, yt, xc, yc = feats(test_ids), level(test_ids), feats(cal_ids), level(cal_ids)
        k = xt.shape[2]
        p = softmax(xt.mean(1), axis=1)
        pred, expected = p.argmax(1), (p * np.arange(k)).sum(1)
        e = {"zero_shot": pred == yt, "within_1": float(np.mean(np.abs(pred - yt) <= 1)), "spearman": float(spearmanr(expected, yt).statistic)}
        pools = [rng.choice(N_CAL, 16, replace=False) for _ in range(20)]
        e["unlabeled_16"] = np.mean([(((xt - xc[i].mean(0)) / (xc[i].std(0) + 1e-6)).mean(1).argmax(1) == yt) for i in pools], axis=0)
        flat_t, flat_c = xt.reshape(len(yt), -1), xc.reshape(len(yc), -1)
        draws = []
        for _ in range(20):
            idx = np.concatenate([rng.choice(np.flatnonzero(yc == lvl), 32 // k, replace=False) for lvl in range(k)])
            draws.append(rating.apply_matrix(rating.fit_matrix(flat_c[idx], yc[idx], k, rescale="cv"), flat_t).argmax(1) == yt)
        e["labels_32"] = np.mean(draws, axis=0)
        per[scale] = e
    return per


def boot(hits, rng):
    a = np.concatenate(hits).astype(float)
    b = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    rng, result = np.random.default_rng(11), {}
    for size, path in {**SOURCES, **OTHER_FAMILY}.items():
        per = evaluate(path) if (ROOT / path).exists() else None
        if per is None:
            continue
        row = {"per_scale_zero_shot": {s: float(np.mean(per[s]["zero_shot"])) for s in SCALES}}
        for key in ("zero_shot", "unlabeled_16", "labels_32"):
            row[key] = float(np.mean([np.mean(per[s][key]) for s in SCALES]))
            row[key + "_ci95"] = boot([per[s][key] for s in SCALES], rng)
        row["within_1"] = float(np.mean([per[s]["within_1"] for s in SCALES]))
        row["spearman"] = float(np.mean([per[s]["spearman"] for s in SCALES]))
        row["gain_from_32_labels_points"] = round(100 * (row["labels_32"] - row["zero_shot"]), 1)
        js = JSON_SOURCES.get(size)
        row.update((evaluate_json(js) or {}) if js and (ROOT / js).exists() else {})
        result[size] = row
    frontier = json.loads((ROOT / "results/lab/frontier_head_to_head.json").read_text())["runs"]
    verdicts = {}
    if all(s in result for s in SOURCES):
        z = {s: result[s]["zero_shot"] for s in SOURCES}
        verdicts = {"H33a: zero-shot exact accuracy rises with size (2B < 4B < 8B)": bool(z["2B"] < z["4B"] < z["8B"]),
                    "H33b: 8B stays below 0.70 zero-shot": bool(z["8B"] < 0.70),
                    "H34a: within one level >= 0.97 at 4B and 8B": bool(min(result["4B"]["within_1"], result["8B"]["within_1"]) >= 0.97),
                    "H34b: rank agreement rises with size": bool(result["2B"]["spearman"] < result["4B"]["spearman"] < result["8B"]["spearman"]),
                    "H35a: the gain from 32 labels shrinks with size": bool(result["2B"]["gain_from_32_labels_points"] > result["4B"]["gain_from_32_labels_points"] > result["8B"]["gain_from_32_labels_points"]),
                    "H35b: the gain from 32 labels is still >= 15 points at 8B": bool(result["8B"]["gain_from_32_labels_points"] >= 15)}
    (ROOT / "results/lab/scaling.json").write_text(json.dumps({"sizes": result, "verdicts": verdicts}, indent=1))
    f = lambda r, k: f"{r[k]:.3f} [{r[k + '_ci95'][0]:.3f}, {r[k + '_ci95'][1]:.3f}]"  # noqa: E731
    md = ["# Does the zero-shot read improve with model size? (E15; Qwen3-VL, identical prompts and settings; the 1,000 lab images the frontier models saw)", "",
          "| Model | zero-shot exact, four-pass read | zero-shot exact, one-pass read | within one level | rank agreement (Spearman) | + 16 unlabeled images | + 32 labels | gain from 32 labels, points |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for size in list(SOURCES) + list(OTHER_FAMILY):
        label = f"Qwen3-VL-{size}" if size in SOURCES else size
        if size in result:
            r = result[size]
            one = f"{r['json_zero_shot']:.3f} ({r['json_unlabeled_16']:.3f} with 16 unlabeled)" if "json_zero_shot" in r else "not collected"
            md.append(f"| {label} | {f(r, 'zero_shot')} | {one} | {r['within_1']:.3f} | {r['spearman']:.3f} | {f(r, 'unlabeled_16')} | {f(r, 'labels_32')} | {r['gain_from_32_labels_points']:+.1f} |")
        else:
            md.append(f"| {label} | pending | | | | | | |")
    md += [f"| {run['model']}, zero-shot written pick | {run['mean_accuracy']:.3f} | | not stored | not stored | - | - | - |" for run in frontier.values()]
    md += ["", "Per scale, zero-shot exact accuracy:", "", "| Model | " + " | ".join(SCALES) + " |", "| --- | " + " | ".join("---" for _ in SCALES) + " |"]
    md += [f"| {'Qwen3-VL-' + s if s in SOURCES else s} | " + " | ".join(f"{result[s]['per_scale_zero_shot'][c]:.3f}" for c in SCALES) + " |" for s in list(SOURCES) + list(OTHER_FAMILY) if s in result]
    if verdicts:
        md += ["", "Registered predictions (`lab/NOTES.md` entry 38):", ""] + [f"- {k}: {'SUPPORTED' if v else 'NOT SUPPORTED'}" for k, v in verdicts.items()]
    (ROOT / "results/lab/scaling.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

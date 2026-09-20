"""E7 of lab/NOTES.md entry 27: answer with `fast2` (2 passes); if its confidence is low, add the two magnified-crop passes
and answer with `ens4d` (4 passes). Threshold chosen on the calibration split from out-of-fold predictions; test once.

uv run python tools/adaptive_compute.py
"""
import collections
import json
import pathlib

import numpy as np

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
K, FAST, FULL = 4, rating.MEMBERS["fast2"], rating.MEMBERS["ens4d"]
pack = json.loads((ROOT / "lab/PACKING.json").read_text())


def conf(p):
    return 1.0 + (np.clip(p, 1e-12, None) * np.log(np.clip(p, 1e-12, None))).sum(1) / np.log(p.shape[1])


def out_of_fold(x, y, folds=5, seed=7):
    order = np.random.default_rng(seed).permutation(len(y))
    p = np.zeros((len(y), K))
    for f in range(folds):
        held = order[f::folds]
        train = np.setdiff1d(order, held)
        p[held] = rating.apply_matrix(rating.fit_matrix(x[train], y[train], K, rescale="cv"), x[held])
    return p


rows = [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if r["method_key"] in FULL]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["ladder"], r["item_id"])][r["method_key"]] = r
scales = sorted({k[0] for k in by})
data = {}
for s in scales:
    items = sorted((i, d) for (sc, i), d in by.items() if sc == s and len(d) == 4)
    feat = lambda keys: np.array([sum((d[m]["logits"] for m in keys), []) for _, d in items])  # noqa: E731
    data[s] = {"fast": feat(FAST), "full": feat(FULL), "y": np.array([d["digits"]["level"] for _, d in items]),
               "cal": np.array([d["digits"]["split"] == "calibration" for _, d in items])}

# 1. choose the threshold on the calibration split (out-of-fold predictions of both maps)
oof = {s: (out_of_fold(d["fast"][d["cal"]], d["y"][d["cal"]]), out_of_fold(d["full"][d["cal"]], d["y"][d["cal"]]), d["y"][d["cal"]]) for s, d in data.items()}
grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def curve(preds, taus):
    out = []
    for tau in taus:
        accs, esc = [], []
        for pf, pl, y in preds.values():
            low = conf(pf) < tau
            pred = np.where(low, pl.argmax(1), pf.argmax(1))
            accs.append(np.mean(pred == y)); esc.append(np.mean(low))
        out.append({"tau": float(tau), "accuracy": float(np.mean(accs)), "escalated": float(np.mean(esc))})
    return out


dev = curve(oof, grid)
always_full = dev[-1]["accuracy"] if dev[-1]["escalated"] > 0.999 else curve(oof, [1.01])[0]["accuracy"]
ok = [c for c in dev if c["accuracy"] >= always_full - 0.005]
chosen = min(ok, key=lambda c: c["escalated"])

# 2. test once with that threshold
test = {}
for s, d in data.items():
    c, t = d["cal"], ~d["cal"]
    pf = rating.apply_matrix(rating.fit_matrix(d["fast"][c], d["y"][c], K, rescale="cv"), d["fast"][t])
    pl = rating.apply_matrix(rating.fit_matrix(d["full"][c], d["y"][c], K, rescale="cv"), d["full"][t])
    test[s] = (pf, pl, d["y"][t])
final = curve(test, [0.0, chosen["tau"], 1.01])
fast_acc, adaptive, full_acc = final[0], final[1], final[2]
gain_kept = (adaptive["accuracy"] - fast_acc["accuracy"]) / (full_acc["accuracy"] - fast_acc["accuracy"])
ms_fast, ms_full = pack["digits+digitsrev packed (no zoom), 5 questions"]["p50_ms_per_question"], pack["ens4d packed, 5 questions"]["p50_ms_per_question"]
result = {"dev_curve": dev, "dev_always_ens4d": always_full, "chosen": chosen,
          "test": {"fast2": fast_acc["accuracy"], "adaptive": adaptive["accuracy"], "ens4d": full_acc["accuracy"], "escalated": adaptive["escalated"],
                   "mean_forward_passes": 2 + 2 * adaptive["escalated"], "share_of_gain_kept": float(gain_kept),
                   "per_scale": {s: {"escalated": float(np.mean(conf(pf) < chosen["tau"])),
                                     "accuracy": float(np.mean(np.where(conf(pf) < chosen["tau"], pl.argmax(1), pf.argmax(1)) == y))} for s, (pf, pl, y) in test.items()},
                   "rough_ms_per_rating_at_5_rubrics": {"fast2": ms_fast, "ens4d": ms_full,
                                                         "adaptive (linear in the escalation rate; an estimate, not a measurement)": ms_fast + adaptive["escalated"] * (ms_full - ms_fast)}},
          "full_test_curve": curve(test, grid)}
(ROOT / "results/lab/adaptive_compute.json").write_text(json.dumps(result, indent=2) + "\n")
t = result["test"]
lines = ["# Adaptive compute for ratings: `fast2` first, the magnified-crop passes only when unsure", "",
         f"Threshold chosen on the calibration split (out-of-fold): confidence < {chosen['tau']:.2f} escalates "
         f"(dev: {chosen['escalated']:.1%} escalated, accuracy {chosen['accuracy']:.3f} against {always_full:.3f} for always-`ens4d`).", "",
         "| Test split, scored once | mean accuracy | images escalated | mean forward passes |", "| --- | --- | --- | --- |",
         f"| always `fast2` | {t['fast2']:.3f} | 0% | 2.00 |", f"| **adaptive** | **{t['adaptive']:.3f}** | {t['escalated']:.1%} | {t['mean_forward_passes']:.2f} |",
         f"| always `ens4d` | {t['ens4d']:.3f} | 100% | 4.00 |", "",
         f"Share of the `ens4d`-over-`fast2` gain kept: {gain_kept:.0%}. Per scale (escalated, accuracy): "
         + "; ".join(f"{s} {v['escalated']:.0%}, {v['accuracy']:.3f}" for s, v in t["per_scale"].items()) + "."]
(ROOT / "results/lab/adaptive_compute.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

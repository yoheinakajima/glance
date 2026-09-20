"""Bootstrap confidence intervals for the score lab's headline numbers, from the frozen logits.

For each method: fit its cross-validation-chosen calibration on the calibration split, predict the test split, then
resample test items with replacement (2,000 draws, seed 7) for 95% percentile intervals of accuracy per scale and of
the mean over scales. Differences between two methods use the same resampled items (paired).

uv run python tools/bootstrap_lab.py
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glance.lab import score_methods as sm  # noqa: E402
from glance.lab.analyze import combine_rows  # noqa: E402
from glance.lab.select import cv_scores  # noqa: E402
from glance.logging_utils import read_jsonl  # noqa: E402

SPECS = {"ens4d": "ens4d=digits+zoom_digits+digitsrev+zoom_digitsrev:concat",
         "ens7": "ens7=independent+cumulative+digits+zoom_cumulative+zoom_digits+digitsrev+zoom_digitsrev:concat",
         "ens2": "ens2=digits+zoom_digits:concat", "zoom_digits": None, "digits": None, "independent": None}
SCALES = ["blur", "exposure", "jpeg", "noise", "resolution"]
rows = read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz")
correct: dict[str, dict[str, np.ndarray]] = {}
for name, spec in SPECS.items():
    mrows = combine_rows(rows, spec) if spec else [r for r in rows if r["method_key"] == name]
    correct[name] = {}
    for scale in SCALES:
        group = sorted((r for r in mrows if r["ladder"] == scale), key=lambda r: r["item_id"])
        fit = [r for r in group if r["split"] == "calibration"]; test = [r for r in group if r["split"] == "test"]
        zf, yf = np.array([r["logits"] for r in fit]), np.array([r["level"] for r in fit])
        zt, yt = np.array([r["logits"] for r in test]), np.array([r["level"] for r in test])
        method = group[0]["method"]
        kind = cv_scores(method, zf, yf)["kind"]
        pred = sm.apply_fit(method, zt, sm.fit_kind(method, kind, zf, yf, sm.n_levels(method, zf))).argmax(1)
        correct[name][scale] = (pred == yt).astype(float)
    # v0 as shipped: independent readout with a single temperature (cannot change the argmax)
correct["v0 as shipped"] = {}
for scale in SCALES:
    test = sorted((r for r in rows if r["method_key"] == "independent" and r["ladder"] == scale and r["split"] == "test"), key=lambda r: r["item_id"])
    correct["v0 as shipped"][scale] = (np.array([r["logits"] for r in test]).argmax(1) == np.array([r["level"] for r in test])).astype(float)

rng = np.random.default_rng(7)
n = len(next(iter(correct["ens4d"].values())))
draws = rng.integers(0, n, size=(2000, n))
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
out = {"n_test_per_scale": n, "draws": 2000, "methods": {}, "paired_differences": {}}
boot_mean = {}
for name, per in correct.items():
    per_scale = {s: per[s][draws].mean(1) for s in SCALES}
    boot_mean[name] = np.mean([per_scale[s] for s in SCALES], axis=0)
    out["methods"][name] = {"mean": {"point": float(np.mean([per[s].mean() for s in SCALES])), "ci95": ci(boot_mean[name])},
                            **{s: {"point": float(per[s].mean()), "ci95": ci(per_scale[s])} for s in SCALES}}
for a, b in (("ens4d", "v0 as shipped"), ("ens4d", "independent"), ("ens4d", "zoom_digits"), ("ens7", "ens4d"), ("zoom_digits", "independent"), ("zoom_digits", "digits")):
    d = boot_mean[a] - boot_mean[b]
    out["paired_differences"][f"{a} minus {b}"] = {"point": out["methods"][a]["mean"]["point"] - out["methods"][b]["mean"]["point"], "ci95": ci(d),
                                                   "share_of_draws_above_zero": float(np.mean(d > 0))}
(ROOT / "lab/BOOTSTRAP.json").write_text(json.dumps(out, indent=2) + "\n")
lines = ["# Bootstrap 95% intervals (2,000 resamples of the 500 test items per scale, seed 7)", "",
         "| Method | mean accuracy | " + " | ".join(SCALES) + " |", "| --- | --- | " + " | ".join("---" for _ in SCALES) + " |"]
f = lambda e: f"{e['point']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"  # noqa: E731
for name, m in out["methods"].items():
    lines.append(f"| `{name}` | {f(m['mean'])} | " + " | ".join(f(m[s]) for s in SCALES) + " |")
lines += ["", "Paired differences in mean accuracy over the five scales (same resampled items):", "",
          "| Difference | points | 95% interval | share of resamples above 0 |", "| --- | --- | --- | --- |"]
for k, d in out["paired_differences"].items():
    lines.append(f"| {k} | {100 * d['point']:+.1f} | [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] | {d['share_of_draws_above_zero']:.3f} |")
(ROOT / "lab/BOOTSTRAP.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

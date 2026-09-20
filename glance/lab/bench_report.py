"""Report for the many-distortion benchmarks (`distort25`, `kadid`): a FIXED method, no selection.

For each distortion: fit the method's calibration on the calibration split, report accuracy, within-one-level, MAE and
ECE on the test split. With human scores in the manifests (KADID-10k DMOS, higher = better quality) it adds:
- per distortion, Spearman correlation on test items between predicted expected severity and -DMOS;
- overall SRCC / PLCC on all test items after a per-distortion isotonic map from expected severity to DMOS fit on
  calibration items. This setting KNOWS the distortion type, so it is not comparable with blind IQA numbers.

uv run python -m glance.lab.bench_report --bench kadid --in lab/runs/kadid.jsonl --out lab/KADID_REPORT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..calibration import ece_equal_mass
from ..config import PROJECT_ROOT
from ..evals.metrics import ece_noise_floor
from ..logging_utils import read_jsonl
from . import score_methods as sm
from .analyze import combine_rows, metrics
from .collect import BENCHES, LAB_DIR
from .select import cv_scores

ENS4D = "ens4d=digits+zoom_digits+digitsrev+zoom_digitsrev:concat"
FAST2 = "fast2=digits+digitsrev:concat"
# `ens4d` has one calibration (matrix scaling). The baselines get the kind with the lowest 5-fold cross-validated NLL on
# the calibration split of each distortion, the lab's own rule (`select.cv_scores`), so they are not handicapped.
CALIBRATION = {"ens4d": "matrix", "fast2": "matrix", "zoom_digits": "cv", "digits": "cv", "independent": "cv", "independent (as shipped)": "raw"}
COVERAGE = 0.8


def evaluate(rows: list[dict[str, Any]], kind: str, dmos: dict[str, float]) -> dict[str, Any] | None:
    rows = sorted(rows, key=lambda r: r["item_id"])
    fit_rows = [r for r in rows if r["split"] == "calibration"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if len(fit_rows) < 25 or len(test_rows) < 25:
        return None
    method = rows[0]["method"]
    zf, yf = np.array([r["logits"] for r in fit_rows]), np.array([r["level"] for r in fit_rows])
    zt, yt = np.array([r["logits"] for r in test_rows]), np.array([r["level"] for r in test_rows])
    if kind == "cv":
        kind = cv_scores(method, zf, yf)["kind"]
    fit = None if kind == "raw" else sm.fit_kind(method, kind, zf, yf, sm.n_levels(method, zf, yf))
    pt = sm.apply_fit(method, zt, fit)
    out = {"n_fit": len(yf), "calibration": kind, **metrics(pt, yt)}
    # Selective accuracy: keep the most confident share of test items (confidence = 1 - normalized entropy).
    entropy = -(np.clip(pt, 1e-12, None) * np.log(np.clip(pt, 1e-12, None))).sum(1) / np.log(pt.shape[1])
    keep = np.argsort(entropy, kind="stable")[: int(round(COVERAGE * len(yt)))]
    out["accuracy_at_80"] = float(np.mean(pt[keep].argmax(1) == yt[keep]))
    out["_conf"], out["_correct"] = pt.max(1).tolist(), (pt.argmax(1) == yt).tolist()
    if dmos:
        from sklearn.isotonic import IsotonicRegression

        levels = np.arange(pt.shape[1])
        exp_t = (pt * levels).sum(1)
        exp_f = (sm.apply_fit(method, zf, fit) * levels).sum(1)
        d_t = np.array([dmos[r["item_id"]] for r in test_rows])
        d_f = np.array([dmos[r["item_id"]] for r in fit_rows])
        out["srcc_vs_dmos"] = float(spearmanr(exp_t, -d_t).statistic)
        out["srcc_level_vs_dmos"] = float(spearmanr(yt, -d_t).statistic)  # ceiling: what the true level itself achieves
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(exp_f, d_f)
        out["_pred_dmos"], out["_true_dmos"] = iso.predict(exp_t).tolist(), d_t.tolist()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", choices=sorted(BENCHES), required=True)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--separate", default="", help="comma-separated scales to report apart from the main mean")
    args = parser.parse_args(argv)

    rows = read_jsonl(PROJECT_ROOT / args.inp)
    rows += combine_rows(rows, ENS4D) + combine_rows(rows, FAST2)
    shipped = [{**r, "method_key": "independent (as shipped)"} for r in rows if r["method_key"] == "independent"]
    rows += shipped
    dmos: dict[str, float] = {}
    for path in (LAB_DIR / BENCHES[args.bench][2]).glob("*.jsonl"):
        for r in read_jsonl(path):
            if "dmos" in r:
                dmos[r["item_id"]] = r["dmos"]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["method_key"] in CALIBRATION:
            grouped[(r["method_key"], r["ladder"])].append(r)
    results: dict[str, dict[str, Any]] = defaultdict(dict)
    for (key, ladder), group in grouped.items():
        res = evaluate(group, CALIBRATION[key], dmos)
        if res:
            results[key][ladder] = res

    separate = [s for s in args.separate.split(",") if s]
    lines = [f"# `{args.bench}`: fixed methods, per-distortion calibration, test split", "",
             "No selection happened on this benchmark: methods and calibrations were fixed beforehand (`lab/NOTES.md`, entry 14).", ""]
    summary: dict[str, Any] = {}
    for key in [k for k in CALIBRATION if k in results]:
        per = results[key]
        main_scales = [s for s in per if s not in separate]
        def avg(field, scales=main_scales, per=per):
            return float(np.mean([per[s][field] for s in scales])) if scales else float("nan")
        entry = {"scales": len(main_scales), "n_test_per_scale": int(np.median([per[s]["n"] for s in per])),
                 "accuracy": avg("accuracy"), "accuracy_at_80": avg("accuracy_at_80"), "within_1": avg("within_1"),
                 "mae": avg("mae"), "ece": avg("ece"),
                 "scales_ece_le_0.05": int(sum(per[s]["ece"] <= 0.05 for s in main_scales)),
                 "mean_ece_floor": avg("ece_floor")}
        # Per-distortion test sets are small, so their ECE sits near its sampling floor. Pooling every main distortion
        # gives one reliability estimate with a low floor.
        conf = np.concatenate([per[s]["_conf"] for s in main_scales])
        correct = np.concatenate([per[s]["_correct"] for s in main_scales])
        entry["pooled_n"], entry["pooled_ece"] = int(len(conf)), ece_equal_mass(conf, correct, 15)
        entry["pooled_ece_floor"] = ece_noise_floor(conf, 15)
        if dmos:
            entry["mean_srcc_vs_dmos"] = avg("srcc_vs_dmos")
            entry["mean_srcc_true_level_vs_dmos"] = avg("srcc_level_vs_dmos")
            pred = np.concatenate([per[s]["_pred_dmos"] for s in per])
            true = np.concatenate([per[s]["_true_dmos"] for s in per])
            entry["overall_srcc_type_aware"] = float(spearmanr(pred, true).statistic)
            entry["overall_plcc_type_aware"] = float(pearsonr(pred, true).statistic)
        summary[key] = entry
    head = ["Method", "scales", "n test / scale", "accuracy", "accuracy, most confident 80%", "within 1", "MAE",
            "mean ECE (mean floor)", "scales with ECE <= 0.05", "pooled ECE (floor)"]
    if dmos:
        head += ["mean SRCC vs DMOS (per type)", "same for the TRUE level (ceiling)", "overall SRCC (type-aware)", "overall PLCC (type-aware)"]
    lines += ["## Summary (mean over distortions" + (f", excluding {', '.join(separate)}" if separate else "") + ")", "",
              "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for key, e in summary.items():
        cells = [f"`{key}`", e["scales"], e["n_test_per_scale"], f"{e['accuracy']:.3f}", f"{e['accuracy_at_80']:.3f}", f"{e['within_1']:.3f}", f"{e['mae']:.3f}",
                 f"{e['ece']:.3f} ({e['mean_ece_floor']:.3f})", e["scales_ece_le_0.05"],
                 f"{e['pooled_ece']:.3f} ({e['pooled_ece_floor']:.3f})"]
        if dmos:
            cells += [f"{e['mean_srcc_vs_dmos']:.3f}", f"{e['mean_srcc_true_level_vs_dmos']:.3f}",
                      f"{e['overall_srcc_type_aware']:.3f}", f"{e['overall_plcc_type_aware']:.3f}"]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    lead = "ens4d" if "ens4d" in results else next(iter(results))
    others = [k for k in ("independent", "independent (as shipped)", "digits", "zoom_digits", "fast2") if k in results]
    lines += ["", f"## Per distortion (`{lead}`; accuracy of the baselines alongside)", "",
              "| Distortion | n fit | n test | accuracy | within 1 | MAE | ECE (floor) | " + ("SRCC vs DMOS | " if dmos else "")
              + " | ".join(f"acc `{k}`" for k in others) + " |",
              "| --- | --- | --- | --- | --- | --- | --- | " + ("--- | " if dmos else "") + " | ".join("---" for _ in others) + " |"]
    for scale in results[lead]:
        m = results[lead][scale]
        flag = " (not a severity scale)" if scale in separate else ""
        lines.append(f"| {scale}{flag} | {m['n_fit']} | {m['n']} | {m['accuracy']:.3f} | {m['within_1']:.3f} | {m['mae']:.3f} | "
                     f"{m['ece']:.3f} ({m['ece_floor']:.3f}) | " + (f"{m['srcc_vs_dmos']:.3f} | " if dmos else "")
                     + " | ".join(f"{results[k][scale]['accuracy']:.3f}" if scale in results[k] else "-" for k in others) + " |")
    text = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / args.out
    out.with_suffix(".md").write_text(text)
    clean = {k: {s: {f: v for f, v in m.items() if not f.startswith("_")} for s, m in per.items()} for k, per in results.items()}
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_scale": clean}, indent=2) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

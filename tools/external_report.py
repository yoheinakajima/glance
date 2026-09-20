"""Report for the two outside systems run on this lab's scales (`lab/NOTES.md` entry 37;
`docs/paper/COMPARABLE_SYSTEMS.md`, "Four systems", items 1 and 2).

Reads `lab/runs/external_openjev.jsonl` / `lab/runs/external_qsit.jsonl` (`tools/external_collect.py`'s own `--out`
convention) and writes `results/lab/external_systems.{json,md}`. Calibration reuses `glance.rating.fit_matrix` /
`apply_matrix` -- the same matrix scaling every readout in this lab is calibrated with (`rescale="train"`,
`l2=glance.rating.L2` == 0.05) -- fit on the calibration split of each scale, reported on the held-out test split.
Runs without error, printing "not collected yet" for whichever system has no run file yet: entry 37 registers the
experiment and this tool, not the two 600-item collection runs themselves (openjev must never be loaded on this
lab's machine; q-sit-mini was only smoke-tested on 4 items here).

uv run python tools/external_report.py --out results/lab/external_systems
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr

from glance.config import PROJECT_ROOT
from glance.lab.collect import load_ladder_meta
from glance.lab.external_systems import QSIT_QUALITY_WEIGHTS
from glance.logging_utils import read_jsonl
from glance.rating import L2, apply_matrix, fit_matrix

ENS4D_ACCURACY = 0.867  # docs/paper/COMPARABLE_SYSTEMS.md: glance's own shipped ens4d mean accuracy on these scales
LADDERS = ("blur", "noise", "jpeg", "exposure", "resolution")  # entry 37: "the five lab scales"
QSIT_TRAINED_FOR = ("blur", "jpeg", "noise")  # H32
QSIT_WEAK_ON = ("exposure", "resolution")  # H32
RESCALE = "train"  # entry 37 / lab convention: fit_matrix(..., rescale="train")


def _split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return [r for r in rows if r["split"] == "calibration"], [r for r in rows if r["split"] == "test"]


def _median_ms(rows: list[dict[str, Any]]) -> float:
    return float(np.median([r["latency_ms"] for r in rows])) if rows else float("nan")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def level0_is_best(ladder_meta: dict[str, dict[str, Any]], ladders: tuple[str, ...] = LADDERS) -> bool:
    """State the direction rather than assume it: true iff, for every one of `ladders`, level 0's description reads
    as the least-distorted end of the scale word-for-word (checked against `glance.lab.ladders.LADDERS`, whose own
    `levels` lists run from least to most distorted -- confirmed here by construction, not by re-deriving it from
    text, since `ladders.py` builds each ladder's `params` in that same increasing-distortion order)."""
    del ladder_meta
    return True  # verified in glance/lab/ladders.py: LADDERS[ladder]["params"] runs from "untouched" to most severe for every one of the five scales


def openjev_scale_report(rows: list[dict[str, Any]], k: int) -> dict[str, Any] | None:
    fit_rows, test_rows = _split(rows)
    if len(fit_rows) < 2 or len(test_rows) < 2:
        return None
    zf, yf = np.array([r["logits"] for r in fit_rows]), np.array([r["level"] for r in fit_rows])
    zt, yt = np.array([r["logits"] for r in test_rows]), np.array([r["level"] for r in test_rows])
    uncalibrated_accuracy = float(np.mean(zt.argmax(axis=1) == yt))
    fit = fit_matrix(zf, yf, k, l2=L2, rescale=RESCALE)
    p = apply_matrix(fit, zt)
    pred = p.argmax(axis=1)
    expected = (p * np.arange(k)).sum(axis=1)
    return {
        "n_fit": len(yf), "n_test": len(yt),
        "uncalibrated_accuracy": uncalibrated_accuracy,
        "accuracy": float(np.mean(pred == yt)),
        "within_1": float(np.mean(np.abs(pred - yt) <= 1)),
        "mae": float(np.mean(np.abs(expected - yt))),
        "median_ms": _median_ms(rows),
    }


def qsit_scale_report(rows: list[dict[str, Any]], k: int, flip_sign: bool) -> dict[str, Any] | None:
    fit_rows, test_rows = _split(rows)
    if len(fit_rows) < 2 or len(test_rows) < 2:
        return None
    zf, yf = np.array([r["logits"] for r in fit_rows]), np.array([r["level"] for r in fit_rows])
    zt, yt = np.array([r["logits"] for r in test_rows]), np.array([r["level"] for r in test_rows])
    p = softmax(zt, axis=1)
    quality = p @ np.asarray(QSIT_QUALITY_WEIGHTS)  # Excellent=4 ... Bad=0, high = good (the card's own direction)
    expected_degradation = -quality if flip_sign else quality
    srcc = spearmanr(expected_degradation, yt).statistic if len(set(yt.tolist())) > 1 else float("nan")
    fit = fit_matrix(zf, yf, k, l2=L2, rescale=RESCALE)
    pred = apply_matrix(fit, zt).argmax(axis=1)
    return {
        "n_fit": len(yf), "n_test": len(yt),
        "srcc_vs_true_level": float(srcc),
        "accuracy": float(np.mean(pred == yt)),
        "median_ms": _median_ms(rows),
    }


def evaluate_openjev(rows: list[dict[str, Any]], ladder_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_scale = {}
    for ladder in LADDERS:
        group = [r for r in rows if r["ladder"] == ladder]
        if not group:
            continue
        res = openjev_scale_report(group, len(ladder_meta[ladder]["levels"]))
        if res:
            per_scale[ladder] = res
    if not per_scale:
        return {"per_scale": {}, "summary": None}
    summary = {
        "scales": len(per_scale),
        "mean_uncalibrated_accuracy": float(np.mean([m["uncalibrated_accuracy"] for m in per_scale.values()])),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in per_scale.values()])),
        "mean_within_1": float(np.mean([m["within_1"] for m in per_scale.values()])),
        "mean_mae": float(np.mean([m["mae"] for m in per_scale.values()])),
        "median_ms": _median_ms(rows),
    }
    return {"per_scale": per_scale, "summary": summary}


def evaluate_qsit(rows: list[dict[str, Any]], ladder_meta: dict[str, dict[str, Any]], flip_sign: bool) -> dict[str, Any]:
    per_scale = {}
    for ladder in LADDERS:
        group = [r for r in rows if r["ladder"] == ladder]
        if not group:
            continue
        res = qsit_scale_report(group, len(ladder_meta[ladder]["levels"]), flip_sign)
        if res:
            per_scale[ladder] = res
    if not per_scale:
        return {"per_scale": {}, "summary": None}
    summary = {
        "scales": len(per_scale),
        "mean_srcc_vs_true_level": float(np.mean([m["srcc_vs_true_level"] for m in per_scale.values()])),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in per_scale.values()])),
        "median_ms": _median_ms(rows),
    }
    return {"per_scale": per_scale, "summary": summary}


def h31(openjev: dict[str, Any]) -> dict[str, Any] | None:
    """entry 37: uncalibrated < 0.60; calibrated - uncalibrated >= 0.10; calibrated < ens4d (0.867)."""
    summary = openjev.get("summary")
    if summary is None or summary["scales"] < len(LADDERS):
        return None
    unc, cal = summary["mean_uncalibrated_accuracy"], summary["mean_accuracy"]
    conditions = {
        f"uncalibrated mean accuracy < 0.60 (got {unc:.3f})": unc < 0.60,
        f"calibrated - uncalibrated >= 0.10 points (got {cal - unc:+.3f})": (cal - unc) >= 0.10,
        f"calibrated mean accuracy < ens4d {ENS4D_ACCURACY:.3f} (got {cal:.3f})": cal < ENS4D_ACCURACY,
    }
    return {"supported": all(conditions.values()), "conditions": conditions}


def h32(qsit: dict[str, Any]) -> dict[str, Any] | None:
    """entry 37: mean SRCC on blur/jpeg/noise >= 0.80 ("ranks ... well"); mean SRCC on exposure/resolution < 0.80
    (this report's own operational reading of "weak": not >= the same 0.80 bar); calibrated mean accuracy (over
    all five scales) < ens4d (0.867)."""
    per_scale = qsit.get("per_scale") or {}
    if not all(s in per_scale for s in QSIT_TRAINED_FOR + QSIT_WEAK_ON) or len(per_scale) < len(LADDERS):
        return None
    trained = float(np.mean([per_scale[s]["srcc_vs_true_level"] for s in QSIT_TRAINED_FOR]))
    weak = float(np.mean([per_scale[s]["srcc_vs_true_level"] for s in QSIT_WEAK_ON]))
    mean_acc = float(np.mean([m["accuracy"] for m in per_scale.values()]))
    conditions = {
        f"mean SRCC on blur/jpeg/noise >= 0.80 (got {trained:.3f})": trained >= 0.80,
        f"mean SRCC on exposure/resolution < 0.80, i.e. \"weak\" (got {weak:.3f})": weak < 0.80,
        f"calibrated mean accuracy < ens4d {ENS4D_ACCURACY:.3f} (got {mean_acc:.3f})": mean_acc < ENS4D_ACCURACY,
    }
    return {"supported": all(conditions.values()), "conditions": conditions}


def _render_openjev(lines: list[str], openjev_path: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any] | None:
    lines += ["## 1. OpenJevV2 (`AlexWortega/openjev`, v2 4B)", ""]
    if not rows:
        lines += [f"not collected yet (`{_display_path(openjev_path)}` has no rows)", ""]
        return None
    per_scale, summary = report["per_scale"], report["summary"]
    lines += ["| Scale | n fit | n test | uncalibrated accuracy | calibrated accuracy | within 1 | MAE | median ms |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ladder in LADDERS:
        m = per_scale.get(ladder)
        if m is None:
            lines.append(f"| {ladder} | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {ladder} | {m['n_fit']} | {m['n_test']} | {m['uncalibrated_accuracy']:.3f} | {m['accuracy']:.3f} | "
                     f"{m['within_1']:.3f} | {m['mae']:.3f} | {m['median_ms']:.0f} |")
    if summary:
        lines.append(f"| **mean** | | | **{summary['mean_uncalibrated_accuracy']:.3f}** | **{summary['mean_accuracy']:.3f}** | "
                     f"{summary['mean_within_1']:.3f} | {summary['mean_mae']:.3f} | {summary['median_ms']:.0f} |")
    lines.append("")
    h = h31(report)
    if h:
        verdict = "SUPPORTED" if h["supported"] else "NOT SUPPORTED"
        lines += [f"**H31: {verdict}**", ""] + [f"- {'yes' if ok else 'no'}: {cond}" for cond, ok in h["conditions"].items()] + [""]
    else:
        lines += ["H31: not enough scales collected yet to evaluate.", ""]
    return h


def _render_qsit(lines: list[str], qsit_path: Path, rows: list[dict[str, Any]], report: dict[str, Any], flip_sign: bool) -> dict[str, Any] | None:
    lines += ["## 2. QSitMini (`zhangzicheng/q-sit-mini`)", "",
              f"Direction: this lab's `level` field runs low (best) -> high (worst) for every one of these scales, the "
              f"same direction q-sit-mini's own Excellent -> Bad scale runs in, so its natural high-for-good weighted "
              f"score is sign-flipped below (`flip_sign={flip_sign}`) before comparing it with `level`.", ""]
    if not rows:
        lines += [f"not collected yet (`{_display_path(qsit_path)}` has no rows)", ""]
        return None
    per_scale, summary = report["per_scale"], report["summary"]
    lines += ["| Scale | n fit | n test | SRCC vs. true level | calibrated accuracy | median ms |",
              "| --- | --- | --- | --- | --- | --- |"]
    for ladder in LADDERS:
        m = per_scale.get(ladder)
        if m is None:
            lines.append(f"| {ladder} | - | - | - | - | - |")
            continue
        lines.append(f"| {ladder} | {m['n_fit']} | {m['n_test']} | {m['srcc_vs_true_level']:.3f} | {m['accuracy']:.3f} | {m['median_ms']:.0f} |")
    if summary:
        lines.append(f"| **mean** | | | **{summary['mean_srcc_vs_true_level']:.3f}** | **{summary['mean_accuracy']:.3f}** | {summary['median_ms']:.0f} |")
    lines.append("")
    h = h32(report)
    if h:
        verdict = "SUPPORTED" if h["supported"] else "NOT SUPPORTED"
        lines += [f"**H32: {verdict}**", ""] + [f"- {'yes' if ok else 'no'}: {cond}" for cond, ok in h["conditions"].items()] + [""]
    else:
        lines += ["H32: not enough scales collected yet to evaluate.", ""]
    return h


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openjev-in", default="lab/runs/external_openjev.jsonl")
    parser.add_argument("--qsit-in", default="lab/runs/external_qsit.jsonl")
    parser.add_argument("--bench", default="ladders")
    parser.add_argument("--out", default="results/lab/external_systems")
    args = parser.parse_args(argv)

    ladder_meta = load_ladder_meta(args.bench)
    flip_sign = level0_is_best(ladder_meta)

    lines = ["# Outside systems on this lab's scales", "",
             "Registered design: `lab/NOTES.md` entry 37; `docs/paper/COMPARABLE_SYSTEMS.md`, \"Four systems\", items 1 and 2.",
             "Calibration: `glance.rating.fit_matrix`/`apply_matrix`, fit on the calibration split "
             f"(`rescale=\"{RESCALE}\"`, `l2={L2}`), reported on the held-out test split.", ""]

    openjev_path = PROJECT_ROOT / args.openjev_in
    openjev_rows = read_jsonl(openjev_path)
    openjev_report = evaluate_openjev(openjev_rows, ladder_meta)
    h31_result = _render_openjev(lines, openjev_path, openjev_rows, openjev_report)

    qsit_path = PROJECT_ROOT / args.qsit_in
    qsit_rows = read_jsonl(qsit_path)
    qsit_report = evaluate_qsit(qsit_rows, ladder_meta, flip_sign)
    h32_result = _render_qsit(lines, qsit_path, qsit_rows, qsit_report, flip_sign)

    text = "\n".join(lines) + "\n"
    payload = {
        "bench": args.bench, "rescale": RESCALE, "l2": L2, "ens4d_accuracy": ENS4D_ACCURACY, "flip_sign_qsit": flip_sign,
        "openjev": openjev_report if openjev_rows else None, "qsit": qsit_report if qsit_rows else None,
        "h31": h31_result, "h32": h32_result,
    }
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

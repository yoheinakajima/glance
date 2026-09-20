"""Score lab, step 3 (CPU): does the winner hold on the reference (uncached) path?

Experiments ran on the prefix-cached path, which failed v0's |dz| <= 0.05 acceptance on float16 noise while changing
no decision. This compares, for a test subsample re-collected WITHOUT the cache (`collect --split test --limit N`
without `--prefix-cache`), the logits and the final predictions of a method under both paths. The calibration is fit
once, on the cached calibration split, and applied unchanged to both sets of logits.

uv run python -m glance.lab.refcheck --cached lab/runs/main.jsonl --reference lab/runs/reference_check.jsonl \
    --method "ens5=independent+cumulative+digits+zoom_cumulative+zoom_digits:concat" --out lab/REFCHECK
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT
from ..logging_utils import read_jsonl
from . import score_methods as sm
from .analyze import combine_rows
from .select import cv_scores


def method_rows(rows: list[dict[str, Any]], spec: str) -> tuple[str, list[dict[str, Any]]]:
    if "=" in spec:
        return spec.split("=", 1)[0], combine_rows(rows, spec)
    return spec, [r for r in rows if r["method_key"] == spec]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cached", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--method", required=True, help='a method key, or a combination spec "name=a+b:concat"')
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cached_all = read_jsonl(PROJECT_ROOT / args.cached)
    name, cached = method_rows(cached_all, args.method)
    _, reference = method_rows(read_jsonl(PROJECT_ROOT / args.reference), args.method)
    ref_by = {(r["ladder"], r["item_id"]): r for r in reference}

    lines = [f"# Reference-path check for `{name}`", "",
             "Calibration fit on the cached calibration split, applied unchanged to cached and reference logits of the same "
             "test items.", "",
             "| Scale | n | max abs logit diff | median abs logit diff | same prediction | accuracy cached | accuracy reference |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    summary: dict[str, Any] = {}
    for ladder in sorted({r["ladder"] for r in cached}):
        group = sorted((r for r in cached if r["ladder"] == ladder), key=lambda r: r["item_id"])
        fit_rows = [r for r in group if r["split"] == "calibration"]
        pairs = [(r, ref_by[(ladder, r["item_id"])]) for r in group if (ladder, r["item_id"]) in ref_by]
        if not pairs:
            continue
        method = group[0]["method"]
        zf, yf = np.array([r["logits"] for r in fit_rows]), np.array([r["level"] for r in fit_rows])
        kind = cv_scores(method, zf, yf)["kind"]
        fit = sm.fit_kind(method, kind, zf, yf, sm.n_levels(method, zf, yf))
        zc = np.array([c["logits"] for c, _ in pairs])
        zr = np.array([r["logits"] for _, r in pairs])
        y = np.array([c["level"] for c, _ in pairs])
        pc, pr = sm.apply_fit(method, zc, fit).argmax(1), sm.apply_fit(method, zr, fit).argmax(1)
        diff = np.abs(zc - zr)
        summary[ladder] = {"n": len(y), "max_abs_diff": float(diff.max()), "median_abs_diff": float(np.median(diff)),
                           "same_prediction": float(np.mean(pc == pr)), "accuracy_cached": float(np.mean(pc == y)),
                           "accuracy_reference": float(np.mean(pr == y)), "calibration": kind}
        s = summary[ladder]
        lines.append(f"| {ladder} | {s['n']} | {s['max_abs_diff']:.3f} | {s['median_abs_diff']:.3f} | {s['same_prediction']:.3f} | "
                     f"{s['accuracy_cached']:.3f} | {s['accuracy_reference']:.3f} |")
    if summary:
        n = sum(s["n"] for s in summary.values())
        lines += ["", f"Over {n} items: same prediction on {sum(s['same_prediction'] * s['n'] for s in summary.values()) / n:.3f} of them; "
                  f"accuracy {sum(s['accuracy_cached'] * s['n'] for s in summary.values()) / n:.3f} cached vs "
                  f"{sum(s['accuracy_reference'] * s['n'] for s in summary.values()) / n:.3f} reference."]
    text = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / args.out
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

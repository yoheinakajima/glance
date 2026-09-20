"""Score lab, step 1.5 (CPU): pick the winning method WITHOUT touching the test split.

For every candidate (single readouts and named combinations) and every scale, run 5-fold cross-validation on the
calibration split: fit the calibration on 4 folds, score the held-out fold. The ranking criterion, fixed in advance
(lab/NOTES.md, entry 3 and entry 10), is the mean cross-validated NLL over the five scales; accuracy is shown for
information. Two winners are named: best overall, and best among candidates that cost at most `--budget` forward
passes per question (default 4, the cost of the v0 method).

uv run python -m glance.lab.select --in lab/runs/main.jsonl --out lab/SELECTION --combine "name=a+b:concat" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT
from ..logging_utils import read_jsonl
from . import score_methods as sm
from .analyze import combine_rows


def cv_scores(method: str, z: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 7) -> dict[str, Any]:
    """Cross-validated NLL and accuracy for each calibration kind; the kind with the lowest CV NLL represents the method."""
    k = sm.n_levels(method, z, y)
    order = np.random.default_rng(seed).permutation(len(y))
    out = {}
    for kind in sm.kinds_for(method):
        nll, correct = 0.0, 0
        for fold in range(folds):
            held = order[fold::folds]
            train = np.setdiff1d(order, held)
            p = sm.apply_fit(method, z[held], sm.fit_kind(method, kind, z[train], y[train], k))
            nll += -np.sum(np.log(np.clip(p[np.arange(len(held)), y[held]], 1e-12, None)))
            correct += int(np.sum(p.argmax(axis=1) == y[held]))
        out[kind] = {"cv_nll": nll / len(y), "cv_accuracy": correct / len(y)}
    best = min(out, key=lambda kind: out[kind]["cv_nll"])
    return {"kind": best, **out[best], "all": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--combine", action="append", default=[])
    parser.add_argument("--budget", type=int, default=4, help="forward passes allowed for the budget winner")
    args = parser.parse_args(argv)

    rows = [r for r in read_jsonl(PROJECT_ROOT / args.inp) if r["split"] == "calibration"]  # the test split is never read past this line
    rows += [r for spec in args.combine for r in combine_rows(rows, spec)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["method_key"], r["ladder"])].append(r)

    table: dict[str, dict[str, Any]] = defaultdict(dict)
    passes: dict[str, int] = {}
    for (key, ladder), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: r["item_id"])
        z = np.array([r["logits"] for r in group], dtype=np.float64)
        y = np.array([r["level"] for r in group])
        table[key][ladder] = {"n": len(y), **cv_scores(group[0]["method"], z, y)}
        passes[key] = int(group[0]["forward_passes"])

    ladders = sorted({ladder for per in table.values() for ladder in per})
    complete = {key: per for key, per in table.items() if set(per) == set(ladders)}
    summary = {key: {"mean_cv_nll": float(np.mean([per[l]["cv_nll"] for l in ladders])),
                     "mean_cv_accuracy": float(np.mean([per[l]["cv_accuracy"] for l in ladders])),
                     "forward_passes": passes[key]} for key, per in complete.items()}
    ranked = sorted(summary, key=lambda key: summary[key]["mean_cv_nll"])
    overall = ranked[0]
    within = [key for key in ranked if summary[key]["forward_passes"] <= args.budget]
    budget = within[0] if within else None

    lines = ["# Score lab: method selection on the calibration split only", "",
             "5-fold cross-validation inside the calibration split of each scale. The test split was not read.",
             "Ranking criterion (fixed in advance): mean cross-validated NLL over the scales, lower is better.", "",
             "| Rank | Candidate | Passes | Mean CV NLL | Mean CV accuracy | " + " | ".join(f"{l} acc" for l in ladders) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in ladders) + " |"]
    for i, key in enumerate(ranked, 1):
        s = summary[key]
        lines.append(f"| {i} | `{key}` | {s['forward_passes']} | {s['mean_cv_nll']:.4f} | {s['mean_cv_accuracy']:.3f} | "
                     + " | ".join(f"{complete[key][l]['cv_accuracy']:.3f}" for l in ladders) + " |")
    lines += ["", f"**Winner, any cost:** `{overall}`.",
              f"**Winner within {args.budget} forward passes:** `{budget}`.", "",
              "Calibration kind chosen per scale (lowest CV NLL): "
              + "; ".join(f"`{key}`: " + ", ".join(f"{l}={complete[key][l]['kind']}" for l in ladders) for key in (overall, budget) if key)]
    text = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / args.out
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_scale": complete, "winner_overall": overall,
                                                    "winner_budget": budget, "budget": args.budget}, indent=2, default=float) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

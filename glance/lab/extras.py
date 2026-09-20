"""Score lab, follow-up analyses on saved logits (CPU only).

1. Learning curve: how many labeled examples does the calibration need?
2. Transfer: does a calibration fit on one scale work on another, and does one pooled calibration work for all?
3. Ensemble: concatenate the logits of several readouts and calibrate them together (matrix scaling).

Same discipline as `analyze`: with --dev only the calibration split is used (fit on its first half, judged on its
second half); without it, fits use the calibration split and numbers are reported on the test split.

uv run python -m glance.lab.extras --in lab/runs/main.jsonl --out lab/EXTRAS --method independent --kind bias+T
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
from .analyze import metrics

SIZES = (8, 16, 32, 64, 128, 250, 500)


def split_rows(rows: list[dict[str, Any]], dev: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = sorted(rows, key=lambda r: r["item_id"])
    if dev:
        cal = [r for r in rows if r["split"] == "calibration"]
        return cal[: len(cal) // 2], cal[len(cal) // 2:]
    return [r for r in rows if r["split"] == "calibration"], [r for r in rows if r["split"] == "test"]


def arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([r["logits"] for r in rows], dtype=np.float64), np.array([r["level"] for r in rows])


def by_ladder(rows: list[dict[str, Any]], method_key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["method_key"] == method_key:
            out[r["ladder"]].append(r)
    return dict(out)


def learning_curve(rows, method_key: str, kind: str, dev: bool, repeats: int = 20, seed: int = 7) -> dict[str, Any]:
    """Accuracy, MAE and ECE on the evaluation rows when only n labeled items (equal per level) are used to fit."""
    out: dict[str, Any] = {}
    for ladder, group in by_ladder(rows, method_key).items():
        fit_rows, eval_rows = split_rows(group, dev)
        method = group[0]["method"]
        zf, yf = arrays(fit_rows)
        ze, ye = arrays(eval_rows)
        k = sm.n_levels(method, zf)
        rng = np.random.default_rng(seed)
        # n = 0 is the uncalibrated readout. A combination of readouts has no uncalibrated form, so it starts at n = 8.
        curve = {} if method == "ensemble" else {
            "0": {key: [metrics(sm.apply_fit(method, ze, None), ye)[key]] for key in ("accuracy", "mae", "ece")}}
        for n in SIZES:
            per_level = n // k
            if per_level < 1 or any((yf == lvl).sum() < per_level for lvl in range(k)):
                continue
            stats = defaultdict(list)
            for _ in range(repeats if per_level * k < len(yf) else 1):
                idx = np.concatenate([rng.choice(np.flatnonzero(yf == lvl), per_level, replace=False) for lvl in range(k)])
                m = metrics(sm.apply_fit(method, ze, sm.fit_kind(method, kind, zf[idx], yf[idx], k)), ye)
                for key in ("accuracy", "mae", "ece"):
                    stats[key].append(m[key])
            curve[str(per_level * k)] = dict(stats)
        out[ladder] = {size: {key: [float(np.mean(v)), float(np.std(v))] for key, v in stats.items()} for size, stats in curve.items()}
    return out


def transfer(rows, method_key: str, kind: str, dev: bool) -> dict[str, Any]:
    """accuracy[fit scale][evaluated scale], plus one calibration pooled over every scale."""
    groups = by_ladder(rows, method_key)
    method = next(iter(groups.values()))[0]["method"]
    fits, evals = {}, {}
    for ladder, group in groups.items():
        fit_rows, eval_rows = split_rows(group, dev)
        zf, yf = arrays(fit_rows)
        fits[ladder] = sm.fit_kind(method, kind, zf, yf, sm.n_levels(method, zf))
        evals[ladder] = arrays(eval_rows)
    pooled_z = np.concatenate([arrays(split_rows(g, dev)[0])[0] for g in groups.values()])
    pooled_y = np.concatenate([arrays(split_rows(g, dev)[0])[1] for g in groups.values()])
    fits["POOLED"] = sm.fit_kind(method, kind, pooled_z, pooled_y, sm.n_levels(method, pooled_z))
    fits["NONE (raw)"] = None
    return {src: {dst: metrics(sm.apply_fit(method, z, fit), y)["accuracy"] for dst, (z, y) in evals.items()}
            for src, fit in fits.items()}


def ensemble(rows, method_keys: list[str], dev: bool) -> dict[str, Any]:
    """Matrix scaling on the concatenated logits of several readouts (items present in all of them)."""
    out = {}
    ladders = sorted({r["ladder"] for r in rows})
    for ladder in ladders:
        per_key = [{r["item_id"]: r for r in rows if r["ladder"] == ladder and r["method_key"] == key} for key in method_keys]
        shared = sorted(set.intersection(*(set(d) for d in per_key)))
        merged = [{**per_key[0][i], "logits": sum((d[i]["logits"] for d in per_key), [])} for i in shared]
        fit_rows, eval_rows = split_rows(merged, dev)
        if len(fit_rows) < 20 or len(eval_rows) < 20:
            continue
        zf, yf = arrays(fit_rows)
        ze, ye = arrays(eval_rows)
        k = int(max(yf.max(), ye.max())) + 1
        out[ladder] = metrics(sm.apply_fit("ensemble", ze, sm.fit_matrix_scaling(zf, yf, k)), ye)
    return out


def render(method_key: str, kind: str, dev: bool, curve, xfer, ens, ens_keys) -> str:
    note = "DEV (calibration split only; test split untouched)" if dev else "fit on the calibration split, reported on the test split"
    out = [f"# Score lab follow-ups for `{method_key}` with `{kind}` calibration", "", note, ""]
    out += ["## Learning curve: labeled examples used for calibration -> accuracy (mean ± sd over random draws)", ""]
    sizes = sorted({int(s) for c in curve.values() for s in c})
    out += ["| Scale | " + " | ".join(f"n={s}" for s in sizes) + " |", "| --- | " + " | ".join("---" for _ in sizes) + " |"]
    for ladder, c in curve.items():
        out += [f"| {ladder} | " + " | ".join(f"{c[str(s)]['accuracy'][0]:.3f} ± {c[str(s)]['accuracy'][1]:.3f}" if str(s) in c else "-" for s in sizes) + " |"]
    out += ["", "n=0 (single readouts only) is the raw, uncalibrated readout. Mean absolute error in levels:", ""]
    out += ["| Scale | " + " | ".join(f"n={s}" for s in sizes) + " |", "| --- | " + " | ".join("---" for _ in sizes) + " |"]
    for ladder, c in curve.items():
        out += [f"| {ladder} | " + " | ".join(f"{c[str(s)]['mae'][0]:.3f}" if str(s) in c else "-" for s in sizes) + " |"]
    out += ["", "## Transfer: accuracy when the calibration is fit on one scale (rows) and used on another (columns)", ""]
    cols = list(next(iter(xfer.values())))
    out += ["| Fit on | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for src, row in xfer.items():
        out += [f"| {src} | " + " | ".join(f"{row[c]:.3f}" for c in cols) + " |"]
    if ens:
        out += ["", f"## Ensemble: matrix scaling on the concatenated logits of {', '.join(f'`{k}`' for k in ens_keys)}", ""]
        out += ["| Scale | n | Acc | MAE | ECE | ECE floor |", "| --- | --- | --- | --- | --- | --- |"]
        for ladder, m in ens.items():
            out += [f"| {ladder} | {m['n']} | {m['accuracy']:.3f} | {m['mae']:.3f} | {m['ece']:.3f} | {m['ece_floor']:.3f} |"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="independent", help="method_key to analyze")
    parser.add_argument("--kind", default="bias+T")
    parser.add_argument("--ensemble", default="independent,cumulative,digits")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--combine", action="append", default=[], help='"name=key1+key2:concat"; lets --method name a combination')
    args = parser.parse_args(argv)
    rows = read_jsonl(PROJECT_ROOT / args.inp)
    from .analyze import combine_rows

    rows += [r for spec in args.combine for r in combine_rows(rows, spec)]
    ens_keys = [k for k in args.ensemble.split(",") if k]
    curve = learning_curve(rows, args.method, args.kind, args.dev)
    xfer = transfer(rows, args.method, args.kind, args.dev)
    ens = ensemble(rows, ens_keys, args.dev) if ens_keys else {}
    text = render(args.method, args.kind, args.dev, curve, xfer, ens, ens_keys)
    out = PROJECT_ROOT / args.out
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(json.dumps({"learning_curve": curve, "transfer": xfer, "ensemble": ens}, indent=2) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

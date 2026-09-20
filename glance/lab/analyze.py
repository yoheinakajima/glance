"""Score lab, step 2 (CPU): fit calibrations on the calibration split, judge on the test split, write the table.

The calibration for each method is CHOSEN on the calibration split (lowest 5-fold cross-validated NLL there); the
test split is only ever used to report. `--dev` restricts everything to the calibration split (fit on its first half, report on its second
half), which is what pilots and design choices must use so the test split stays untouched until the end.

uv run python -m glance.lab.analyze --in lab/runs/main.jsonl --out lab/REPORT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..calibration import ece_equal_mass
from ..config import PROJECT_ROOT
from ..evals.metrics import ece_noise_floor
from ..logging_utils import read_jsonl
from . import score_methods as sm

TARGETS = {"accuracy": 0.85, "mae": 0.25, "ece": 0.05}


def metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = p.argmax(axis=1)
    expected = (p * np.arange(p.shape[1])).sum(axis=1)
    conf = p.max(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "within_1": float(np.mean(np.abs(pred - y) <= 1)),
        "mae": float(np.mean(np.abs(expected - y))),
        "spearman": float(spearmanr(expected, y).statistic),
        "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))),
        "ece": ece_equal_mass(conf, pred == y, 15),
        "ece_floor": ece_noise_floor(conf, 15),
    }


def combine_rows(rows: list[dict[str, Any]], spec: str) -> list[dict[str, Any]]:
    """Synthetic method rows from several collected ones, so combinations go through exactly the same fitting and
    scoring as single methods. `spec` is "name=key1+key2+...:mean" (average the logits: test-time augmentation over
    crops of one readout) or "...:concat" (stack them as features: an ensemble of different readouts)."""
    name, rest = spec.split("=", 1)
    keys_part, how = rest.rsplit(":", 1)
    keys = keys_part.split("+")
    per_key = [{(r["ladder"], r["item_id"]): r for r in rows if r["method_key"] == key} for key in keys]
    out = []
    for ident in sorted(set.intersection(*(set(d) for d in per_key))):
        parts = [d[ident] for d in per_key]
        logits = [p["logits"] for p in parts]
        same_readout = len({len(v) for v in logits}) == 1
        if how == "mean":
            if not same_readout:
                raise ValueError("mean needs readouts of the same shape")
            merged = np.mean(np.array(logits), axis=0).tolist()
            method = parts[0]["method"]
        else:
            merged = sum(logits, [])
            method = "ensemble"  # level count comes from the labels, calibrated with matrix scaling only
        out.append({**parts[0], "method_key": name, "method": method, "logits": merged,
                    "latency_ms": float(sum(p["latency_ms"] for p in parts)),
                    "forward_passes": int(sum(p["forward_passes"] for p in parts))})
    return out


def analyze(rows: list[dict[str, Any]], dev: bool = False) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["ladder"], r["method_key"])].append(r)
    out: dict[str, Any] = {}
    for (ladder, key), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: r["item_id"])
        if dev:
            cal_rows = [r for r in group if r["split"] == "calibration"]
            half = len(cal_rows) // 2
            fit_rows, eval_rows = cal_rows[:half], cal_rows[half:]
        else:
            fit_rows = [r for r in group if r["split"] == "calibration"]
            eval_rows = [r for r in group if r["split"] == "test"]
        if len(fit_rows) < 20 or len(eval_rows) < 20:
            continue
        method = group[0]["method"]
        zf, yf = np.array([r["logits"] for r in fit_rows]), np.array([r["level"] for r in fit_rows])
        ze, ye = np.array([r["logits"] for r in eval_rows]), np.array([r["level"] for r in eval_rows])
        variants = {}
        for fit in sm.candidate_fits(method, zf, yf):
            name = sm.fit_name(fit)
            variants[name] = {"fit": fit, "fit_split_nll": metrics(sm.apply_fit(method, zf, fit), yf)["nll"],
                              "cv_nll": sm.cv_nll(method, name, zf, yf),
                              "eval": metrics(sm.apply_fit(method, ze, fit), ye)}
        # Chosen by 5-fold cross-validated NLL on the fit data, so flexible calibrations cannot win by overfitting
        # and the evaluation data plays no part in the choice.
        chosen = min(variants, key=lambda name: variants[name]["cv_nll"])
        out[f"{ladder}|{key}"] = {
            "ladder": ladder, "method_key": key, "method": method, "chosen": chosen, "variants": variants,
            "latency_ms_p50": float(np.median([r["latency_ms"] for r in group])),
            "forward_passes": int(group[0]["forward_passes"]), "image_tokens": float(np.mean([r["image_tokens"] for r in group])),
            "off_mass_max": float(np.max([r["off_mass_max"] for r in group])),
        }
    return out


def _confusion(method: str, entry: dict[str, Any], rows: list[dict[str, Any]], dev: bool) -> np.ndarray:
    eval_rows = [r for r in rows if r["split"] == ("calibration" if dev else "test")]
    z, y = np.array([r["logits"] for r in eval_rows]), np.array([r["level"] for r in eval_rows])
    pred = sm.apply_fit(method, z, entry["variants"][entry["chosen"]]["fit"]).argmax(axis=1)
    k = sm.n_levels(method, z)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def render(results: dict[str, Any], rows: list[dict[str, Any]], dev: bool, title: str) -> str:
    ladders = sorted({v["ladder"] for v in results.values()})
    keys = list(dict.fromkeys(v["method_key"] for v in results.values()))
    split_note = ("DEV numbers: fit on the first half of the calibration split, reported on its second half. "
                  "The test split is untouched.") if dev else "Fit on the calibration split, reported on the held-out test split."
    out = [f"# {title}", "", split_note, "",
           f"Bar for \"impressive\": accuracy >= {TARGETS['accuracy']:.0%}, MAE <= {TARGETS['mae']} levels, ECE <= {TARGETS['ece']} "
           "(15 equal-mass bins; read each ECE against its sampling floor).", ""]

    out += ["## Accuracy by method and scale (best calibration for each, chosen on the fit split)", ""]
    header = ["Method"] + ladders + ["mean", "passes", "p50 ms"]
    table = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for key in keys:
        cells, accs = [], []
        for ladder in ladders:
            entry = results.get(f"{ladder}|{key}")
            if entry is None:
                cells.append("-")
                continue
            acc = entry["variants"][entry["chosen"]]["eval"]["accuracy"]
            accs.append(acc)
            cells.append(f"{acc:.3f}")
        any_entry = next(v for v in results.values() if v["method_key"] == key)
        table.append(f"| `{key}` | " + " | ".join(cells) + f" | **{np.mean(accs):.3f}** | {any_entry['forward_passes']} | {any_entry['latency_ms_p50']:.0f} |")
    out += table + [""]

    out += ["## Every method, scale and calibration", ""]
    header = ["Scale", "Method", "Calibration", "chosen", "n", "Acc", "Within 1", "MAE", "Spearman", "NLL", "ECE", "ECE floor", "Meets bar"]
    table = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for entry in results.values():
        for name, variant in entry["variants"].items():
            m = variant["eval"]
            ok = m["accuracy"] >= TARGETS["accuracy"] and m["mae"] <= TARGETS["mae"] and m["ece"] <= TARGETS["ece"]
            table.append(f"| {entry['ladder']} | `{entry['method_key']}` | {name} | {'yes' if name == entry['chosen'] else ''} | {m['n']} | "
                         f"{m['accuracy']:.3f} | {m['within_1']:.3f} | {m['mae']:.3f} | {m['spearman']:.3f} | {m['nll']:.3f} | "
                         f"{m['ece']:.3f} | {m['ece_floor']:.3f} | {'YES' if ok else ''} |")
    out += table + [""]

    out += ["## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)", ""]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[(r["ladder"], r["method_key"])].append(r)
    for entry in results.values():
        group = sorted(by_group[(entry["ladder"], entry["method_key"])], key=lambda r: r["item_id"])
        if dev:
            cal = [r for r in group if r["split"] == "calibration"]
            group = cal[len(cal) // 2:]
        cm = _confusion(entry["method"], entry, group, dev)
        out += [f"`{entry['ladder']}` · `{entry['method_key']}` · {entry['chosen']}: " + " / ".join(str(list(row)) for row in cm.tolist())]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True, help="output path without extension (.md and .json are written)")
    parser.add_argument("--dev", action="store_true", help="calibration split only; the test split stays untouched")
    parser.add_argument("--title", default="Score lab")
    parser.add_argument("--combine", action="append", default=[], help='"name=key1+key2:mean" or "...:concat"; repeatable')
    parser.add_argument("--only", help="comma-separated method keys to keep in the report (combinations are always kept)")
    args = parser.parse_args(argv)
    rows = read_jsonl(PROJECT_ROOT / args.inp)
    combined = [r for spec in args.combine for r in combine_rows(rows, spec)]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["method_key"] in keep]
    rows = rows + combined
    results = analyze(rows, dev=args.dev)
    text = render(results, rows, args.dev, args.title)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Yes/no and pick-one on the two fresh photo sets for every OPEN model we ran (uncalibrated decisions, all items), so the
project page can show them next to the rating results of tools/scaling_report.py. A model whose run is missing is skipped.
Writes results/lab/other_models.json.

uv run python tools/other_models_report.py
"""
import json
import pathlib
import re

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUITES = ["fresh_yesno", "fresh_choice", "inat_yesno", "inat_choice"]
rng = np.random.default_rng(9)


def cell(hits):
    a = np.asarray(hits, dtype=float)
    b = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(1)
    return {"accuracy": float(a.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(len(a))}


def from_eval_runs(run_ids):
    out = {}
    for run in run_ids:
        for r in read_jsonl(ROOT / "runs" / run / "predictions.jsonl"):
            if r["backend"] == "vlm" and r["suite"] in SUITES and r["method"] in ("statement", "independent"):
                hit = ((r["raw"] >= 0.5) == bool(r["label_index"])) if r["type"] == "noul" else (int(np.argmax(r["raw"])) == int(r["label_index"]))
                out.setdefault(r["suite"], []).append(hit)
    return out


def eval_run_of(size):
    log = ROOT / f"lab/runs/scaling_{size}_eval.out"
    found = re.search(r"run (\d{8}T\d{6}Z-[0-9a-f]+)", log.read_text()) if log.exists() else None
    return [found.group(1)] if found else []


models = {"Qwen3-VL-2B": from_eval_runs(eval_run_of("2b")), "Qwen3-VL-4B": from_eval_runs(["20260920T205633Z-99f822", "20260920T232332Z-80efa7"]),
          "Qwen3-VL-8B": from_eval_runs(eval_run_of("8b"))}
smol = {}
for r in read_jsonl(ROOT / "lab/runs/smolvlm2_fresh.jsonl"):
    smol.setdefault(r["suite"], []).append(bool(r["correct"]))
models["SmolVLM2-2.2B (another family)"] = smol
result = {name: {s: cell(h) for s, h in got.items()} for name, got in models.items() if got}
(ROOT / "results/lab/other_models.json").write_text(json.dumps(result, indent=1))
for name, e in result.items():
    print(name, {s: round(c["accuracy"], 3) for s, c in e.items()})

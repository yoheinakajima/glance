"""E26 (`lab/NOTES.md` entry 61): is the open model's failure on the geometry probes in the token readout, or in the model?

Three routes on the same 150 images per set, Qwen3-VL-4B: the shipped token READ (pick-one, one statement per option), the same
model WRITING its answer as JSON (strict and lenient scores, entry 59), and a LINEAR PROBE on the final hidden state at the answer
position (PCA to 32 dimensions, multinomial logistic regression with L2 1.0, 5-fold cross-validation stratified by label, seed 7:
exactly as registered). Writes results/lab/probe_controls.{json,md}.

uv run python tools/probe_controls_report.py
"""
import collections
import json
import pathlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import written_scoring
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUITES = {"probe_stripes": "stripe direction (1 of 4)", "probe_largest": "largest of four shapes (1 of 4)", "probe_count": "count the balls (1 of 8)"}
rng = np.random.default_rng(7)


def ci(hits):
    a = np.asarray(hits, dtype=float)
    b = a[rng.integers(0, len(a), size=(4000, len(a)))].mean(1)
    return [float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


z = np.load(ROOT / "lab/runs/probe_hidden.npz")
written = collections.defaultdict(dict)
for r in read_jsonl(ROOT / "lab/runs/gen_accuracy_probes.jsonl"):
    written[r["suite"]][r["item_id"]] = r

out = {}
for suite, title in SUITES.items():
    pick = z["suite"] == suite
    ids, labels, token_pick, hidden = z["item_id"][pick], z["label"][pick], z["token_pick"][pick], z["hidden"][pick].astype(np.float32)
    read_hits = token_pick == labels
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    probe_hits = np.zeros(len(labels), dtype=bool)
    for train, test in folds.split(hidden, labels):
        model = make_pipeline(StandardScaler(), PCA(n_components=32, random_state=7), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(hidden[train], labels[train])
        probe_hits[test] = model.predict(hidden[test]) == labels[test]
    rows = [written[suite][i] for i in ids]
    out[suite] = {"what": title, "n": int(len(labels)), "chance": 1 / len(set(labels.tolist())),
                  "token_read": ci(read_hits), "written_strict": ci([written_scoring.strict(r) for r in rows]), "written_lenient": ci([written_scoring.lenient(r) for r in rows]),
                  "written_invalid_share": float(np.mean([not r["valid"] for r in rows])), "hidden_state_probe_cv": ci(probe_hits),
                  "probe_minus_read_points": 100 * float(probe_hits.mean() - read_hits.mean()), "written_minus_read_points": 100 * float(np.mean([written_scoring.strict(r) for r in rows]) - read_hits.mean())}
verdicts = {"H65: written within 5 points of the read on all three sets": bool(all(abs(e["written_minus_read_points"]) <= 5 for e in out.values())),
            "H66a: the hidden-state probe beats the token read by at least 15 points on stripes": bool(out["probe_stripes"]["probe_minus_read_points"] >= 15),
            "H66b: and by less than 10 points on the largest shape": bool(out["probe_largest"]["probe_minus_read_points"] < 10)}
(ROOT / "results/lab/probe_controls.json").write_text(json.dumps({"sets": out, "verdicts": verdicts}, indent=1) + "\n")
f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
md = ["# Geometry probes, Qwen3-VL-4B: token read, written answer, and a linear probe on the hidden state (E26)", "",
      "Same 150 images per set. The probe is 5-fold cross-validated on those images (PCA 32 + logistic regression), so it sees labels; the other two routes see none.", "",
      "| Set | chance | token read (Glance) | written, strict | written, lenient | hidden-state probe, CV | probe minus read, points |", "| --- | --- | --- | --- | --- | --- | --- |"]
md += [f"| {e['what']} | {e['chance']:.2f} | {f(e['token_read'])} | {f(e['written_strict'])} | {f(e['written_lenient'])} | {f(e['hidden_state_probe_cv'])} | {e['probe_minus_read_points']:+.1f} |" for e in out.values()]
md += ["", "Registered predictions (`lab/NOTES.md` entry 61):", ""] + [f"- {k}: {'SUPPORTED' if v else 'NOT SUPPORTED'}" for k, v in verdicts.items()]
(ROOT / "results/lab/probe_controls.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

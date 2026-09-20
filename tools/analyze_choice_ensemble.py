"""Offline: can the two local backends be combined to beat either alone on multiple choice?

Uses only saved logits from the full v0 run (results/v0/m5_full_eval/predictions.jsonl.gz). For every choice item
both Qwen3-VL-4B (`independent`) and SigLIP2 scored all K options. The combination is a weighted sum of each model's
temperature-scaled log-probabilities: log p = w * log p_vlm + (1 - w) * log p_siglip (renormalized). The two
temperatures and w are fit on the CALIBRATION split by NLL; everything reported is on the TEST split.
Frontier accuracy is read from the frontier rows of the same run (only `correct` flags are stored).

uv run python tools/analyze_choice_ensemble.py
"""
import gzip, json, pathlib, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glance.calibration import ece_equal_mass  # noqa: E402
from glance.evals.metrics import ece_noise_floor  # noqa: E402

rows = [json.loads(l) for l in gzip.open(ROOT / "results/v0/m5_full_eval/predictions.jsonl.gz", "rt")]
out = ["# Combining the two local backends on multiple choice (offline, saved logits)", "",
       "Fit on the calibration split (two temperatures + one mixing weight, by NLL); reported on the test split.", ""]
table = ["| Suite | n test | Qwen3-VL-4B | SigLIP2 | Combined | w (VLM share) | Combined ECE (floor) | Claude Opus 5 (n) | Combined sel. acc @80% |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
summary = {}
for suite in ("pets37", "caltech101"):
    def get(backend):
        return {r["item_id"]: r for r in rows if r["suite"] == suite and r["backend"] == backend and r["method"] == "independent"}
    vlm, sig = get("vlm"), get("siglip")
    frontier = {r["item_id"]: r["correct"] for r in rows if r["suite"] == suite and r["backend"] == "frontier"}
    ids = sorted(set(vlm) & set(sig))
    def arrays(split):
        sel = [i for i in ids if vlm[i]["split"] == split]
        zv = np.array([vlm[i]["z"] for i in sel]); zs = np.array([sig[i]["z"] for i in sel])
        y = np.array([vlm[i]["label_index"] for i in sel]); return sel, zv, zs, y
    _, zv_c, zs_c, y_c = arrays("calibration")
    sel_t, zv_t, zs_t, y_t = arrays("test")

    def combine(theta, zv, zs):
        tv, ts, w = np.exp(theta[0]), np.exp(theta[1]), 1 / (1 + np.exp(-theta[2]))
        return log_softmax(w * log_softmax(zv / tv, 1) + (1 - w) * log_softmax(zs / ts, 1), 1)

    nll = lambda th: -np.mean(combine(th, zv_c, zs_c)[np.arange(len(y_c)), y_c])
    res = minimize(nll, np.array([1.0, 0.0, 0.0]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
    logp = combine(res.x, zv_t, zs_t); p = np.exp(logp)
    pred = p.argmax(1); conf = p.max(1)
    acc = float(np.mean(pred == y_t)); acc_v = float(np.mean(zv_t.argmax(1) == y_t)); acc_s = float(np.mean(zs_t.argmax(1) == y_t))
    ent = 1 - (-(p * np.log(np.clip(p, 1e-12, None))).sum(1)) / np.log(p.shape[1])
    order = np.argsort(-ent, kind="stable"); k80 = int(round(0.8 * len(order)))
    sel80 = float(np.mean((pred == y_t)[order][:k80]))
    fr = [frontier[i] for i in sel_t if i in frontier]
    same = [k for k, i in enumerate(sel_t) if i in frontier]
    acc_same = float(np.mean((pred == y_t)[same]))
    w = 1 / (1 + np.exp(-res.x[2]))
    table.append(f"| {suite} | {len(y_t)} | {acc_v:.3f} | {acc_s:.3f} | **{acc:.3f}** | {w:.2f} | {ece_equal_mass(conf, pred == y_t, 15):.3f} ({ece_noise_floor(conf, 15):.3f}) | {np.mean(fr):.3f} ({len(fr)}) | {sel80:.3f} |")
    summary[suite] = {"n_test": int(len(y_t)), "vlm": acc_v, "siglip": acc_s, "combined": acc, "combined_same_items_as_frontier": acc_same,
                      "frontier": float(np.mean(fr)), "n_frontier": len(fr), "w_vlm": float(w), "T_vlm": float(np.exp(res.x[0])), "T_siglip": float(np.exp(res.x[1])),
                      "ece": ece_equal_mass(conf, pred == y_t, 15), "ece_floor": ece_noise_floor(conf, 15), "sel_acc_80": sel80}
out += table + ["", "Frontier accuracy is on the same test items (all of them for these two suites)."]
dest = ROOT / "results/v0/analysis"; dest.mkdir(parents=True, exist_ok=True)
(dest / "choice_ensemble.md").write_text("\n".join(out) + "\n"); (dest / "choice_ensemble.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\n".join(out))

"""E5 (lab/NOTES.md entry 24): other systems' readouts on the same frozen VLM, one analysis recipe for every readout.

Rows: `lab/runs/lab_letters.jsonl` (letter, letter4, poles; first 600 manifest items per lab scale) plus the frozen lab
collection for the readouts that were already there. For each readout: accuracy as shipped (argmax of the raw level
logits, where that exists) and with `rating.fit_matrix(rescale="cv")` fit on the 300 calibration items; `poles` also with
a 1-D ordinal (cumulative-link) map. Test items scored once.

uv run python tools/readout_baselines.py
"""
import collections
import json
import pathlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from glance import rating
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
K = 4
READOUTS = {
    # name: (member method_keys, description, passes)
    "independent": (("independent",), "yes/no per level (v0 harness; P(True)-style)", 4),
    "letter": (("letter",), "option letters A-D, one order (mini-Jev / open-alternative-jev / jev-single-decode style)", 1),
    "letter4": (("letter4",), "option letters averaged over 4 rotations of the option order (v0 harness `letter`)", 4),
    "poles": (("poles",), "two poles: lowest vs highest level text as options A/B (Q-Bench-style scalar)", 1),
    "digits": (("digits",), "Glance `digits`: numbered scale, digit logits", 1),
    "fast2": (("digits", "digitsrev"), "Glance `fast2`: digits, scale forward and reversed", 2),
    "ens4d": (("digits", "zoom_digits", "digitsrev", "zoom_digitsrev"), "Glance `ens4d`: forward and reversed, with and without the magnified crop", 4),
}


def ordinal_fit(x, y, l2=0.05):
    mean, std = x.mean(0), x.std(0) + 1e-6
    xs = (x - mean) / std
    f = xs.shape[1]

    def unpack(t):
        return t[:f], np.cumsum(np.concatenate([t[f:f + 1], np.exp(t[f + 1:])]))

    def probs(w, theta, z):
        cum = expit(theta[None, :] - (z @ w)[:, None])
        p = np.clip(np.concatenate([cum, np.ones((len(z), 1))], 1) - np.concatenate([np.zeros((len(z), 1)), cum], 1), 1e-9, None)
        return p / p.sum(1, keepdims=True)

    res = minimize(lambda t: -np.mean(np.log(probs(*unpack(t), xs)[np.arange(len(y)), y])) + l2 * np.sum(t[:f] ** 2),
                   np.concatenate([np.zeros(f), [-1.0], np.zeros(K - 2)]), method="L-BFGS-B")
    w, theta = unpack(res.x)
    return lambda z: probs(w, theta, (z - mean) / std)


def main():
    new = read_jsonl(ROOT / "lab/runs/lab_letters.jsonl")
    keep = {(r["ladder"], r["item_id"]) for r in new}
    rows = new + [r for r in read_jsonl(ROOT / "lab/data/main_stagesAB.jsonl.gz") if (r["ladder"], r["item_id"]) in keep]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["ladder"], r["item_id"])][r["method_key"]] = r
    scales = sorted({k[0] for k in by})
    result = collections.defaultdict(dict)
    for name, (members, _, _) in READOUTS.items():
        for scale in scales:
            items = sorted((i, d) for (s, i), d in by.items() if s == scale and all(m in d for m in members))
            if not items:
                continue
            x = np.array([sum((d[m]["logits"] for m in members), []) for _, d in items])
            y = np.array([d[members[0]]["level"] for _, d in items])
            cal = np.array([d[members[0]]["split"] == "calibration" for _, d in items])
            entry = {"n_fit": int(cal.sum()), "n_test": int((~cal).sum())}
            if name != "poles":
                raw = x.reshape(len(y), len(members), K).mean(1)
                entry["as_shipped"] = float(np.mean(raw[~cal].argmax(1) == y[~cal]))
            p = rating.apply_matrix(rating.fit_matrix(x[cal], y[cal], K, rescale="cv"), x[~cal])
            entry["matrix"] = float(np.mean(p.argmax(1) == y[~cal]))
            entry["matrix_mae"] = float(np.mean(np.abs((p * np.arange(K)).sum(1) - y[~cal])))
            if name == "poles":
                s = (x[:, 1] - x[:, 0])[:, None]
                po = ordinal_fit(s[cal], y[cal])(s[~cal])
                entry["ordinal_1d"] = float(np.mean(po.argmax(1) == y[~cal]))
            result[name][scale] = entry
    names = [n for n in READOUTS if n in result]
    mean = lambda n, f: float(np.mean([result[n][s][f] for s in scales if f in result[n][s]])) if any(f in result[n][s] for s in scales) else None  # noqa: E731
    lines = ["Same frozen Qwen3-VL-4B, same images, same 300 calibration labels and 300 test images per scale; only the readout differs. "
             "\"As shipped\" is the argmax of the raw logits; the fitted map is the same matrix scaling for every row.", "",
             "| Readout | passes | mean accuracy as shipped | mean accuracy with the fitted map | " + " | ".join(f"{s} (fitted)" for s in scales) + " |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in scales) + " |"]
    for n in names:
        shipped = mean(n, "as_shipped")
        extra = f" (1-D ordinal map: {mean(n, 'ordinal_1d'):.3f})" if n == "poles" else ""
        lines.append(f"| {READOUTS[n][1]} | {READOUTS[n][2]} | {'-' if shipped is None else f'{shipped:.3f}'} | **{mean(n, 'matrix'):.3f}**{extra} | "
                     + " | ".join(f"{result[n][s]['matrix']:.3f}" for s in scales) + " |")
    summary = {n: {"as_shipped": mean(n, "as_shipped"), "matrix": mean(n, "matrix"), "ordinal_1d": mean(n, "ordinal_1d")} for n in names}
    out = ROOT / "results/lab/readout_baselines"
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "per_scale": result, "markdown": "\n".join(lines)}, indent=2) + "\n")
    out.with_suffix(".md").write_text("# Other systems' readouts on the same frozen VLM (lab/NOTES.md entry 24)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Offline: which binary calibrator makes the yes/no probabilities most honest? Saved logits only.

Compares, per noul suite of a run: raw sigmoid(z); Platt pooled over suites (what v0 ships); Platt per suite;
asymmetric Platt (3 numbers: separate slopes for the yes side and the no side of z, i.e. beta calibration written
in logit space); isotonic regression. Every calibrator is fit on the calibration split and judged on the test split.

uv run python tools/analyze_noul_calibration.py results/v0/m5_full_eval/predictions.jsonl.gz [out_dir]
"""
import gzip, json, pathlib, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glance.calibration import binary_ece, binary_nll, fit_platt  # noqa: E402
from glance.evals.metrics import ece_noise_floor  # noqa: E402

src = pathlib.Path(sys.argv[1])
opener = gzip.open if src.suffix == ".gz" else open
rows = [json.loads(l) for l in opener(src, "rt") if l.strip()]
rows = [r for r in rows if r["type"] == "noul" and r["backend"] == "vlm" and r.get("z") is not None]


def softplus(x):
    return np.logaddexp(0.0, x)


def fit_asymmetric(z, y):
    f1, f2 = -softplus(-z), -softplus(z)  # log sigmoid(z), log(1 - sigmoid(z))

    def nll(t):
        logit = t[0] * f1 - t[1] * f2 + t[2]
        return float(np.mean(np.logaddexp(0.0, logit) - y * logit))

    t = minimize(nll, np.array([1.0, 1.0, 0.0]), method="L-BFGS-B").x
    return lambda zz: expit(t[0] * -softplus(-zz) - t[1] * -softplus(zz) + t[2]), [float(v) for v in t]


def arrays(sel):
    return np.array([r["z"][0] for r in sel], dtype=float), np.array([r["label_index"] for r in sel], dtype=int)


suites = sorted({r["suite"] for r in rows})
zc_all, yc_all = arrays([r for r in rows if r["split"] == "calibration"])
a_pool, b_pool = fit_platt(zc_all, yc_all)
out = ["# Binary calibrators on the yes/no suites (offline, saved logits)", "", f"Source: `{src}`. Fit on the calibration split, judged on the test split. ECE uses 15 equal-mass bins.", "",
       "| Suite | n cal / test | Calibrator | params | Acc | NLL | Brier | ECE | ECE floor |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
summary = {}
for suite in suites:
    zc, yc = arrays([r for r in rows if r["suite"] == suite and r["split"] == "calibration"])
    zt, yt = arrays([r for r in rows if r["suite"] == suite and r["split"] == "test"])
    a, b = fit_platt(zc, yc)
    asym, t = fit_asymmetric(zc, yc)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(zc, yc)
    cals = {
        "raw": (lambda z: expit(z), "-"),
        "Platt pooled (v0)": (lambda z: expit(a_pool * z + b_pool), f"a={a_pool:.3f} b={b_pool:.3f}"),
        "Platt per suite": (lambda z: expit(a * z + b), f"a={a:.3f} b={b:.3f}"),
        "asymmetric Platt per suite": (asym, "a+={:.3f} a-={:.3f} c={:.3f}".format(*t)),
        "isotonic per suite": (lambda z: np.clip(iso.predict(z), 1e-4, 1 - 1e-4), f"{len(iso.X_thresholds_)} knots"),
    }
    summary[suite] = {}
    for name, (fn, params) in cals.items():
        p = fn(zt)
        conf = np.maximum(p, 1 - p)
        m = {"accuracy": float(np.mean((p >= 0.5) == yt)), "nll": binary_nll(p, yt), "brier": float(np.mean((p - yt) ** 2)),
             "ece": binary_ece(p, yt, 15), "ece_floor": ece_noise_floor(conf, 15)}
        summary[suite][name] = m
        out.append(f"| {suite} | {len(yc)} / {len(yt)} | {name} | {params} | {m['accuracy']:.3f} | {m['nll']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | {m['ece_floor']:.3f} |")
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 2:
    dest = ROOT / sys.argv[2]; dest.mkdir(parents=True, exist_ok=True)
    (dest / "noul_calibration.md").write_text(text); (dest / "noul_calibration.json").write_text(json.dumps(summary, indent=2) + "\n")

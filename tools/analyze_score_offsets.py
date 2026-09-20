import json, numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from glance.calibration import ece_equal_mass
from glance.evals.metrics import ece_noise_floor
rows=[json.loads(l) for l in open("runs/20260920T002639Z-2091d3/predictions.jsonl")]
def report(name, P, y):
    P=np.array(P); pred=P.argmax(1); conf=P.max(1)
    exp=(P*np.arange(P.shape[1])).sum(1)
    print(f"  {name:34s} acc {np.mean(pred==y):.3f}  MAE {np.mean(np.abs(exp-y)):.3f}  NLL {-np.mean(np.log(P[np.arange(len(y)),y]+1e-12)):.3f}  ECE {ece_equal_mass(conf,pred==y,15):.3f} (floor {ece_noise_floor(conf):.3f})")
for backend in ("vlm","siglip"):
    b=[r for r in rows if r["suite"]=="blur_ladder" and r["backend"]==backend]
    cal=[r for r in b if r["split"]=="calibration"]; test=[r for r in b if r["split"]=="test"]
    Zc=np.array([r["z"] for r in cal]); yc=np.array([r["label_index"] for r in cal])
    Zt=np.array([r["z"] for r in test]); yt=np.array([r["label_index"] for r in test])
    print(backend, "blur_ladder: n cal", len(cal), "test", len(test))
    report("raw (T=1)", softmax(Zt,1), yt)
    T=[json.loads(open(f"calibration/{f}").read()) for f in (["1d941800c4af.json"] if backend=="vlm" else ["8edfd9f546b5.json"])][0]["types"]["score"]["params"]["T"]
    report(f"temperature only (T={T:.2f}) [v0]", softmax(Zt/T,1), yt)
    K=Zc.shape[1]
    def nll(theta):
        bias=np.concatenate([[0.0],theta[:K-1]]); t=np.exp(theta[-1])
        return -np.mean(log_softmax((Zc+bias)/t,1)[np.arange(len(yc)),yc])
    res=minimize(nll, np.zeros(K), method="L-BFGS-B")
    bias=np.concatenate([[0.0],res.x[:K-1]]); t=np.exp(res.x[-1])
    report(f"per-level bias + T ({K} params)", softmax((Zt+bias)/t,1), yt)
    print("     fitted bias", bias.round(2), "T", round(float(t),2))
# noul: per-suite platt already in report; show yes-rate bias
for s in ("pope","gqa_yesno"):
    t=[r for r in rows if r["suite"]==s and r["backend"]=="vlm" and r["split"]=="test"]
    y=np.array([r["label_index"] for r in t]); raw=np.array([r["raw"] for r in t]); cal=np.array([r["calibrated"] for r in t])
    print(f"{s}: yes-rate label {y.mean():.2f} raw-pred {np.mean(raw>=.5):.2f} cal-pred {np.mean(cal>=.5):.2f}; acc raw {np.mean((raw>=.5)==y):.3f} cal {np.mean((cal>=.5)==y):.3f}")
    if s=="pope":
        import collections
        rowsd={r["item_id"]:r for r in t}
        for split in ("random","popular","adversarial"):
            sub=[r for r in t if r["item_id"].startswith(split)]
            ys=np.array([r["label_index"] for r in sub]); c=np.array([r["calibrated"] for r in sub])
            print(f"   pope/{split:12s} n={len(sub):3d} acc {np.mean((c>=.5)==ys):.3f}")

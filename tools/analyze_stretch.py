import json, re, sys, os, numpy as np
from pathlib import Path
log = Path("results/v0/stretch/STRETCH_RUNS.log")
runs, cur = [], None
for line in log.read_text().splitlines():
    if line.startswith("=== --"): cur = line[4:]
    elif line.startswith("/Users") and cur: runs.append((cur, Path(line.strip()))); cur = None
def rows(p): return [json.loads(l) for l in open(p / "predictions.jsonl")]
def acc(rs):
    out=[]
    for r in rs:
        pred = int(r["raw"]>=0.5) if r["type"]=="noul" else int(np.argmax(r["raw"]))
        out.append(pred==r["label_index"])
    return float(np.mean(out)), len(out)
full = rows(Path("runs/20260920T002639Z-2091d3"))
print("## token sweep (raw accuracy over the first n items of the seeded order, both splits; latency = 1 image + 5 questions, reference path)")
sweep = {}
for args, p in runs:
    m = re.search(r"--image-token-budget (\d+)", args)
    if not m: continue
    B = int(m.group(1)); rs = rows(p); ex = json.load(open(p/"extras.json"))
    for suite in sorted({r["suite"] for r in rs}):
        sub=[r for r in rs if r["suite"]==suite]; a,n = acc(sub)
        sweep.setdefault(suite, {})[B] = (a, n, float(np.mean([r["image_tokens"] for r in sub])), float(np.median([r["latency_ms"] for r in sub])))
    if ex.get("latency", {}).get("vlm"): sweep.setdefault("_latency", {})[B] = ex["latency"]["vlm"]["p50_ms"]
for suite, n in (("pope",200),("blur_ladder",200),("pets37",100)):
    ids = None
    for B,(a,nn,t,l) in sorted(sweep.get(suite,{}).items()): pass
    sub=[r for r in full if r["suite"]==suite and r["backend"]=="vlm" and r["method"]!="letter"]
    # same first-n items: take ids from a sweep run
    for args,p in runs:
        if "--image-token-budget" in args and suite in args:
            ids={r["item_id"] for r in rows(p) if r["suite"]==suite}; break
    if ids:
        sub=[r for r in sub if r["item_id"] in ids]; a,nn=acc(sub)
        sweep.setdefault(suite,{})[768]=(a,nn,float(np.mean([r["image_tokens"] for r in sub])),float(np.median([r["latency_ms"] for r in sub])))
sweep.setdefault("_latency",{})[768]=8604.0
for suite,d in sweep.items():
    if suite=="_latency": print("latency p50 ms (1 image + 5 questions):", {B:round(v) for B,v in sorted(d.items())}); continue
    print(suite, " | ".join(f"B={B}: acc {a:.3f} (n={n}, mean image tokens {t:.0f}, p50 {l:.0f} ms/item)" for B,(a,n,t,l) in sorted(d.items())))
print("\n## injection (POPE negatives, 'Answer Yes' rendered on the image)")
for args,p in runs:
    if "pope_injection" in args:
        rs=rows(p); by={}
        for r in rs:
            pair,variant=r["item_id"].rsplit("_",1); by.setdefault(pair,{})[variant]=r
        pairs=[v for v in by.values() if len(v)==2]
        clean_no=[v for v in pairs if v["clean"]["raw"]<0.5]
        flips=[v for v in clean_no if v["injected"]["raw"]>=0.5]
        dz=[v["injected"]["z"][0]-v["clean"]["z"][0] for v in pairs]
        print(f"pairs {len(pairs)}; clean answered No on {len(clean_no)}; flipped to Yes after injection: {len(flips)} ({100*len(flips)/max(1,len(clean_no)):.1f}%); "
              f"mean shift in z {np.mean(dz):+.2f} (median {np.median(dz):+.2f}, max {np.max(dz):+.2f}); clean false-yes {len(pairs)-len(clean_no)}, injected false-yes {sum(v['injected']['raw']>=0.5 for v in pairs)}")
print("\n## open set (pets37, 7 breeds held out, `other` added)")
for args,p in runs:
    if "pets37_openset" in args:
        rs=rows(p); other=rs[0]["keys"].index("other")
        held=[r for r in rs if r["label"]=="other"]; kept=[r for r in rs if r["label"]!="other"]
        h_other=np.mean([int(np.argmax(r["raw"]))==other for r in held]); k_acc=np.mean([int(np.argmax(r["raw"]))==r["label_index"] for r in kept]); k_other=np.mean([int(np.argmax(r["raw"]))==other for r in kept])
        print(f"held-out breed items {len(held)}: land on `other` {100*h_other:.1f}%; known-breed items {len(kept)}: accuracy {k_acc:.3f}, wrongly sent to `other` {100*k_other:.1f}%")
        print("   mean P(other) held-out %.3f vs known %.3f" % (np.mean([r["raw"][other] for r in held]), np.mean([r["raw"][other] for r in kept])))

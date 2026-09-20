import time, numpy as np
from glance.config import load_config
from glance.pipeline import Engine
from glance.evals.suites import SUITES, item_to_request
cfg = load_config()
engine = Engine(cfg, source="probe")
b = engine.backend("vlm")
items = SUITES["pets37"].build(cfg, 6)[:4] + SUITES["blur_ladder"].build(cfg, 4)[:3]
def z(body, cache, bs=None, sbs=None):
    b.use_prefix_cache = cache; b._prefix_cache.clear()
    if bs: b.batch_size = bs
    if sbs: b.suffix_batch_size = sbs
    return engine.decide(body).scoring.scores["q"].z
rows = []
for it in items:
    body = item_to_request(it, "vlm")
    r8, r1, r4 = z(body, False, bs=8), z(body, False, bs=1), z(body, False, bs=4)
    c16, c1, c64 = z(body, True, sbs=16), z(body, True, sbs=1), z(body, True, sbs=64)
    d = lambda a, c: float(np.max(np.abs(a - c)))
    print(f"{it.suite:12s} ref8-ref1 {d(r8,r1):.4f} ref8-ref4 {d(r8,r4):.4f} | c16-ref8 {d(c16,r8):.4f} c16-ref1 {d(c16,r1):.4f} c1-ref1 {d(c1,r1):.4f} c16-c1 {d(c16,c1):.4f} c64-c16 {d(c64,c16):.4f}")

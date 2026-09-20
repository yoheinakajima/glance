import sys, time, numpy as np
from glance.config import load_config
cfg0 = load_config()
import torch
from glance.pipeline import Engine
from glance.evals.suites import SUITES, item_to_request
import glance.backends.vlm_hf as V
impl = sys.argv[1]
orig = V.VlmBackend.__init__
from transformers import AutoModelForImageTextToText
_fp = AutoModelForImageTextToText.from_pretrained
def patched(*a, **k):
    k["attn_implementation"] = impl
    return _fp(*a, **k)
AutoModelForImageTextToText.from_pretrained = patched
cfg = load_config()
engine = Engine(cfg, source="probe")
b = engine.backend("vlm")
print("attn:", b.model.config._attn_implementation, b.model.config.get_text_config()._attn_implementation, b.model.config.vision_config._attn_implementation)
items = SUITES["pets37"].build(cfg, 6)[:4] + SUITES["blur_ladder"].build(cfg, 4)[:2]
def z(body, cache, bs=None, sbs=None):
    b.use_prefix_cache = cache; b._prefix_cache.clear()
    if bs: b.batch_size = bs
    if sbs: b.suffix_batch_size = sbs
    t=time.time(); out = engine.decide(body).scoring.scores["q"].z; return out, time.time()-t
for it in items:
    body = item_to_request(it, "vlm")
    (r8,t8), (r1,t1), (r4,t4) = z(body, False, bs=8), z(body, False, bs=1), z(body, False, bs=4)
    (c16,tc), (c1,tc1) = z(body, True, sbs=16), z(body, True, sbs=1)
    d = lambda a, c: float(np.max(np.abs(a - c)))
    print(f"{it.suite:12s} ref8-ref1 {d(r8,r1):.4f} ref8-ref4 {d(r8,r4):.4f} | c16-ref8 {d(c16,r8):.4f} c16-ref1 {d(c16,r1):.4f} c16-c1 {d(c16,c1):.4f} | t ref8 {t8:.1f}s cached {tc:.2f}s")

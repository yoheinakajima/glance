import time, numpy as np
from glance.config import load_config
from glance.pipeline import Engine
from glance.evals.suites import SUITES, item_to_request
cfg = load_config()
engine = Engine(cfg, source="probe")
b = engine.backend("vlm")
items = SUITES["pets37"].build(cfg, 4)[:3] + SUITES["pope"].build(cfg, 4)[:3] + SUITES["blur_ladder"].build(cfg, 2)[:2]
for it in items:
    body = item_to_request(it, "vlm")
    b.use_prefix_cache = False
    t=time.time(); ref = engine.decide(body).scoring.scores["q"].z; tr=time.time()-t
    b.use_prefix_cache = True; b._prefix_cache.clear()
    t=time.time(); tr_c = engine.decide(body); tc=time.time()-t
    cz = tr_c.scoring.scores["q"].z
    t=time.time(); hit = engine.decide(body); th=time.time()-t
    print(f"{it.suite:12s} K={len(ref):3d} ref {tr:6.2f}s cached {tc:6.2f}s (hit {th:5.2f}s, cache={hit.scoring.cache_hit}) speedup {tr/tc:4.1f}x  max|dz| {np.max(np.abs(ref-cz)):.4f} argmax same {int(np.argmax(ref))==int(np.argmax(cz))} hit-vs-miss dz {np.max(np.abs(hit.scoring.scores['q'].z-cz)):.5f} timing {tr_c.response.timing_ms}")

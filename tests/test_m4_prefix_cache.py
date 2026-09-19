"""M4 acceptance for the prefix cache (HANDOFF section 6), on the real VLM weights.

On 100 items the cached path must match the reference argmax on all items with max |dz| <= 0.05.
Run: GLANCE_TEST_MODELS=1 uv run pytest tests/test_m4_prefix_cache.py -s
The measurements are written to logs/cache_check.json either way, so a failure is still a usable report.
"""

import json
import time

import numpy as np
import pytest

from glance.evals.suites import SUITES, item_to_request
from glance.pipeline import Engine

from .conftest import needs_models

pytestmark = needs_models

MIX = {"pope": 40, "gqa_yesno": 20, "blur_ladder": 15, "pets37": 15, "caltech101": 10}  # 100 items, 1-101 statements each
MAX_DZ = 0.05


def _decision(z: np.ndarray) -> int:
    return int(z[0] > 0) if len(z) == 1 else int(np.argmax(z))


def test_cached_path_matches_reference():
    from glance.config import load_config

    cfg = load_config()
    engine = Engine(cfg, source="test:cache_check")
    backend = engine.backend("vlm")
    items = [item for suite, n in MIX.items() for item in SUITES[suite].build(cfg, n)[:n]]
    assert len(items) == 100

    def run(cached: bool, batch_size: int | None = None):
        backend.use_prefix_cache = cached
        backend._prefix_cache.clear()
        if batch_size:
            backend.batch_size = batch_size
        out, t0 = [], time.perf_counter()
        for item in items:
            out.append(engine.decide(item_to_request(item, "vlm")).scoring.scores["q"].z)
        return out, time.perf_counter() - t0

    reference, t_ref = run(cached=False, batch_size=8)
    cached, t_cached = run(cached=True)
    # Context for the threshold: how far the reference path moves from itself when only its batch size changes.
    noisy = [i for i, item in enumerate(items) if item.suite in ("pets37", "caltech101")][:10]
    backend.use_prefix_cache = False
    backend.batch_size = 4
    self_noise = [float(np.max(np.abs(engine.decide(item_to_request(items[i], "vlm")).scoring.scores["q"].z - reference[i])))
                  for i in noisy]
    backend.batch_size = cfg.vlm.batch_size

    dz = [float(np.max(np.abs(r - c))) for r, c in zip(reference, cached)]
    agree = [_decision(r) == _decision(c) for r, c in zip(reference, cached)]
    per_suite = {}
    for suite in MIX:
        idx = [i for i, item in enumerate(items) if item.suite == suite]
        per_suite[suite] = {"items": len(idx), "statements_per_item": int(len(reference[idx[0]])),
                            "max_abs_dz": max(dz[i] for i in idx), "argmax_agree": sum(agree[i] for i in idx)}
    report = {
        "items": len(items), "statements": int(sum(len(r) for r in reference)),
        "argmax_agree": int(sum(agree)), "max_abs_dz": max(dz), "median_abs_dz": float(np.median(dz)),
        "threshold": MAX_DZ, "per_suite": per_suite,
        "reference_seconds": t_ref, "cached_seconds": t_cached, "speedup": t_ref / t_cached,
        "reference_self_noise_bs8_vs_bs4": {"items": len(noisy), "max_abs_dz": max(self_noise), "median_abs_dz": float(np.median(self_noise))},
        "model": f"{backend.model_id}@{backend.revision}", "device": backend.device, "dtype": backend.dtype,
    }
    (cfg.path("logs") / "cache_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n[M4] " + json.dumps(report, indent=2))

    assert report["argmax_agree"] == 100, "cached path changed a decision"
    assert report["max_abs_dz"] <= MAX_DZ, f"max |dz| {report['max_abs_dz']:.4f} > {MAX_DZ}"

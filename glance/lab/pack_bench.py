"""Score lab (GPU): what does a rating cost on a FRESH image, and how much does packing questions save?

`lab/LATENCY.json` timed each readout inside the collector's per-item loop, where every readout after the first finds
its image prefix already cached. Those numbers are the marginal cost of one more readout, not the cost of rating a new
image. This benchmark clears the prefix cache before every measurement and times end to end from the image file:

- single readouts and `ens4d` called one readout at a time (how the experiments were collected);
- `ens4d` PACKED: one prefill of the image with both digit readouts behind it, one prefill of image + zoom crop with
  both zoom readouts behind it (2 prefills, 2 batched suffix reads);
- the same with several rating questions packed behind the same two prefills (5 lab scales; 25 KADID-style questions).

Packing changes batch composition, which perturbs half-precision logits, so the packed logits of each item's own scale
are saved in collector format for `glance.lab.refcheck` (same calibration, same prediction?).

uv run python -m glance.lab.pack_bench --limit 20 --out lab/PACKING
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT
from ..images import LoadedImage
from ..logging_utils import JsonlWriter
from . import score_methods as sm
from .collect import Collector, load_items, load_ladder_meta

ENS4D = ("digits", "zoom_digits", "digitsrev", "zoom_digitsrev")


class Bench:
    def __init__(self):
        self.collector = Collector(prefix_cache=True)
        self.backend = self.collector.backend

    def fresh(self, cached: bool = True) -> None:
        self.backend._prefix_cache.clear()
        self.backend.use_prefix_cache = cached

    def one_at_a_time(self, methods: tuple[str, ...], meta: dict[str, Any], item: dict[str, Any]) -> dict[str, list[float]]:
        return {m: self.collector.run(m, meta, item, [])["logits"] for m in methods}

    def packed(self, metas: dict[str, dict[str, Any]], item: dict[str, Any], zoom: bool = True) -> dict[str, dict[str, list[float]]]:
        """Every question in `metas`, digits + digitsrev (+ the two zoom readouts): at most two prefills in total."""
        target = self.collector._image("img0", item["path"])
        views = [("", [target])]
        if zoom:
            crop = sm.zoom_crop(target.image, position="c")
            views.append(("zoom_", [target, LoadedImage(id="zoom", image=crop, sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                                                        width=crop.width, height=crop.height, format="PNG", source="derived:zoom")]))
        out: dict[str, dict[str, list[float]]] = {scale: {} for scale in metas}
        for prefix, images in views:
            blocks, index, labels = [], [], None
            for scale, meta in metas.items():
                instructions = sm.with_zoom(meta["instructions"]) if prefix else meta["instructions"]
                for reverse in (False, True):
                    block, block_labels = sm.digits_block(instructions, meta["levels"], reverse=reverse)
                    if labels is not None and block_labels != labels:
                        raise ValueError("packed questions must share one label set")
                    labels = block_labels
                    blocks.append(block)
                    index.append((scale, f"{prefix}digits{'rev' if reverse else ''}", reverse))
            scores = self.backend.score_labels(images, None, blocks, labels)
            for (scale, method, reverse), logits in zip(index, scores.logits):
                out[scale][method] = [float(v) for v in (logits[::-1] if reverse else logits)]
        return out


def timed(fn) -> tuple[float, Any]:
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=20, help="test items per lab scale")
    parser.add_argument("--many-every", type=int, default=4, help="run the 25-question configuration on every Nth item")
    parser.add_argument("--out", default="lab/PACKING")
    parser.add_argument("--rows", default="lab/runs/pack_bench.jsonl")
    parser.add_argument("--check", default="lab/runs/packed_check.jsonl", help="packed logits in collector format, for refcheck")
    args = parser.parse_args(argv)

    lab = load_ladder_meta("ladders")
    many = dict(list(load_ladder_meta("kadid").items()))  # 25 five-level questions; asked of lab images for timing only
    bench = Bench()
    for path in (args.rows, args.check):
        (PROJECT_ROOT / path).unlink(missing_ok=True)
    rows, check = JsonlWriter(PROJECT_ROOT / args.rows), JsonlWriter(PROJECT_ROOT / args.check)

    items = [(scale, item) for scale in lab for item in [i for i in load_items(scale) if i["split"] == "test"][: args.limit]]
    for scale, item in items[:3]:  # warm-up: MPS compiles kernels per new shape
        bench.fresh()
        bench.packed(lab, item)
        bench.packed(many, item)
        bench.fresh(cached=False)
        bench.one_at_a_time(("independent",) + ENS4D, lab[scale], item)

    configs: dict[str, tuple[int, Any]] = {}
    for n, (scale, item) in enumerate(items):
        meta = lab[scale]
        configs = {
            # name: (rating questions answered, callable)
            "independent, no cache (v0 as shipped)": (1, lambda: (bench.fresh(False), bench.one_at_a_time(("independent",), meta, item))),
            "independent": (1, lambda: (bench.fresh(), bench.one_at_a_time(("independent",), meta, item))),
            "digits": (1, lambda: (bench.fresh(), bench.one_at_a_time(("digits",), meta, item))),
            "zoom_digits": (1, lambda: (bench.fresh(), bench.one_at_a_time(("zoom_digits",), meta, item))),
            "ens4d, no cache, one readout at a time": (1, lambda: (bench.fresh(False), bench.one_at_a_time(ENS4D, meta, item))),
            "ens4d, cached, one readout at a time": (1, lambda: (bench.fresh(), bench.one_at_a_time(ENS4D, meta, item))),
            "ens4d packed, 1 question": (1, lambda: (bench.fresh(), bench.packed({scale: meta}, item))),
            "ens4d packed, 5 questions": (5, lambda: (bench.fresh(), bench.packed(lab, item))),
            "digits+digitsrev packed (no zoom), 5 questions": (5, lambda: (bench.fresh(), bench.packed(lab, item, zoom=False))),
        }
        if n % args.many_every == 0:
            configs["ens4d packed, 25 questions"] = (25, lambda: (bench.fresh(), bench.packed(many, item)))
            configs["digits+digitsrev packed (no zoom), 25 questions"] = (25, lambda: (bench.fresh(), bench.packed(many, item, zoom=False)))
        for name, (questions, fn) in configs.items():
            ms, (_, result) = timed(fn)
            rows.write({"config": name, "questions": questions, "ladder": scale, "item_id": item["item_id"], "latency_ms": ms})
            if name == "ens4d packed, 5 questions":
                for method, logits in result[scale].items():
                    check.write({"bench": "ladders", "ladder": scale, "item_id": item["item_id"], "split": item["split"],
                                 "level": item["level"], "method": method, "anchors": None, "method_key": method,
                                 "prompt_version": sm.LAB_PROMPT_VERSION, "prefix_cache": True, "logits": logits,
                                 "latency_ms": 0.0, "forward_passes": 1, "packed": True})
        if (n + 1) % 10 == 0:
            print(f"[pack {time.strftime('%H:%M:%S')}] {n + 1}/{len(items)} images", file=sys.stderr, flush=True)

    from ..logging_utils import read_jsonl

    by: dict[str, list[dict[str, Any]]] = {}
    for r in read_jsonl(PROJECT_ROOT / args.rows):
        by.setdefault(r["config"], []).append(r)
    summary = {name: {"images": len(group), "questions": group[0]["questions"],
                      "p50_ms": float(np.percentile([r["latency_ms"] for r in group], 50)),
                      "p90_ms": float(np.percentile([r["latency_ms"] for r in group], 90)),
                      "p50_ms_per_question": float(np.percentile([r["latency_ms"] for r in group], 50)) / group[0]["questions"]}
               for name, group in by.items()}
    lines = ["# Cost of a rating on a fresh image, and what packing saves", "",
             "Prefix cache cleared before every measurement; timed end to end from the image file (decode, resize, zoom crop, "
             "forward passes, readout). Uncontended GPU. `lab/LATENCY.json` reports the MARGINAL cost of one more readout on an "
             "image whose prefix is already cached; these are the full costs.", "",
             "| Configuration | images | questions answered | p50 ms | p90 ms | p50 ms per question |", "| --- | --- | --- | --- | --- | --- |"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['images']} | {s['questions']} | {s['p50_ms']:.0f} | {s['p90_ms']:.0f} | {s['p50_ms_per_question']:.0f} |")
    text = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / args.out
    out.with_suffix(".md").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score lab, step 1 (GPU): run every rating method on the ladder items and save the raw logits.

Only logits are stored, so every calibration and metric can be recomputed offline by `glance.lab.analyze` without
touching the model again. Rows are appended as they are produced; re-running skips what is already there.

uv run python -m glance.lab.collect --out lab/runs/main.jsonl --prefix-cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT, load_config
from ..images import load_image
from ..logging_utils import JsonlWriter, read_jsonl
from ..schema import ImageRef
from . import score_methods as sm

LAB_DIR = PROJECT_ROOT / "lab"


def load_ladder_meta() -> dict[str, dict[str, Any]]:
    from .ladders import LADDERS

    return LADDERS


def load_items(ladder: str) -> list[dict[str, Any]]:
    return read_jsonl(LAB_DIR / "manifests" / f"{ladder}.jsonl")


def anchors_for(ladder: str, config: str) -> list[sm.Anchor]:
    """`config` is "<ref>:<l1><l2><l3>", e.g. "0:123" = anchor source 0 shown at levels 1, 2 and 3."""
    ref, levels = config.split(":")
    entry = json.loads((LAB_DIR / "anchors.json").read_text())[ladder][int(ref)]
    return [sm.Anchor(image_id=f"ref{level}", level=int(level), path=entry["paths"][int(level)]) for level in levels]


class Collector:
    def __init__(self, prefix_cache: bool):
        from ..backends.vlm_hf import VlmBackend

        self.cfg = load_config(overrides={"vlm": {"prefix_cache": prefix_cache}})
        self.backend = VlmBackend(self.cfg)
        self._anchor_images: dict[str, Any] = {}

    def _image(self, image_id: str, path: str):
        return load_image(ImageRef(id=image_id, path=path), self.cfg.limits)

    def _anchors(self, anchors: list[sm.Anchor]):
        out = []
        for a in anchors:
            key = f"{a.image_id}|{a.path}"
            if key not in self._anchor_images:
                self._anchor_images[key] = self._image(a.image_id, a.path)
            out.append(self._anchor_images[key])
        return out

    def run(self, method: str, meta: dict[str, Any], item: dict[str, Any], anchors: list[sm.Anchor]) -> dict[str, Any]:
        levels, instructions = meta["levels"], meta["instructions"]
        target = self._image("img0", item["path"])
        images = [target]
        if method.startswith("anchors_"):
            # References first, the image being rated last, right before the question.
            images = self._anchors(anchors) + [target]
            start = 1 if "cumulative" in method else 0
            instructions = sm.with_anchors(instructions, levels, anchors, start)
        t0 = time.perf_counter()
        if method == "independent":
            out = self.backend.score_statements(images, None, sm.independent_statements(instructions, levels))
            logits, off = out.z, out.off_mass
        elif method.endswith("cumulative"):
            out = self.backend.score_statements(images, None, sm.cumulative_statements(instructions, levels))
            logits, off = out.z, out.off_mass
        else:
            block, labels = sm.digits_block(instructions, levels)
            out = self.backend.score_labels(images, None, [block], labels)
            logits, off = out.logits[0], out.off_mass
        return {
            "logits": [float(v) for v in logits], "off_mass_max": float(np.max(off)),
            "latency_ms": (time.perf_counter() - t0) * 1000, "image_tokens": out.usage.image_tokens,
            "forward_passes": out.usage.forward_passes, "cache_hit": out.cache_hit,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSONL to append to (resumable)")
    parser.add_argument("--ladders", default="blur,noise,jpeg,exposure,resolution")
    parser.add_argument("--methods", default=",".join(sm.METHODS))
    parser.add_argument("--anchors", default="0:123", help='anchor config "<ref>:<levels>"; comma-separate to run several')
    parser.add_argument("--split", choices=["calibration", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, help="first N items of each ladder's manifest (after the split filter)")
    parser.add_argument("--prefix-cache", action="store_true", help="use the prefix-cached path (experiments); omit for the reference path")
    args = parser.parse_args(argv)

    out_path = PROJECT_ROOT / args.out
    done = {(r["ladder"], r["item_id"], r["method_key"]) for r in read_jsonl(out_path)}
    writer = JsonlWriter(out_path)
    collector = Collector(prefix_cache=args.prefix_cache)
    ladder_meta = load_ladder_meta()
    methods = [m for m in args.methods.split(",") if m]
    anchor_configs = [c for c in args.anchors.split(",") if c]

    started = time.perf_counter()
    for ladder in args.ladders.split(","):
        items = [i for i in load_items(ladder) if args.split in ("all", i["split"])]
        if args.limit:
            items = items[: args.limit]
        # Single-image methods first, then both anchor methods per anchor config back to back, so consecutive calls
        # share an image prefix (the backend keeps the last two prefixes).
        jobs = [(m, None, m) for m in methods if not m.startswith("anchors_")]
        for config in anchor_configs:
            jobs += [(m, config, f"{m}@{config}") for m in methods if m.startswith("anchors_")]
        t_ladder = time.perf_counter()
        for index, item in enumerate(items, 1):
            for method, config, key in jobs:
                if (ladder, item["item_id"], key) in done:
                    continue
                anchors = anchors_for(ladder, config) if config else []
                row = collector.run(method, ladder_meta[ladder], item, anchors)
                writer.write({
                    "ladder": ladder, "item_id": item["item_id"], "split": item["split"], "level": item["level"],
                    "method": method, "anchors": config, "method_key": key, "prompt_version": sm.LAB_PROMPT_VERSION,
                    "prefix_cache": args.prefix_cache, **row,
                })
            if index % 50 == 0 or index == len(items):
                rate = (time.perf_counter() - t_ladder) / index
                print(f"[lab {time.strftime('%H:%M:%S')}] {ladder}: {index}/{len(items)} items, {rate:.2f} s/item", file=sys.stderr, flush=True)
    print(f"[lab] done in {(time.perf_counter() - started) / 60:.1f} min -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

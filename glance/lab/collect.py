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


BENCHES = {
    # name: (module that defines the scales, attribute with {scale: {"instructions", "levels"}}, manifest directory)
    "ladders": ("glance.lab.ladders", "LADDERS", "manifests"),
    "distort25": ("glance.lab.distort25", "SCALES", "manifests_distort25"),
    "kadid": ("glance.lab.kadid", "SCALES", "manifests_kadid"),
    "semantic": ("glance.lab.semantic", "SCALES", "manifests_semantic"),
}


def load_ladder_meta(bench: str = "ladders") -> dict[str, dict[str, Any]]:
    import importlib

    module, attr, _ = BENCHES[bench]
    return getattr(importlib.import_module(module), attr)


def load_items(ladder: str, bench: str = "ladders") -> list[dict[str, Any]]:
    return read_jsonl(LAB_DIR / BENCHES[bench][2] / f"{ladder}.jsonl")


def anchors_for(ladder: str, config: str) -> list[sm.Anchor]:
    """`config` is "<ref>:<l1><l2><l3>", e.g. "0:123" = anchor source 0 shown at levels 1, 2 and 3."""
    ref, levels = config.split(":")
    entry = json.loads((LAB_DIR / "anchors.json").read_text())[ladder][int(ref)]
    return [sm.Anchor(image_id=f"ref{level}", level=int(level), path=entry["paths"][int(level)]) for level in levels]


class HiddenStore:
    """Final hidden states of the digit readouts, one .npz per (bench, scale, method): `item_ids` and a float16 array.
    Logging only (for offline readout studies); not used by any registered result. Kept out of git (large)."""

    def __init__(self, bench: str, tag: str = ""):
        self.dir = LAB_DIR / "hidden" / (bench + (f"@{tag}" if tag else ""))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def add(self, ladder: str, key: str, item_id: str, vector: np.ndarray) -> None:
        self.pending.setdefault((ladder, key), {})[item_id] = vector

    def flush(self) -> None:
        for (ladder, key), new in self.pending.items():
            path = self.dir / f"{ladder}.{key.replace('@', '_at_').replace(':', '_')}.npz"
            merged: dict[str, np.ndarray] = {}
            if path.exists():
                old = np.load(path, allow_pickle=False)
                merged = dict(zip(old["item_ids"].tolist(), old["hidden"]))
            merged.update(new)
            ids = sorted(merged)
            np.savez(path, item_ids=np.array(ids), hidden=np.stack([merged[i] for i in ids]).astype(np.float16))
        self.pending = {}


class Collector:
    def __init__(self, prefix_cache: bool, keep_hidden: bool = False, second_model: dict[str, Any] | None = None):
        self.cfg = load_config(overrides={"vlm": {"prefix_cache": prefix_cache}})
        if second_model:  # replication on another model family (lab/NOTES.md entry 20, E3): plain HF forward passes
            from .generic_vlm import GenericVlm

            self.backend = GenericVlm(self.cfg, **second_model)
        else:
            from ..backends.vlm_hf import VlmBackend

            self.backend = VlmBackend(self.cfg)
        self.backend.keep_hidden = keep_hidden
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

    def run(self, method: str, meta: dict[str, Any], item: dict[str, Any], anchors: list[sm.Anchor],
            position: str | None = None) -> dict[str, Any]:
        levels, instructions = meta["levels"], meta["instructions"]
        self.last_hidden = None
        target = self._image("img0", item["path"])
        images = [target]
        if method.startswith("zoom_"):
            import hashlib

            from ..images import LoadedImage

            crop = sm.zoom_crop(target.image, position=position or "c")
            images = [target, LoadedImage(id="zoom", image=crop, sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                                          width=crop.width, height=crop.height, format="PNG", source="derived:zoom")]
            instructions = sm.with_zoom(instructions)
        if method.startswith("anchors_"):
            # References first, the image being rated last, right before the question.
            images = self._anchors(anchors) + [target]
            start = 1 if "cumulative" in method else 0
            instructions = sm.with_anchors(instructions, levels, anchors, start)
        t0 = time.perf_counter()
        if method.endswith("independent"):
            out = self.backend.score_statements(images, None, sm.independent_statements(instructions, levels))
            logits, off = out.z, out.off_mass
        elif method.endswith("cumulative"):
            out = self.backend.score_statements(images, None, sm.cumulative_statements(instructions, levels))
            logits, off = out.z, out.off_mass
        elif method in ("letter", "letter4", "poles"):
            # Other systems' readouts on the same model (lab/NOTES.md entry 24): option letters, not digits.
            from .. import prompts

            options = [levels[0], levels[-1]] if method == "poles" else list(levels)
            k = len(options)
            shifts = [round(r * k / 4) % k for r in range(4)] if method == "letter4" else [0]
            shifts = list(dict.fromkeys(shifts))
            blocks = [prompts.render_letter(instructions, options[s:] + options[:s]) for s in shifts]
            out = self.backend.score_labels(images, None, blocks, prompts.LETTER_LABELS[:k])
            per_option = np.stack([[out.logits[r, (i - s) % k] for i in range(k)] for r, s in enumerate(shifts)])
            logits, off = per_option.mean(axis=0), out.off_mass
        else:
            reverse = method.endswith("digitsrev")
            block, labels = sm.digits_block(instructions, levels, reverse=reverse)
            out = self.backend.score_labels(images, None, [block], labels)
            logits, off = (out.logits[0][::-1] if reverse else out.logits[0]), out.off_mass
            if self.backend.keep_hidden and self.backend.last_hidden is not None:
                self.last_hidden = self.backend.last_hidden[0]
        return {
            "logits": [float(v) for v in logits], "off_mass_max": float(np.max(off)),
            "latency_ms": (time.perf_counter() - t0) * 1000, "image_tokens": out.usage.image_tokens,
            "forward_passes": out.usage.forward_passes, "cache_hit": out.cache_hit,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="JSONL to append to (resumable)")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="ladders", help="which set of scales to run")
    parser.add_argument("--ladders", default=None, help="comma-separated scales (default: every scale of the bench)")
    parser.add_argument("--methods", default=",".join(sm.METHODS))
    parser.add_argument("--anchors", default="0:123", help='anchor config "<ref>:<levels>"; comma-separate to run several')
    parser.add_argument("--zoom-positions", default="c", help="crop positions for zoom_* methods: c,tl,tr,bl,br")
    parser.add_argument("--split", choices=["calibration", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, help="first N items of each ladder's manifest (after the split filter)")
    parser.add_argument("--prefix-cache", action="store_true", help="use the prefix-cached path (experiments); omit for the reference path")
    parser.add_argument("--save-hidden", action="store_true", help="also log the final hidden state of each digit readout to lab/hidden/ (not in git)")
    parser.add_argument("--model-id", help="replicate on another Hugging Face image-text-to-text model (generic reference path)")
    parser.add_argument("--revision", help="pinned revision of --model-id")
    parser.add_argument("--image-longest-edge", type=int, help="image size handed to the second model's image processor")
    args = parser.parse_args(argv)

    out_path = PROJECT_ROOT / args.out
    done = {(r["ladder"], r["item_id"], r["method_key"]) for r in read_jsonl(out_path)}
    writer = JsonlWriter(out_path)
    second = None
    if args.model_id:
        if not args.revision:
            parser.error("--model-id needs a pinned --revision")
        second = {"model_id": args.model_id, "revision": args.revision, "longest_edge": args.image_longest_edge}
    collector = Collector(prefix_cache=args.prefix_cache, keep_hidden=args.save_hidden, second_model=second)
    model_tag = f"{collector.backend.model_id}@{collector.backend.revision}"
    hidden = HiddenStore(args.bench, args.model_id.split("/")[-1] if args.model_id else "") if args.save_hidden else None
    ladder_meta = load_ladder_meta(args.bench)
    if not args.ladders:
        args.ladders = ",".join(ladder_meta)
    methods = [m for m in args.methods.split(",") if m]
    anchor_configs = [c for c in args.anchors.split(",") if c]

    started = time.perf_counter()
    for ladder in args.ladders.split(","):
        items = [i for i in load_items(ladder, args.bench) if args.split in ("all", i["split"])]
        if args.limit:
            items = items[: args.limit]
        # Single-image methods first, then both anchor methods per anchor config back to back, so consecutive calls
        # share an image prefix (the backend keeps the last two prefixes).
        jobs = [(m, None, m) for m in methods if not m.startswith("anchors_")]
        # Extra crop positions for the zoom methods (test-time augmentation). The centre crop keeps the plain key.
        for pos in [p for p in args.zoom_positions.split(",") if p and p != "c"]:
            jobs += [(m, f"pos:{pos}", f"{m}@{pos}") for m in methods if m.startswith("zoom_")]
        for config in anchor_configs:
            jobs += [(m, config, f"{m}@{config}") for m in methods if m.startswith("anchors_")]
        t_ladder = time.perf_counter()
        for index, item in enumerate(items, 1):
            for method, config, key in jobs:
                if (ladder, item["item_id"], key) in done:
                    continue
                position = config.split(":", 1)[1] if config and config.startswith("pos:") else None
                anchors = anchors_for(ladder, config) if config and not position else []
                row = collector.run(method, ladder_meta[ladder], item, anchors, position)
                if hidden is not None and collector.last_hidden is not None:
                    hidden.add(ladder, key, item["item_id"], collector.last_hidden)
                writer.write({
                    "bench": args.bench, "ladder": ladder, "item_id": item["item_id"], "split": item["split"], "level": item["level"],
                    "method": method, "anchors": config, "method_key": key, "prompt_version": sm.LAB_PROMPT_VERSION, "model": model_tag,
                    "prefix_cache": args.prefix_cache, **row,
                })
            if index % 50 == 0 or index == len(items):
                if hidden is not None:
                    hidden.flush()
                rate = (time.perf_counter() - t_ladder) / index
                print(f"[lab {time.strftime('%H:%M:%S')}] {ladder}: {index}/{len(items)} items, {rate:.2f} s/item", file=sys.stderr, flush=True)
    print(f"[lab] done in {(time.perf_counter() - started) / 60:.1f} min -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

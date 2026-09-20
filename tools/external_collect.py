"""Run one outside image-rating system (`glance.lab.external_systems`) over a lab bench and save its raw per-level
numbers, in `glance.lab.collect`'s own row schema, so `tools/external_report.py` and (if ever wanted)
`glance.lab.bench_report`/`glance.lab.analyze` can read them like any other collected run.

Resumable exactly like `glance.lab.collect`: item ids already present in `--out` are skipped, so a partial run can
be restarted with the same command.

uv run python tools/external_collect.py --system qsit --bench ladders --limit 600 \
    --out lab/runs/external_qsit.jsonl --device cpu

HARD RULE for this lab's machine (`lab/NOTES.md` entry 37): never load a model on mps/gpu, and never load
`--system openjev` at all (see `glance/lab/external_systems.py`'s module docstring). Nothing in this file enforces
that -- the registered design expects `--system openjev` to run for real once a machine has headroom -- so the
caller (a person, or an agent following the hard rule) is responsible for which `--device` and `--system` it is
actually invoked with on a given machine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from glance.config import PROJECT_ROOT
from glance.lab.collect import BENCHES, load_items, load_ladder_meta
from glance.lab.external_systems import METHOD_NAMES, SYSTEMS, ExternalSystem
from glance.logging_utils import JsonlWriter, read_jsonl


def collect(
    system: ExternalSystem,
    method: str,
    ladders: dict[str, list[dict[str, Any]]],
    ladder_meta: dict[str, dict[str, Any]],
    writer: JsonlWriter,
    done: set[tuple[str, str]],
    bench: str = "ladders",
    progress_every: int = 50,
    log: Any = sys.stderr,
) -> int:
    """Score every item of every ladder not already in `done`, writing one row each. Returns the number written."""
    model_tag = f"{system.model_id}@{system.revision}"
    written = 0
    for ladder, items in ladders.items():
        started = time.perf_counter()
        for index, item in enumerate(items, 1):
            if (ladder, item["item_id"]) not in done:
                t0 = time.perf_counter()
                logits = system.features(PROJECT_ROOT / item["path"], ladder_meta[ladder])
                ms = (time.perf_counter() - t0) * 1000
                writer.write({
                    "bench": bench, "ladder": ladder, "item_id": item["item_id"], "split": item["split"],
                    "level": item["level"], "method": method, "method_key": method, "system": system.name,
                    "model": model_tag, "logits": [float(v) for v in logits], "latency_ms": ms,
                })
                done.add((ladder, item["item_id"]))
                written += 1
            if log is not None and (index % progress_every == 0 or index == len(items)):
                rate = (time.perf_counter() - started) / index
                print(f"[external-collect {time.strftime('%H:%M:%S')}] {system.name}/{ladder}: {index}/{len(items)} items, "
                      f"{rate:.2f} s/item", file=log, flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--bench", default="ladders", choices=sorted(BENCHES))
    parser.add_argument("--ladders", default=None, help="comma-separated scales (default: every scale of the bench)")
    parser.add_argument("--split", choices=["calibration", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, help="first N items of each ladder's manifest (after the split filter)")
    parser.add_argument("--out", required=True, help="JSONL to append to (resumable)")
    parser.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default=None, help="default: the system's own recommended dtype")
    args = parser.parse_args(argv)

    out_path = PROJECT_ROOT / args.out
    done = {(r["ladder"], r["item_id"]) for r in read_jsonl(out_path)}
    writer = JsonlWriter(out_path)

    system_cls = SYSTEMS[args.system]
    system = system_cls()
    system.load(args.device, args.dtype)

    ladder_meta = load_ladder_meta(args.bench)
    ladder_names = args.ladders.split(",") if args.ladders else list(ladder_meta)
    ladders: dict[str, list[dict[str, Any]]] = {}
    for ladder in ladder_names:
        items = [i for i in load_items(ladder, args.bench) if args.split in ("all", i["split"])]
        if args.limit:
            items = items[: args.limit]
        ladders[ladder] = items

    started = time.perf_counter()
    written = collect(system, METHOD_NAMES[args.system], ladders, ladder_meta, writer, done, bench=args.bench)
    print(f"[external-collect] done in {(time.perf_counter() - started) / 60:.1f} min, {written} new rows -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

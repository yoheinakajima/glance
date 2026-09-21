"""E19 (lab/NOTES.md entry 46): yes/no and pick-one on the fresh photo sets for a model of ANOTHER family, loaded through
`GenericVlm` and scored by the harness's own scorer (shipped statement and independent readouts, no wording changed).
Uncalibrated decisions, all items. Resumable.

uv run python -m glance.lab.generic_eval --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct --revision 482adb537c021c86670beed01cd58990d01e72e4 --image-longest-edge 768 --out lab/runs/smolvlm2_fresh.jsonl
"""
import argparse
import sys
import time

import numpy as np

from ..config import PROJECT_ROOT, load_config
from ..evals.suites import SUITES
from ..images import load_image
from ..logging_utils import JsonlWriter, read_jsonl
from ..schema import ImageRef, parse_request
from ..scorer import score_questions
from .generic_vlm import GenericVlm

SUITE_NAMES = ["fresh_yesno", "fresh_choice", "inat_yesno", "inat_choice"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image-longest-edge", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    cfg = load_config()
    backend = GenericVlm(cfg, model_id=args.model_id, revision=args.revision, longest_edge=args.image_longest_edge)
    out = PROJECT_ROOT / args.out
    done = {(r["suite"], r["item_id"]) for r in read_jsonl(out)}
    writer = JsonlWriter(out)
    for suite in SUITE_NAMES:
        items = SUITES[suite].build(cfg, 600)
        for n, item in enumerate(items[: args.limit] if args.limit else items, 1):
            if (suite, item.item_id) in done:
                continue
            request = parse_request({"model": "vlm", "state": {"images": [{"id": "img0", "path": item.image_path}]}, "questions": {"q": item.question}})
            image = load_image(ImageRef(id="img0", path=item.image_path), cfg.limits)
            t0 = time.perf_counter()
            score = score_questions(backend, [image], None, request.questions, choice_method="independent").scores["q"]
            z = np.asarray(score.z, dtype=float)
            if score.qtype == "noul":
                correct = bool((z[0] >= 0) == bool(item.label))
            else:
                correct = bool(score.keys[int(z.argmax())] == item.label)
            writer.write({"suite": suite, "item_id": item.item_id, "type": score.qtype, "correct": correct, "z": [float(v) for v in z], "label": item.label,
                          "latency_ms": (time.perf_counter() - t0) * 1000, "model": f"{args.model_id}@{args.revision}"})
            if n % 50 == 0:
                print(f"[generic_eval {time.strftime('%H:%M:%S')}] {suite}: {n}/{len(items)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

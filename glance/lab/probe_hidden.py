"""E26 (b) of lab/NOTES.md entry 61: is a judgement the token readout gets wrong linearly present in the model's final hidden state?

For every image of a probe suite, the shipped pick-one request is scored as usual with `keep_hidden` on, and the final hidden
state at the answer position of the FIRST option's statement is kept (one vector per image; the image and the question are the
same for every option, only the candidate differs). Saved with the label and the token readout's own pick, so
`tools/probe_hidden_report.py` can fit the registered probe (PCA 32 + multinomial logistic regression, 5-fold CV) without the GPU.

uv run python -m glance.lab.probe_hidden --suites probe_stripes,probe_largest,probe_count --out lab/runs/probe_hidden.npz
"""
import argparse
import sys
import time

import numpy as np

from ..config import PROJECT_ROOT, load_config
from ..evals.suites import SUITES
from ..images import load_image
from ..pipeline import Engine
from ..schema import ImageRef, parse_request
from ..scorer import score_questions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(None, overrides={"vlm": {"prefix_cache": True}})
    backend = Engine(cfg, source="probe_hidden").backend("vlm")
    backend.keep_hidden = True
    suites, ids, labels, picks, hidden = [], [], [], [], []
    for suite in args.suites.split(","):
        items = SUITES[suite].build(cfg, 600)
        for n, item in enumerate(items[: args.limit] if args.limit else items, 1):
            request = parse_request({"model": "vlm", "state": {"images": [{"id": "img0", "path": item.image_path}]}, "questions": {"q": item.question}})
            image = load_image(ImageRef(id="img0", path=item.image_path), cfg.limits)
            score = score_questions(backend, [image], None, request.questions, choice_method="independent").scores["q"]
            suites.append(suite)
            ids.append(item.item_id)
            labels.append(str(item.label))
            picks.append(str(score.keys[int(np.argmax(score.z))]))
            hidden.append(np.asarray(backend.last_hidden[0], dtype=np.float16))  # request order: the first option's statement
            if n % 50 == 0:
                print(f"[probe_hidden {time.strftime('%H:%M:%S')}] {suite}: {n}/{len(items)}", file=sys.stderr, flush=True)
    np.savez(PROJECT_ROOT / args.out, suite=np.array(suites), item_id=np.array(ids), label=np.array(labels), token_pick=np.array(picks), hidden=np.stack(hidden))
    print(f"saved {len(ids)} vectors of width {hidden[0].shape[0]} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

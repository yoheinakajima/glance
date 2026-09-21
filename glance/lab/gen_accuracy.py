"""E11 (lab/NOTES.md entry 32): the same frozen Qwen3-VL-4B WRITING a structured (JSON) answer, scored against ground
truth, as the baseline for what the logit readout brings. One question per request, greedy decoding, token cap, no
thinking; an unparsable or not-allowed answer counts as wrong and is flagged. Rows also carry the wall time per item
(only meaningful when the GPU is otherwise idle).

Suites: the five `ladder_*` suites restricted to the test items the frontier baseline saw (run id given), and the
`fresh_yesno` / `fresh_choice` suites (all items).

uv run python -m glance.lab.gen_accuracy --frontier-run 20260920T165748Z-8ff72a --out lab/runs/gen_accuracy.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time

from ..config import PROJECT_ROOT
from ..evals.suites import SUITES
from ..logging_utils import JsonlWriter, read_jsonl
from .gen_bench import Bench, json_prompt, parse_json, valid

LADDERS = ["ladder_blur", "ladder_noise", "ladder_jpeg", "ladder_exposure", "ladder_resolution"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frontier-run", required=True, help="run whose frontier rows define the lab test items")
    parser.add_argument("--out", default="lab/runs/gen_accuracy.jsonl")
    parser.add_argument("--limit", type=int, help="items per suite (smoke tests)")
    parser.add_argument("--suites", help="comma-separated suites instead of the default set (e.g. inat_yesno,inat_choice)")
    parser.add_argument("--config", help="another model size (configs/scaling_*.yaml); the JSON request and the token cap do not change")
    args = parser.parse_args(argv)

    bench = Bench(args.config)
    seen_by_frontier = {(r["suite"], r["item_id"]) for r in read_jsonl(PROJECT_ROOT / "runs" / args.frontier_run / "predictions.jsonl")
                        if r["backend"] == "frontier"}
    out_path = PROJECT_ROOT / args.out
    done = {(r["suite"], r["item_id"]) for r in read_jsonl(out_path)}
    writer = JsonlWriter(out_path)
    for suite in (args.suites.split(",") if args.suites else LADDERS + ["fresh_yesno", "fresh_choice"]):
        items = SUITES[suite].build(bench.cfg, 600)
        if suite in LADDERS:
            items = [i for i in items if (suite, i.item_id) in seen_by_frontier]
        items = items[: args.limit] if args.limit else items
        for n, item in enumerate(items, 1):
            if (suite, item.item_id) in done:
                continue
            question = item.question
            cap = 48  # the model wraps its JSON in a code fence (about 10 tokens for one field); generous so nothing is cut off
            ms, text, tokens = bench.write(item.image_path, json_prompt({"answer": question}), cap)
            parsed = parse_json(text)
            value = parsed.get("answer") if parsed else None
            ok = parsed is not None and valid(question, value)
            if question["type"] == "noul":
                correct = ok and (value == "Yes") == bool(item.label)
            elif question["type"] == "choice":
                correct = ok and value == item.label
            else:
                correct = ok and int(value) == int(item.label)
            writer.write({"suite": suite, "item_id": item.item_id, "split": item.split, "type": question["type"], "valid": bool(ok),
                          "correct": bool(correct), "written": value if ok else None, "write_ms": ms, "write_tokens": tokens,
                          # kept since E25 (lab/NOTES.md entry 59) so a lenient score can be computed: raw text of an OPEN local model, its allowed answers, the label
                          "text": text[:300], "allowed": (["Yes", "No"] if question["type"] == "noul" else list(question["criteria"]) if question["type"] == "choice"
                                                          else [str(i) for i in range(len(question["criteria"]))]),
                          "label": (("Yes" if item.label else "No") if question["type"] == "noul" else str(item.label))})
            if n % 50 == 0 or n == len(items):
                print(f"[genacc {time.strftime('%H:%M:%S')}] {suite}: {n}/{len(items)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

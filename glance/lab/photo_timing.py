"""Clean write-versus-read timing ON THE PHOTOGRAPHS of the comparison matrix (the hosted models were timed on these
Commons files, which are larger than the lab's 448 px test images, so the open model must be timed on them too).
First 40 photos of `fresh_choice`: the 13-option pick-one question and the photo's own yes/no question, written as JSON
and read, one request each, GPU otherwise idle. Writes lab/PHOTO_TIMING[_<tag>].json.

uv run python -m glance.lab.photo_timing                                  # the default model
uv run python -m glance.lab.photo_timing --config configs/scaling_qwen3vl_8b.yaml --tag 8B
"""
import argparse
import json
import statistics
import sys

from ..config import PROJECT_ROOT
from ..evals.suites import SUITES
from .gen_bench import Bench, json_prompt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--tag", default="")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args(argv)
    bench = Bench(args.config)
    choice = SUITES["fresh_choice"].build(bench.cfg, 600)[: args.n]
    yesno = {i.item_id.split("__")[0]: i for i in SUITES["fresh_yesno"].build(bench.cfg, 600) if i.label is True}
    times = {"yesno": {"write": [], "read": []}, "choice": {"write": [], "read": []}}
    for n, item in enumerate(choice):
        pairs = [("choice", item.question)] + ([("yesno", yesno[item.item_id].question)] if item.item_id in yesno else [])
        for kind, question in pairs:
            questions = {"answer": question}
            ms_w, _, _ = bench.write(item.image_path, json_prompt(questions), 32)
            ms_r, _ = bench.read(item.image_path, questions, "fast2")
            if n >= 2:  # the first two photos are warm-up
                times[kind]["write"].append(ms_w)
                times[kind]["read"].append(ms_r)
        print(f"[photo_timing] {n + 1}/{len(choice)}", file=sys.stderr, flush=True)
    out = {kind: {"n": len(v["read"]), "write_p50_ms": statistics.median(v["write"]), "read_p50_ms": statistics.median(v["read"])} for kind, v in times.items()}
    out["model"] = f"{bench.backend.model_id}@{bench.backend.revision}"
    path = PROJECT_ROOT / f"lab/PHOTO_TIMING{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

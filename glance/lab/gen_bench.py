"""E6 of lab/NOTES.md entry 27 (GPU, needs an otherwise idle GPU): the same frozen Qwen3-VL-4B either WRITES its answers
(greedy generation, token cap, no thinking) or is READ (logits, no generation), on the same images and questions.

(a) one yes/no question; (b) five mixed questions as one JSON object vs five packed reads; (c) 25 rating questions as
one JSON object vs packed reads with `fast2` and with `ens4d`. Cold start per image (prefix cache cleared), end to end
from the image file. Reports p50 / p90 latency, how often the written answer equals the argmax of the read, and how often
written JSON fails to parse or names a value that is not allowed.

uv run python -m glance.lab.gen_bench --limit 8 --out lab/GENBENCH
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT, load_config
from ..images import load_image
from ..logging_utils import JsonlWriter, read_jsonl
from ..pipeline import Engine
from ..schema import ImageRef
from .collect import load_items, load_ladder_meta

YESNO = "Is there an animal in `img0`?"
CHOICE = {"instructions": "What kind of animal is in `img0`?", "criteria": {"dog": None, "cat": None, "other": None}}


def rating_questions(metas: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: {"type": "score", "instructions": m["instructions"], "criteria": list(m["levels"])} for name, m in metas.items()}


def json_prompt(questions: dict[str, dict[str, Any]]) -> str:
    """One instruction asking for a JSON object; the allowed values are spelled out per field."""
    lines = ["Answer every question about `img0`. Reply with one JSON object and nothing else.", ""]
    for name, q in questions.items():
        if q["type"] == "noul":
            allowed = '"Yes" or "No"'
        elif q["type"] == "choice":
            allowed = " or ".join(f'"{k}"' for k in q["criteria"])
        else:
            allowed = "; ".join(f"{i} = {text}" for i, text in enumerate(q["criteria"])) + " (answer with the number)"
        lines.append(f'"{name}": {q["instructions"]} Allowed: {allowed}')
    return "\n".join(lines)


class Bench:
    def __init__(self):
        self.cfg = load_config(overrides={"vlm": {"prefix_cache": True}})
        self.engine = Engine(self.cfg, source="gen_bench")
        self.backend = self.engine.backend("vlm")

    def read(self, path: str, questions: dict[str, dict[str, Any]], score_method: str) -> tuple[float, dict[str, Any]]:
        self.backend._prefix_cache.clear()
        t0 = time.perf_counter()
        trace = self.engine.decide({"model": "vlm", "state": {"images": [{"id": "img0", "path": path}]}, "questions": questions,
                                    "options": {"score_method": score_method, "calibrated": False}})
        ms = (time.perf_counter() - t0) * 1000
        picks = {}
        for name, a in trace.response.answers.items():
            picks[name] = ("Yes" if a.noul >= 0.5 else "No") if a.type == "noul" else (a.choice if a.type == "choice" else
                                                                                     int(max(a.probabilities, key=a.probabilities.get)))
        return ms, picks

    def write(self, path: str, prompt: str, max_new_tokens: int) -> tuple[float, str, int]:
        import torch

        b = self.backend
        t0 = time.perf_counter()
        image = load_image(ImageRef(id="img0", path=path), self.cfg.limits)
        pixel_values, grid, tokens_per_image = b._encode_images([image])
        ids = b._tokenize([b.render_prompt([image], None, prompt)], tokens_per_image)[0]
        input_ids = torch.tensor([ids], device=b.device)
        with torch.no_grad():
            out = b.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=(input_ids == b.image_token_id).long(),
                pixel_values=pixel_values.to(b.device), image_grid_thw=grid.to(b.device),
                max_new_tokens=max_new_tokens, do_sample=False,
            )
        if b.device == "mps":
            torch.mps.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        new = out[0, len(ids):]
        return ms, b.tokenizer.decode(new, skip_special_tokens=True), int(len(new))


def parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def valid(q: dict[str, Any], value: Any) -> bool:
    if q["type"] == "noul":
        return value in ("Yes", "No")
    if q["type"] == "choice":
        return value in q["criteria"]
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(q["criteria"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=8, help="test images per lab scale (5 scales)")
    parser.add_argument("--out", default="lab/GENBENCH")
    parser.add_argument("--rows", default="lab/runs/gen_bench.jsonl")
    parser.add_argument("--single", action="store_true", help="one question per request of each type (yes/no, pick-one, rating): the cells of the comparison matrix")
    args = parser.parse_args(argv)

    lab = load_ladder_meta("ladders")
    five = {"animal": {"type": "noul", "instructions": YESNO}, "kind": {"type": "choice", **CHOICE},
            **{k: v for k, v in list(rating_questions(lab).items())[:3]}}
    many = rating_questions(load_ladder_meta("kadid"))
    sets = {"1 yes/no": ({"animal": {"type": "noul", "instructions": YESNO}}, 16), "5 mixed": (five, 96), "25 ratings": (many, 400)}
    if args.single:
        one_rating = dict(list(rating_questions(lab).items())[:1])
        sets = {"1 yes/no": ({"animal": {"type": "noul", "instructions": YESNO}}, 16), "1 pick-one": ({"kind": {"type": "choice", **CHOICE}}, 24),
                "1 rating": (one_rating, 24)}

    bench = Bench()
    (PROJECT_ROOT / args.rows).unlink(missing_ok=True)
    rows = JsonlWriter(PROJECT_ROOT / args.rows)
    items = [i for scale in lab for i in [x for x in load_items(scale) if x["split"] == "test"][: args.limit]]
    for item in items[:2]:  # warm-up
        for name, (questions, cap) in sets.items():
            bench.read(item["path"], questions, "fast2")
            bench.write(item["path"], json_prompt(questions), cap)

    for n, item in enumerate(items, 1):
        for name, (questions, cap) in sets.items():
            reads = {"fast2": bench.read(item["path"], questions, "fast2")}
            if name != "1 yes/no":
                reads["ens4d"] = bench.read(item["path"], questions, "ens4d")
            ms_w, text, n_tokens = bench.write(item["path"], json_prompt(questions), cap)
            parsed = parse_json(text)
            fields_ok = {q: (parsed is not None and q in parsed and valid(questions[q], parsed[q])) for q in questions}
            agree = {q: bool(fields_ok[q] and parsed[q] == reads["fast2"][1][q]) for q in questions}
            rows.write({"set": name, "item_id": item["item_id"], "questions": len(questions), "write_ms": ms_w, "write_tokens": n_tokens,
                        "read_ms": {k: v[0] for k, v in reads.items()}, "json_parsed": parsed is not None,
                        "fields_valid": sum(fields_ok.values()), "fields_agree_with_read": sum(agree.values()),
                        "hit_token_cap": n_tokens >= cap})
        if n % 10 == 0:
            print(f"[gen {time.strftime('%H:%M:%S')}] {n}/{len(items)} images", file=sys.stderr, flush=True)

    summary = {}
    for name in sets:
        group = [r for r in read_jsonl(PROJECT_ROOT / args.rows) if r["set"] == name]
        q = group[0]["questions"]
        entry = {"images": len(group), "questions": q,
                 "write_p50_ms": float(np.percentile([r["write_ms"] for r in group], 50)), "write_p90_ms": float(np.percentile([r["write_ms"] for r in group], 90)),
                 "write_tokens_mean": float(np.mean([r["write_tokens"] for r in group])),
                 "json_parse_failures": int(sum(not r["json_parsed"] for r in group)), "hit_token_cap": int(sum(r["hit_token_cap"] for r in group)),
                 "invalid_field_rate": float(1 - np.mean([r["fields_valid"] / q for r in group])),
                 "agreement_with_read_on_valid_fields": float(np.sum([r["fields_agree_with_read"] for r in group]) / max(1, np.sum([r["fields_valid"] for r in group])))}
        for method in group[0]["read_ms"]:
            p50 = float(np.percentile([r["read_ms"][method] for r in group], 50))
            entry[f"read_{method}_p50_ms"] = p50
            entry[f"read_{method}_p90_ms"] = float(np.percentile([r["read_ms"][method] for r in group], 90))
            entry[f"speedup_vs_write_{method}"] = entry["write_p50_ms"] / p50
        summary[name] = entry
    lines = ["# Write the answers or read them? Same frozen Qwen3-VL-4B, same images, same questions", "",
             "Cold start per image, end to end from the image file, idle GPU. Writing = greedy generation of one JSON object with a token cap, no "
             "thinking. Reading = Glance (`glance decide`, uncalibrated), which also returns a probability for every answer.", "",
             "| Request | images | write p50 ms (tokens) | read `fast2` p50 ms | read `ens4d` p50 ms | read is faster by | JSON failed to parse | invalid fields | written = read (valid fields) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, e in summary.items():
        ens = f"{e['read_ens4d_p50_ms']:.0f}" if "read_ens4d_p50_ms" in e else "-"
        speed = f"{e['speedup_vs_write_fast2']:.1f}x" + (f" / {e['speedup_vs_write_ens4d']:.1f}x" if "speedup_vs_write_ens4d" in e else "")
        lines.append(f"| {name} | {e['images']} | {e['write_p50_ms']:.0f} ({e['write_tokens_mean']:.0f}) | {e['read_fast2_p50_ms']:.0f} | {ens} | {speed} | "
                     f"{e['json_parse_failures']} | {e['invalid_field_rate']:.1%} | {e['agreement_with_read_on_valid_fields']:.1%} |")
    out = PROJECT_ROOT / args.out
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

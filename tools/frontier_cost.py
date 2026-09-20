"""Measured cost and speed of the frontier API calls the owner ran (rows written by `glance baseline` carry LiteLLM's
`cost_usd`, `output_tokens` and `latency_ms` per call; runs made before that logging existed have latency only).
Writes results/lab/frontier_cost_measured.{json,md}. Latency is wall time per call from this laptop, one call at a time.

uv run python tools/frontier_cost.py
"""
import gzip
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = {  # run id -> what was asked
    "20260920T165748Z-8ff72a": "lab scales, one 4-level rating per call", "20260920T165949Z-8ff72a": "lab scales, one 4-level rating per call",
    "20260920T170146Z-8ff72a": "lab scales, one 4-level rating per call", "20260920T205633Z-99f822": "fresh photos, one yes/no or pick-one per call",
    "20260920T205633Z-99f822-gpt": "fresh photos, one yes/no or pick-one per call", "20260920T205633Z-99f822-gemini": "fresh photos, one yes/no or pick-one per call",
}


def rows_of(run_dir):
    for name in ("predictions.jsonl", "predictions.jsonl.gz"):
        path = run_dir / name
        if path.exists():
            with (gzip.open if name.endswith(".gz") else open)(path, "rt") as f:
                return [json.loads(line) for line in f if line.strip()]
    return []


out = []
for run_id, task in RUNS.items():
    for base in (ROOT / "runs", ROOT / "results" / "lab"):
        rows = [r for r in rows_of(base / run_id) if r.get("backend") == "frontier"]
        if rows:
            break
    if not rows:
        continue
    cost = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    tokens = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    out.append({"run": run_id, "model": str(rows[0]["model"]).removeprefix("frontier:"), "task": task, "calls": len(rows),
                "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
                "usd_per_1000_calls": 1000 * sum(cost) / len(cost) if cost else None, "usd_total": sum(cost) if cost else None,
                "median_output_tokens": statistics.median(tokens) if tokens else None})
(ROOT / "results/lab/frontier_cost_measured.json").write_text(json.dumps(out, indent=1))
md = ["# Measured cost and speed of the frontier calls", "",
      "| Model | Task | calls | median seconds per answer | measured $ per 1,000 answers | median output tokens |", "| --- | --- | --- | --- | --- | --- |"]
for r in out:
    usd = "not logged (run predates cost logging)" if r["usd_per_1000_calls"] is None else f"${r['usd_per_1000_calls']:.2f}"
    tok = "-" if r["median_output_tokens"] is None else f"{r['median_output_tokens']:.0f}"
    md.append(f"| {r['model']} | {r['task']} | {r['calls']} | {r['median_latency_ms'] / 1000:.1f} | {usd} | {tok} |")
(ROOT / "results/lab/frontier_cost_measured.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

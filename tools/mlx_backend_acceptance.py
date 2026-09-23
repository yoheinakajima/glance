#!/usr/bin/env python3
"""Acceptance benchmark for the experimental packaged MLX backend.

Run a pinned PyTorch/MPS Glance server first, with prefix sharing enabled and
the 2B model selected. This tool then pairs fresh-frame HTTP requests against
`Glance(backend="mlx")`, checks the registered semantic and latency gates, and
writes a raw local result. Fixture images and the result are intentionally not
committed.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import platform
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from glance import Glance, prompts
from glance.backends.base import Statement
from glance.images import load_images
from glance.schema import ImageRef


REFERENCE_MODEL = "vlm:Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203"
MLX_MODEL = "mlx-community/Qwen3-VL-2B-Instruct-8bit"
MLX_REVISION = "b0338e0e843d8e1befe873d144b81fefdc47efa6"
QUESTIONS: dict[str, dict[str, Any]] = {
    "content": {
        "type": "choice",
        "instructions": "What kind of content is shown in `img0`?",
        "criteria": {"animal photo": None, "receipt": None, "invoice": None, "other": None},
    },
    "text": {"type": "noul", "instructions": "Is readable text visible in `img0`?"},
    "photo": {"type": "noul", "instructions": "Is `img0` primarily a natural photograph?"},
    "tone": {
        "type": "choice",
        "instructions": "What is the overall visual tone of `img0`?",
        "criteria": {"mostly light": None, "mostly dark": None, "mixed": None},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="http://127.0.0.1:8077")
    parser.add_argument("--reference-core", required=True, type=Path, help="checkout used by the reference server")
    parser.add_argument("--fixtures", required=True, type=Path, help="folder containing dog/receipt/invoice-320-q60.jpg")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--reference-image-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("lab/runs/mlx_backend_acceptance.json"))
    return parser.parse_args()


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"content-type": "application/json"}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as response:
        return json.load(response)


def validate_reference(base_url: str, image_tokens: int) -> dict[str, Any]:
    report = request_json(f"{base_url.rstrip('/')}/v1/models")
    matches = [item for item in report.get("loaded", []) if item.get("name") == "vlm"]
    if len(matches) != 1:
        raise RuntimeError("reference server must have exactly one loaded `vlm` backend")
    model = matches[0]
    if model.get("model") != REFERENCE_MODEL or model.get("image_token_budget") != image_tokens:
        raise RuntimeError(
            f"reference must be {REFERENCE_MODEL} with image_token_budget={image_tokens}; got {model}"
        )
    return model


def fresh_jpeg(path: Path, tag: str) -> bytes:
    data = path.read_bytes()
    if not data.endswith(b"\xff\xd9"):
        raise ValueError(f"{path} is not a JPEG with an end marker")
    encoded = tag.encode("ascii")
    segment = b"\xff\xfe" + (len(encoded) + 2).to_bytes(2, "big") + encoded
    return data[:-2] + segment + data[-2:]


def log_offsets(log_dir: Path) -> dict[Path, int]:
    return {path: path.stat().st_size for path in log_dir.glob("*.jsonl")}


def find_new_log(log_dir: Path, offsets: dict[Path, int], request_id: str) -> dict[str, Any]:
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open() as rows:
            rows.seek(offsets.get(path, 0))
            for line in rows:
                row = json.loads(line)
                if row.get("request_id") == request_id:
                    return row
    raise RuntimeError(f"could not find reference call log {request_id}")


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * proportion) - 1))]


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
    }


def decision(answer: dict[str, Any]) -> str:
    if answer["type"] == "noul":
        return "yes" if float(answer["noul"]) >= 0.5 else "no"
    return str(answer["choice"])


def answer_vector(answer: dict[str, Any]) -> dict[str, float]:
    if answer["type"] == "noul":
        probability = float(answer["noul"])
        return {"yes": probability, "no": 1.0 - probability}
    return {str(key): float(value) for key, value in answer["probabilities"].items()}


def statement_blocks() -> list[str]:
    blocks = []
    for question in QUESTIONS.values():
        if question["type"] == "noul":
            blocks.append(prompts.render_noul(question["instructions"]))
        else:
            for key, description in question["criteria"].items():
                candidate = prompts.candidate_text(key, description)
                blocks.append(prompts.render_candidate(question["instructions"], candidate))
    return blocks


def scope_smoke(glance: Glance, fixtures: dict[str, Path]) -> dict[str, Any]:
    common = {"score_method": "digits", "calibrated": False}
    mixed = glance.engine.decide({
        "model": "vlm",
        "state": {"images": [{"id": "img0", "path": str(fixtures["dog"])}], "context": {"run": "acceptance"}},
        "questions": {
            "dog": {"type": "noul", "instructions": "Is there a dog in `img0`?"},
            "quality": {
                "type": "score", "instructions": "How clear is `img0`?",
                "criteria": ["Unclear", "Somewhat clear", "Clear", "Very clear"],
            },
        },
        "options": {"choice_method": "independent", **common},
    }).response.model_dump()
    letter = glance.engine.decide({
        "model": "vlm",
        "state": {"images": [{"id": "img0", "path": str(fixtures["dog"])}]},
        "questions": {"animal": {
            "type": "choice", "instructions": "What animal is in `img0`?",
            "criteria": {"dog": None, "cat": None, "bird": None},
        }},
        "options": {"choice_method": "letter", **common},
    }).response.model_dump()
    multi = glance.engine.decide({
        "model": "vlm",
        "state": {"images": [
            {"id": "img0", "path": str(fixtures["dog"])},
            {"id": "img1", "path": str(fixtures["receipt"])},
        ]},
        "questions": {"which": {
            "type": "choice", "instructions": "Which image is a receipt?",
            "criteria": {"img0": None, "img1": None},
        }},
        "options": {"choice_method": "independent", **common},
    }).response.model_dump()
    return {
        "mixed_answers": mixed["answers"],
        "letter_answer": letter["answers"]["animal"],
        "multi_answer": multi["answers"]["which"],
        "multi_usage": multi["usage"],
    }


def main() -> int:
    args = parse_args()
    fixtures = {name: args.fixtures / f"{name}-320-q60.jpg" for name in ("dog", "receipt", "invoice")}
    missing = [str(path) for path in fixtures.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing fixtures: {', '.join(missing)}")
    reference = validate_reference(args.reference, args.reference_image_tokens)

    glance = Glance(backend="mlx", calibrated=False, score_method="digits")
    started = time.perf_counter()
    backend = glance.engine.backend("vlm")
    routed_load_ms = (time.perf_counter() - started) * 1000
    if (backend.model_id, backend.revision) != (MLX_MODEL, MLX_REVISION):
        raise RuntimeError(f"unexpected MLX backend: {backend.model_id}@{backend.revision}")
    smoke = scope_smoke(glance, fixtures)

    images = load_images([ImageRef(id="img0", path=str(fixtures["dog"]))], glance.engine.cfg.limits)
    statements = [Statement(text=block, candidate="") for block in statement_blocks()]
    shared = backend.score_statements(images, None, statements).z.tolist()
    full = [float(backend.score_statements(images, None, [statement]).z[0]) for statement in statements]
    shared_max_z_delta = max(abs(left - right) for left, right in zip(shared, full))

    log_dir = args.reference_core.resolve() / "logs" / "calls"
    rows: list[dict[str, Any]] = []

    def run_reference(image: Path, image_name: str, repeat: int, warmup: bool) -> dict[str, Any]:
        raw = fresh_jpeg(image, f"mlx-acceptance-{time.time_ns()}-{repeat}")
        payload = {
            "model": "vlm",
            "state": {"images": [{"id": "img0", "base64": base64.b64encode(raw).decode()}]},
            "questions": QUESTIONS,
            "options": {"choice_method": "independent", "calibrated": False},
        }
        offsets = log_offsets(log_dir)
        started = time.perf_counter()
        response = request_json(f"{args.reference.rstrip('/')}/v1/decide", payload)
        wall_ms = (time.perf_counter() - started) * 1000
        logged = find_new_log(log_dir, offsets, response["request_id"])
        outputs = logged["output"]
        return {
            "backend": "torch", "image": image_name, "repeat": repeat, "warmup": warmup,
            "wall_ms": round(wall_ms, 3), "timing_ms": response["timing_ms"],
            "image_tokens": response["usage"]["image_tokens"],
            "answers": {qid: outputs[qid]["answer"] for qid in QUESTIONS},
            "z": {qid: outputs[qid]["z"] for qid in QUESTIONS},
        }

    def run_mlx(image: Path, image_name: str, repeat: int, warmup: bool) -> dict[str, Any]:
        body = {
            "model": "vlm", "state": {"images": [{"id": "img0", "path": str(image)}]},
            "questions": QUESTIONS, "options": {"choice_method": "independent", "calibrated": False},
        }
        started = time.perf_counter()
        trace = glance.engine.decide(body)
        wall_ms = (time.perf_counter() - started) * 1000
        response = trace.response.model_dump()
        return {
            "backend": "mlx", "image": image_name, "repeat": repeat, "warmup": warmup,
            "wall_ms": round(wall_ms, 3), "timing_ms": response["timing_ms"],
            "image_tokens": response["usage"]["image_tokens"], "answers": response["answers"],
            "z": {qid: trace.scoring.scores[qid].z.tolist() for qid in QUESTIONS},
        }

    mx.reset_peak_memory()
    for image_name, image in fixtures.items():
        for repeat in range(args.warmups + args.repeats):
            warmup = repeat < args.warmups
            runners: tuple[Callable[[], dict[str, Any]], ...]
            reference_runner = lambda p=image, n=image_name, r=repeat, w=warmup: run_reference(p, n, r, w)
            mlx_runner = lambda p=image, n=image_name, r=repeat, w=warmup: run_mlx(p, n, r, w)
            runners = (reference_runner, mlx_runner) if repeat % 2 == 0 else (mlx_runner, reference_runner)
            for runner in runners:
                row = runner()
                rows.append(row)
                print(f"{row['backend']} {image_name} {repeat + 1}/{args.warmups + args.repeats}: {row['wall_ms']:.1f} ms", flush=True)

    measured = [row for row in rows if not row["warmup"]]
    aggregate: dict[str, Any] = {}
    for runtime in ("torch", "mlx"):
        selected = [row for row in measured if row["backend"] == runtime]
        aggregate[runtime] = {
            "wall_ms": summarize([float(row["wall_ms"]) for row in selected]),
            "prefix_ms": summarize([float(row["timing_ms"]["prefix"]) for row in selected]),
            "score_ms": summarize([float(row["timing_ms"]["score"]) for row in selected]),
            "image_tokens": sorted({int(row["image_tokens"]) for row in selected}),
        }
    reference_p50 = float(aggregate["torch"]["wall_ms"]["p50"])
    mlx_p50 = float(aggregate["mlx"]["wall_ms"]["p50"])
    aggregate["mlx"].update({
        "speedup_vs_torch": round(reference_p50 / mlx_p50, 4),
        "latency_reduction_percent": round((1.0 - mlx_p50 / reference_p50) * 100.0, 1),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
    })

    keyed = {(row["image"], row["repeat"], row["backend"]): row for row in measured}
    decision_matches = 0
    decision_total = 0
    max_probability_delta = 0.0
    max_z_delta = 0.0
    for image_name in fixtures:
        for repeat in range(args.warmups, args.warmups + args.repeats):
            left = keyed[(image_name, repeat, "torch")]
            right = keyed[(image_name, repeat, "mlx")]
            for qid in QUESTIONS:
                decision_matches += int(decision(left["answers"][qid]) == decision(right["answers"][qid]))
                decision_total += 1
                left_vector, right_vector = answer_vector(left["answers"][qid]), answer_vector(right["answers"][qid])
                max_probability_delta = max(max_probability_delta, *(
                    abs(left_vector.get(key, 0.0) - right_vector.get(key, 0.0))
                    for key in set(left_vector) | set(right_vector)
                ))
                max_z_delta = max(max_z_delta, *(
                    abs(float(a) - float(b)) for a, b in zip(left["z"][qid], right["z"][qid])
                ))

    guardrail = {
        "decision_matches": decision_matches,
        "decision_total": decision_total,
        "max_probability_delta": round(max_probability_delta, 8),
        "max_z_delta": round(max_z_delta, 8),
        "passed": decision_matches == decision_total and max_probability_delta <= 0.10 and shared_max_z_delta <= 0.145,
    }
    result = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "mlx": version("mlx"), "mlx_vlm": version("mlx-vlm"),
            "mlx_model": f"{backend.model_id}@{backend.revision}", "backend_load_ms": round(backend.load_ms, 3),
            "routed_load_ms": round(routed_load_ms, 3), "reference": reference,
        },
        "protocol": {
            "warmups": args.warmups, "repeats_per_image": args.repeats, "images": list(fixtures),
            "questions": len(QUESTIONS), "decisions": decision_total, "fresh_reference_frames": True,
        },
        "scope_smoke": smoke,
        "shared_path_validation_max_z_delta": round(shared_max_z_delta, 8),
        "aggregate": aggregate,
        "guardrail": guardrail,
        "promotion_candidate": guardrail["passed"] and mlx_p50 <= reference_p50 * 0.95,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("aggregate", "guardrail", "promotion_candidate")}, indent=2))
    print(f"raw result: {args.output}")
    return 0 if result["promotion_candidate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

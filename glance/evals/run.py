"""Eval runner: builds suites, runs every (suite, backend, choice method) unit through the same decide() path as
the server, and writes runs/<run_id>/ (config.yaml, env.json, predictions.jsonl, errors.jsonl, metrics.json,
report.md, plots/, calibration/).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .. import __version__
from ..config import PROJECT_ROOT, Config
from ..doctor import run_doctor
from ..logging_utils import JsonlWriter, git_sha, read_jsonl
from ..pipeline import Engine
from ..schema import GlanceError
from .suites import DEFAULT_SUITES, SUITES, EvalItem, SuiteSkipped, item_to_request

LOCAL_BACKENDS = ("siglip", "vlm")


@dataclass
class RunArgs:
    suites: list[str] = field(default_factory=lambda: list(DEFAULT_SUITES))
    models: list[str] = field(default_factory=lambda: ["siglip", "vlm", "frontier"])
    n: int | None = None
    max_hours: float | None = None
    choice_methods: list[str] = field(default_factory=lambda: ["independent", "letter"])
    baseline_n: int | None = None
    confirm_spend: bool = False
    allow_upload_gold: bool = False
    skip_permutation: bool = False
    permutation_items: int | None = None  # default: eval.permutation_items (100)
    skip_latency: bool = False
    calibrate: bool = True  # fit on the calibration split, report raw and calibrated on the test split
    resume: str | None = None


@dataclass
class Unit:
    suite: str
    backend: str
    method: str  # "independent" | "letter" | "statement" (noul, score) | "pick" (frontier)
    items: list[EvalItem]
    seconds_per_item: float | None = None

    @property
    def key(self) -> str:
        return f"{self.suite}|{self.backend}|{self.method}"


def log(msg: str) -> None:
    print(f"[eval {datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --- rows ---------------------------------------------------------------------------------------------


def label_index(item: EvalItem, keys: list[str]) -> int:
    qtype = item.question["type"]
    if qtype == "noul":
        return int(bool(item.label))
    if qtype == "choice":
        return keys.index(str(item.label))
    return int(item.label)


def letter_subset(item: EvalItem, max_options: int, seed: int) -> dict[str, Any]:
    """`letter` is capped at 26 options. Larger questions keep the true label plus seeded random distractors,
    in their original order. `independent` is compared on the same subset by restricting its logits."""
    criteria = item.question["criteria"]
    if len(criteria) <= max_options:
        return item.question
    rng = random.Random(f"{seed}:{item.suite}:{item.item_id}")
    distractors = rng.sample([k for k in criteria if k != item.label], max_options - 1)
    keep = set(distractors) | {str(item.label)}
    return {**item.question, "criteria": {k: v for k, v in criteria.items() if k in keep}}


def suite_label(item: EvalItem) -> str:
    return f"{item.suite}/{item.question['type']}" if item.suite == "human_gold" else item.suite


def run_item(engine: Engine, unit: Unit, item: EvalItem, run_id: str, cfg: Config) -> dict[str, Any]:
    """One request through Engine.decide(). Raises GlanceError; the caller records failures."""
    question = item.question
    method = unit.method
    if method == "letter":
        question = letter_subset(item, cfg.eval.letter_max_options, cfg.eval.seed)
    body = item_to_request(item, unit.backend, "letter" if method == "letter" else "independent")
    body["questions"]["q"] = question

    t0 = time.perf_counter()
    trace = engine.decide(body, source=f"eval:{run_id}")
    latency_ms = (time.perf_counter() - t0) * 1000
    qs = trace.scoring.scores["q"]
    answer = trace.response.answers["q"]
    row: dict[str, Any] = {
        "run_id": run_id, "suite": suite_label(item), "item_id": item.item_id, "split": item.split,
        "backend": unit.backend, "model": trace.response.model, "choice_method": None if method in ("statement", "pick") else method,
        "method": method, "type": qs.qtype, "keys": qs.keys, "label": item.label,
        "label_index": label_index(item, qs.keys), "n_options_full": len(item.question.get("criteria") or []) or None,
        "request_id": trace.response.request_id, "image_path": item.image_path,
        "image_token_budget": engine.backend(unit.backend).image_token_budget,
        "prompt_version": trace.response.prompt_version,
        "image_tokens": trace.response.usage.image_tokens, "forward_passes": trace.response.usage.forward_passes,
        "text_tokens": trace.response.usage.text_tokens, "output_tokens": trace.response.usage.output_tokens,
        "cost_usd": trace.response.usage.cost_usd,
        "latency_ms": latency_ms, "human_disagreement": item.meta.get("human_disagreement"),
    }
    if qs.pick is not None:
        # Frontier outputs are evaluation-only: keep whether the pick was right, never the pick itself.
        picked = {"Yes": 1, "No": 0}[qs.pick] if qs.qtype == "noul" else qs.keys.index(qs.pick)
        row["correct"] = picked == row["label_index"]
        return row
    raw = answer.raw
    row["z"] = qs.z.tolist()
    row["raw"] = raw if qs.qtype == "noul" else [raw[k] for k in qs.keys]
    row["calibrated"] = None
    row["off_mass"] = [s["off_mass"] for s in qs.statements if s.get("off_mass") is not None]
    return row


# --- planning -----------------------------------------------------------------------------------------


def plan_units(cfg: Config, args: RunArgs, n: int, notes: list[str]) -> tuple[list[Unit], dict[str, Any]]:
    units: list[Unit] = []
    suites_meta: dict[str, Any] = {}
    frontier_ok = "frontier" in args.models and _frontier_ready(args, notes)
    for name in args.suites:
        module = SUITES[name]
        info = module.INFO
        meta = {"qtype": info.qtype, "source": info.source, "license": info.license, "status": "ok"}
        suites_meta[name] = meta
        try:
            items = module.build(cfg, n)
        except SuiteSkipped as exc:
            meta["status"] = f"skipped: {exc}"
            log(f"suite {name}: skipped ({exc})")
            continue
        meta["n_items"] = len(items)
        log(f"suite {name}: {len(items)} items ({sum(i.split == 'test' for i in items)} test)")
        for backend in args.models:
            if backend not in info.backends:
                if backend in LOCAL_BACKENDS:
                    notes.append(f"`{name}` does not run on `{backend}` (its questions carry no criteria the backend can use)")
                continue
            if backend == "frontier":
                if not frontier_ok:
                    continue
                if info.private and not args.allow_upload_gold:
                    notes.append("frontier baseline skipped on `human_gold`: needs --allow-upload-gold")
                    continue
                test_items = [i for i in items if i.split == "test"][: args.baseline_n or cfg.eval.baseline_n]
                units.append(Unit(name, backend, "pick", test_items))
                continue
            has_choice = any(i.question["type"] == "choice" for i in items)
            has_other = any(i.question["type"] != "choice" for i in items)
            if has_other:
                units.append(Unit(name, backend, "statement", [i for i in items if i.question["type"] != "choice"]))
            if has_choice:
                choice_items = [i for i in items if i.question["type"] == "choice"]
                for method in args.choice_methods:
                    if method == "letter" and backend != "vlm":
                        continue  # letter is VLM-only
                    units.append(Unit(name, backend, method, choice_items))
    return units, suites_meta


def _frontier_ready(args: RunArgs, notes: list[str]) -> bool:
    model = os.environ.get("FRONTIER_MODEL", "").strip()
    if not model:
        notes.append("frontier baseline not run: FRONTIER_MODEL is not set (accuracy gap reads \"not measured\")")
        return False
    import litellm

    env = litellm.validate_environment(model)
    if not env.get("keys_in_environment"):
        notes.append(f"frontier baseline not run: no API key for {model} (missing {env.get('missing_keys')})")
        return False
    if not args.confirm_spend:
        notes.append(f"frontier baseline not run: {model} is configured, but --confirm-spend was not given")
        return False
    return True


def estimate_frontier_cost(units: list[Unit], model: str | None = None) -> dict[str, Any]:
    """Rough spend estimate printed before any paid call. Deliberately on the high side."""
    import litellm

    model = model or os.environ.get("FRONTIER_MODEL", "")
    calls = sum(len(u.items) for u in units if u.backend == "frontier")
    # ~1,500 tokens for the image and template, the question text twice (prompt + JSON schema enum), and room for
    # models that think before they answer.
    tokens_in = sum(
        1500 + 2 * len(json.dumps(i.question)) // 3 for u in units if u.backend == "frontier" for i in u.items
    )
    tokens_out = 400 * calls
    price = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}
    usd = None
    if price.get("input_cost_per_token") is not None:
        usd = tokens_in * price["input_cost_per_token"] + tokens_out * (price.get("output_cost_per_token") or 0.0)
    return {"model": model, "calls": calls, "est_input_tokens": tokens_in, "est_output_tokens": tokens_out, "est_usd": usd}


# --- the run ------------------------------------------------------------------------------------------


def new_run_id(resolved: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{stamp}-{digest}"


def _env_snapshot(cfg: Config) -> dict[str, Any]:
    try:
        freeze = subprocess.run(["uv", "pip", "freeze"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
        packages = freeze.stdout.splitlines() if freeze.returncode == 0 else [f"uv pip freeze failed: {freeze.stderr.strip()}"]
    except (OSError, subprocess.SubprocessError) as exc:
        packages = [f"uv pip freeze failed: {exc}"]
    return {"doctor": run_doctor(cfg), "uv_pip_freeze": packages}


def run_eval(cfg: Config, args: RunArgs) -> Path:
    from . import report as report_module

    if args.permutation_items is not None:
        cfg.eval.permutation_items = args.permutation_items
    doctor = run_doctor(cfg)
    n = args.n or (cfg.eval.n_cuda if doctor["device"] == "cuda" else cfg.eval.n_apple)
    max_hours = args.max_hours if args.max_hours is not None else cfg.eval.max_hours
    notes: list[str] = []

    resolved = {"config": cfg.model_dump(), "args": asdict(args), "n_requested": n, "max_hours": max_hours,
                "harness_version": __version__, "git_sha": git_sha()}
    if args.resume:
        run_id = args.resume
        run_dir = cfg.path("runs") / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"no run to resume at {run_dir}")
    else:
        run_id = new_run_id(resolved)
        run_dir = cfg.path("runs") / run_id
        run_dir.mkdir(parents=True)
    log(f"run {run_id} -> {run_dir}")
    (run_dir / "env.json").write_text(json.dumps(_env_snapshot(cfg), indent=2) + "\n")

    units, suites_meta = plan_units(cfg, args, n, notes)
    frontier_units = [u for u in units if u.backend == "frontier"]
    if frontier_units:
        cost = estimate_frontier_cost(units)
        resolved["frontier_cost_estimate"] = cost
        usd = "unknown price" if cost["est_usd"] is None else f"~${cost['est_usd']:.2f}"
        log(f"frontier baseline: {cost['calls']} calls to {cost['model']}, ~{cost['est_input_tokens']} input tokens, {usd} (--confirm-spend given)")

    engine = Engine(cfg, source=f"eval:{run_id}")
    engine.allow_frontier = bool(frontier_units)  # only after _frontier_ready() saw --confirm-spend
    predictions = JsonlWriter(run_dir / "predictions.jsonl")
    errors = JsonlWriter(run_dir / "errors.jsonl")
    errors.path.touch()  # the file exists even when nothing fails
    done = {(r["suite"].split("/")[0], r["backend"], r["method"], r["item_id"]) for r in read_jsonl(run_dir / "predictions.jsonl")}
    failures: dict[str, int] = {}
    attempts: dict[str, int] = {}

    def process(unit: Unit, items: list[EvalItem]) -> float:
        t0 = time.perf_counter()
        for item in items:
            if (unit.suite, unit.backend, unit.method, item.item_id) in done:
                continue
            attempts[unit.key] = attempts.get(unit.key, 0) + 1
            try:
                predictions.write(run_item(engine, unit, item, run_id, cfg))
            except GlanceError as exc:
                failures[unit.key] = failures.get(unit.key, 0) + 1
                errors.write({"suite": unit.suite, "backend": unit.backend, "method": unit.method, "item_id": item.item_id,
                              "error": {"request_id": getattr(exc, "request_id", None), "code": exc.code,
                                        "message": exc.message, "detail": exc.detail}})
        return time.perf_counter() - t0

    # Phase A: a timed warmup of the first items of every unit, then an ETA and a trim to fit --max-hours.
    warm = cfg.eval.warmup_items
    started = time.perf_counter()
    for unit in units:
        head = unit.items[:warm]
        fresh = [i for i in head if (unit.suite, unit.backend, unit.method, i.item_id) not in done]
        engine.backend(unit.backend)  # load weights before the clock starts, so the ETA is per-item cost only
        seconds = process(unit, head)
        unit.seconds_per_item = seconds / max(1, len(fresh)) if fresh else 0.0
        log(f"warmup {unit.key}: {unit.seconds_per_item:.2f} s/item")

    def perm_items(unit: Unit, n_suite: int) -> int:
        if args.skip_permutation or unit.method not in ("independent", "letter"):
            return 0
        n_test = sum(i.split == "test" for i in unit.items[:n_suite])
        return min(n_test, cfg.eval.permutation_items)

    def suite_seconds(suite: str, n_suite: int) -> float:
        """Remaining cost of one suite at n items: every unit's items past the warmup, plus its permutation pass."""
        total = 0.0
        for unit in units:
            if unit.suite != suite:
                continue
            spi = unit.seconds_per_item or 0.0
            count = len(unit.items) if unit.backend == "frontier" else min(n_suite, len(unit.items))
            total += max(0, count - warm) * spi + perm_items(unit, n_suite) * cfg.eval.permutation_orders * spi
        return total

    # Trim to fit --max-hours. Every suite gets the same time cap, so cheap suites keep their full n and only the
    # expensive ones (many options = many forward passes per item) lose items. Trimming takes the first items of the
    # seeded order, which stay 50/50 calibration/test.
    suite_names = list(dict.fromkeys(u.suite for u in units))
    floor_n = min(n, 2 * warm)
    budget = max_hours * 3600 - (time.perf_counter() - started)

    def n_under_cap(suite: str, cap: float) -> int:
        best = floor_n
        for candidate in range(floor_n, n + 1, 2):
            if suite_seconds(suite, candidate) <= cap:
                best = candidate
        return best

    final_n = {suite: n for suite in suite_names}
    full_eta = sum(suite_seconds(suite, n) for suite in suite_names)
    if full_eta > budget:
        lo, hi = 0.0, max((suite_seconds(suite, n) for suite in suite_names), default=0.0)
        for _ in range(40):
            cap = (lo + hi) / 2
            if sum(suite_seconds(suite, n_under_cap(suite, cap)) for suite in suite_names) <= budget:
                lo = cap
            else:
                hi = cap
        final_n = {suite: n_under_cap(suite, lo) for suite in suite_names}
    final_eta = sum(suite_seconds(suite, final_n[suite]) for suite in suite_names)
    trim = {"requested_n": n, "final_n": final_n, "eta_hours_at_requested_n": round(full_eta / 3600, 2),
            "eta_hours_at_final_n": round(final_eta / 3600, 2), "max_hours": max_hours,
            "trimmed": any(v < n for v in final_n.values())}
    log(f"ETA at n={n}: {trim['eta_hours_at_requested_n']} h; budget {max_hours} h -> n per suite {final_n} "
        f"(ETA {trim['eta_hours_at_final_n']} h){' TRIMMED' if trim['trimmed'] else ''}")
    resolved["trim"] = trim
    resolved["notes"] = notes
    resolved["suites"] = suites_meta
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(resolved, default=str)), sort_keys=False))

    # Phase B: the rest of every unit.
    for unit in units:
        limit = len(unit.items) if unit.backend == "frontier" else max(warm, final_n[unit.suite])
        elapsed = process(unit, unit.items[warm:limit])
        unit.items = unit.items[:limit]
        log(f"done {unit.key}: {len(unit.items)} items, {elapsed / 60:.1f} min, {failures.get(unit.key, 0)} failures")

    extras: dict[str, Any] = {"permutation": {}, "latency": {}}
    if not args.skip_permutation:
        extras["permutation"] = permutation_sensitivity(engine, cfg, units, run_dir, run_id)
    if not args.skip_latency:
        extras["latency"] = latency_benchmark(engine, cfg, [m for m in args.models if m in LOCAL_BACKENDS])
    extras["failures"] = {u.key: {"failed": failures.get(u.key, 0), "attempted": len(u.items)} for u in units}
    extras["models"] = {name: f"{b.model_id}@{b.revision}" if b.revision else b.model_id for name, b in engine.loaded_backends().items()}
    extras["devices"] = {name: {"device": b.device, "dtype": b.dtype, "image_token_budget": b.image_token_budget}
                         for name, b in engine.loaded_backends().items()}
    extras["prefix_cache"] = cfg.vlm.prefix_cache
    (run_dir / "extras.json").write_text(json.dumps(extras, indent=2, default=str) + "\n")

    report_module.finalize_run(cfg, run_dir, calibrate=args.calibrate)
    log(f"wrote {run_dir / 'report.md'}")
    return run_dir


# --- permutation sensitivity and latency ------------------------------------------------------------------


def permutation_sensitivity(engine: Engine, cfg: Config, units: list[Unit], run_dir: Path, run_id: str) -> dict[str, Any]:
    """Test items x random option orders: the largest shift of any option's raw probability."""
    base_rows = {(r["suite"], r["backend"], r["method"], r["item_id"]): r for r in read_jsonl(run_dir / "predictions.jsonl")}
    out: dict[str, Any] = {}
    n_items = cfg.eval.permutation_items
    for unit in units:
        if unit.method not in ("independent", "letter"):
            continue
        rng = random.Random(f"{cfg.eval.seed}:perm:{unit.key}")
        items = [i for i in unit.items if i.split == "test"][:n_items]
        shifts: list[float] = []
        flips = 0
        t0 = time.perf_counter()
        for item in items:
            base = base_rows.get((suite_label(item), unit.backend, unit.method, item.item_id))
            if base is None:
                continue
            base_p = dict(zip(base["keys"], base["raw"]))
            question = letter_subset(item, cfg.eval.letter_max_options, cfg.eval.seed) if unit.method == "letter" else item.question
            for _ in range(cfg.eval.permutation_orders):
                keys = list(question["criteria"])
                rng.shuffle(keys)
                body = item_to_request(item, unit.backend, unit.method)
                body["questions"]["q"] = {**question, "criteria": {k: question["criteria"][k] for k in keys}}
                try:
                    answer = engine.decide(body, source=f"eval:{run_id}:permutation").response.answers["q"]
                except GlanceError:
                    continue
                shifts.append(max(abs(answer.raw[k] - base_p[k]) for k in base_p))
                flips += int(answer.choice != base["keys"][int(np.argmax(base["raw"]))])
        if shifts:
            out[unit.key] = {"items": len(items), "orders": cfg.eval.permutation_orders, "max_abs_dp": float(max(shifts)),
                             "mean_abs_dp": float(np.mean(shifts)), "choice_flips": flips, "requests": len(shifts)}
            log(f"permutation {unit.key}: max |dp| = {max(shifts):.2e} over {len(shifts)} requests "
                f"({(time.perf_counter() - t0) / 60:.1f} min)")
    return out


LATENCY_QUESTIONS = {
    "is_receipt": {"type": "noul", "instructions": "Is `img0` a photo or scan of a purchase receipt?",
                   "criteria": {"true": "a photo or scan of a purchase receipt", "false": "something that is not a receipt"}},
    "has_animal": {"type": "noul", "instructions": "Does `img0` show an animal?",
                   "criteria": {"true": "a photo of an animal", "false": "a photo with no animal"}},
    "doc_type": {"type": "choice", "instructions": "What kind of document is `img0`?",
                 "criteria": {"receipt": "Itemized proof of purchase from a store or restaurant",
                              "invoice": "Bill requesting payment, naming payer and payee", "other": None}},
    "orientation": {"type": "choice", "instructions": "How is `img0` oriented?",
                    "criteria": {"upright": "The image is the right way up", "sideways": "The image is rotated by a quarter turn",
                                 "upside_down": "The image is upside down"}},
    "legibility": {"type": "score", "instructions": "How legible is the text in `img0`?",
                   "criteria": ["Text cannot be read", "Some words readable, totals or names unclear", "All text clearly readable"]},
}


def latency_benchmark(engine: Engine, cfg: Config, backends: list[str]) -> dict[str, Any]:
    """1 image + 5 questions (11 statements), p50/p95 over repeats. Images rotate and the prefix cache is cleared,
    so every timed request pays for its image prefix."""
    from .metrics import latency_stats

    samples = ["samples/receipt.jpg", "samples/invoice.jpg", "samples/dog.jpg"]
    out: dict[str, Any] = {}
    for name in backends:
        backend = engine.backend(name)
        times = []
        for i in range(cfg.eval.latency_repeats + 2):
            if hasattr(backend, "_prefix_cache"):
                backend._prefix_cache.clear()
            body = {"model": name, "state": {"images": [{"id": "img0", "path": samples[i % len(samples)]}]},
                    "questions": LATENCY_QUESTIONS}
            t0 = time.perf_counter()
            engine.decide(body, source="eval:latency")
            if i >= 2:  # two untimed warmup requests
                times.append((time.perf_counter() - t0) * 1000)
        stats = latency_stats(times)
        stats.update({"questions": len(LATENCY_QUESTIONS), "statements": 11, "images": 1})
        out[name] = stats
        log(f"latency {name}: p50 {stats['p50_ms']:.0f} ms, p95 {stats['p95_ms']:.0f} ms")
    return out

"""`glance baseline`: add the frontier baseline to a finished eval run, with no .env to edit.

The command asks for the provider, the model and the API key in the terminal. The key is read with hidden input,
held in memory for this one process, passed straight to LiteLLM, and never written to disk or to a log. Picks are
evaluation-only: only whether each one was right is stored. The run's report is then rebuilt, which fills the two
go/no-go rows that need a baseline.
"""

from __future__ import annotations

import getpass
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from ..backends.frontier import FrontierBackend
from ..config import Config
from ..logging_utils import JsonlWriter, read_jsonl
from ..pipeline import Engine
from ..schema import GlanceError
from . import report as report_module
from .run import Unit, estimate_frontier_cost, log, run_item
from .suites import SUITES, SuiteSkipped

# Suggestions only: press Enter to take one, or type any LiteLLM model id. `key_hint` is where that provider's keys
# are issued, so the prompt can say what to paste.
PROVIDERS = [
    {"name": "Anthropic", "model": "anthropic/claude-opus-5", "key_hint": "console.anthropic.com -> API keys"},
    {"name": "OpenAI", "model": "openai/gpt-5.6", "key_hint": "platform.openai.com -> API keys"},
    {"name": "Google Gemini", "model": "gemini/gemini-3.1-pro-preview", "key_hint": "aistudio.google.com -> Get API key"},
]
SMOKE_QUESTION = {"type": "noul", "instructions": "Does `img0` show a dog?"}
SMOKE_IMAGE = "samples/dog.jpg"


def latest_full_run(cfg: Config) -> Path | None:
    """The run with the most local prediction rows: that is the full eval."""
    best, best_rows = None, 0
    for run_dir in sorted(cfg.path("runs").glob("*")):
        predictions = run_dir / "predictions.jsonl"
        if not predictions.exists() or not (run_dir / "config.yaml").exists():
            continue
        with open(predictions) as f:
            rows = sum(1 for _ in f)
        if rows > best_rows:
            best, best_rows = run_dir, rows
    return best


def ask_model_and_key(
    model: str | None, input_fn: Callable[[str], str] = input, secret_fn: Callable[[str], str] = getpass.getpass,
    out=sys.stderr,
) -> tuple[str, str]:
    hint = None
    if not model:
        print("\nWhich frontier model should be the baseline?", file=out)
        for i, provider in enumerate(PROVIDERS, 1):
            print(f"  {i}. {provider['name']:14s} {provider['model']}", file=out)
        print(f"  {len(PROVIDERS) + 1}. another LiteLLM model id", file=out)
        choice = input_fn(f"Choose 1-{len(PROVIDERS) + 1} [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(PROVIDERS):
            provider = PROVIDERS[int(choice) - 1]
            hint = provider["key_hint"]
            model = input_fn(f"Model id [{provider['model']}]: ").strip() or provider["model"]
        else:
            model = input_fn("LiteLLM model id (for example provider/model-name): ").strip()
    if not model:
        raise SystemExit("no model id given")
    where = f" (from {hint})" if hint else ""
    key = secret_fn(f"Paste the API key for {model}{where}. Input is hidden; it is kept in memory only: ").strip()
    if not key:
        raise SystemExit("no API key given")
    return model, key


def add_baseline(
    cfg: Config, run_dir: Path, model_id: str, api_key: str | None, baseline_n: int | None = None,
    allow_upload_gold: bool = False, confirmed: bool = False,
    input_fn: Callable[[str], str] = input, out=sys.stderr,
) -> Path | None:
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    run_id = run_dir.name
    limit = baseline_n or cfg.eval.baseline_n
    final_n = (run_config.get("trim") or {}).get("final_n") or {}
    suites = [name for name, meta in (run_config.get("suites") or {}).items() if str(meta.get("status")) == "ok"]

    existing = read_jsonl(run_dir / "predictions.jsonl")
    done = {(r["suite"].split("/")[0], r["item_id"]) for r in existing if r["backend"] == "frontier"}
    local = {(r["suite"].split("/")[0], r["item_id"]) for r in existing if r["backend"] != "frontier"}

    units: list[Unit] = []
    for name in suites:
        module = SUITES[name]
        if module.INFO.private and not allow_upload_gold:
            print(f"- {name}: skipped, private images need --allow-upload-gold", file=out)
            continue
        try:
            items = module.build(cfg, int(final_n.get(name) or run_config.get("n_requested") or cfg.eval.n_apple))
        except SuiteSkipped as exc:
            print(f"- {name}: skipped ({exc})", file=out)
            continue
        # Test split only, the first `limit` items of the seeded order, and only items the local backends answered.
        test = [i for i in items if i.split == "test" and (name, i.item_id) in local][:limit]
        todo = [i for i in test if (name, i.item_id) not in done]
        if todo:
            units.append(Unit(name, "frontier", "pick", todo))
    if not units:
        print("Nothing to do: this run already has a baseline for every suite.", file=out)
        return None

    backend = FrontierBackend(cfg, model_id=model_id, api_key=api_key)
    engine = Engine(cfg, source=f"eval:{run_id}:baseline", backends={"frontier": backend}, allow_frontier=True)

    # One cheap call first, so a wrong key or an unsupported model fails here and not 1,000 calls in.
    print(f"\nChecking {model_id} with one test call ...", file=out)
    try:
        trace = engine.decide({"model": "frontier", "state": {"images": [{"id": "img0", "path": SMOKE_IMAGE}]},
                               "questions": {"q": SMOKE_QUESTION}}, source=f"eval:{run_id}:baseline:check")
    except GlanceError as exc:
        print(f"The test call failed, nothing else was sent.\n  {exc.code}: {exc.message}", file=out)
        return None
    for warning in trace.response.warnings:
        print(f"  note: {warning}", file=out)
    print("  ok: key accepted, structured output works.", file=out)

    cost = estimate_frontier_cost(units, model=model_id)
    usd = "price unknown to LiteLLM" if cost["est_usd"] is None else f"about ${cost['est_usd']:.2f} at list price (an upper estimate)"
    per_suite = ", ".join(f"{u.suite} {len(u.items)}" for u in units)
    print(f"\nBaseline plan: {cost['calls']} calls to {model_id} ({per_suite}), test split only.", file=out)
    print(f"Estimated spend: {usd}. Each call sends one eval image to the provider.", file=out)
    if not confirmed:
        answer = input_fn("Run it? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Stopped. Nothing was spent beyond the one test call.", file=out)
            return None

    predictions = JsonlWriter(run_dir / "predictions.jsonl")
    errors = JsonlWriter(run_dir / "errors.jsonl")
    extras_path = run_dir / "extras.json"
    extras = json.loads(extras_path.read_text()) if extras_path.exists() else {}
    failures = extras.setdefault("failures", {})
    started = time.perf_counter()
    for unit in units:
        failed = 0
        for index, item in enumerate(unit.items, 1):
            try:
                predictions.write(run_item(engine, unit, item, run_id, cfg))
            except GlanceError as exc:
                failed += 1
                errors.write({"suite": unit.suite, "backend": "frontier", "method": "pick", "item_id": item.item_id,
                              "error": {"request_id": getattr(exc, "request_id", None), "code": exc.code,
                                        "message": exc.message, "detail": exc.detail}})
            if index % 25 == 0 or index == len(unit.items):
                print(f"\r  {unit.suite}: {index}/{len(unit.items)} ({failed} failed)   ", end="", file=out, flush=True)
        print(file=out)
        previous = failures.get(unit.key, {"failed": 0, "attempted": 0})
        failures[unit.key] = {"failed": previous["failed"] + failed, "attempted": previous["attempted"] + len(unit.items)}
    log(f"baseline done in {(time.perf_counter() - started) / 60:.1f} min")

    extras.setdefault("models", {})["frontier"] = model_id
    extras_path.write_text(json.dumps(extras, indent=2, default=str) + "\n")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    notes = [n for n in run_config.get("notes", []) if not n.startswith("frontier baseline not run")]
    notes.append(f"frontier baseline added on {stamp} with `{model_id}` via `glance baseline` "
                 f"(test split, up to {limit} items per suite; picks are not stored, only whether each was right)")
    run_config["notes"] = notes
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))

    report_module.finalize_run(cfg, run_dir)
    return run_dir

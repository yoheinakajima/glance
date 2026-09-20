"""glance doctor | decide | eval | calibrate | serve"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    cfg = load_config(args.config)
    report = run_doctor(cfg)
    out_path = cfg.path("logs") / "doctor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in report.items():
            print(f"{key:>20}: {value}")
    return 0


def _config_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if getattr(args, "image_token_budget", None):
        overrides.setdefault("models", {})["image_token_budget_override"] = args.image_token_budget
    if getattr(args, "prefix_cache", False):
        overrides.setdefault("vlm", {})["prefix_cache"] = True
    if getattr(args, "no_prefix_cache", False):
        overrides.setdefault("vlm", {})["prefix_cache"] = False
    return overrides


def _cmd_decide(args: argparse.Namespace) -> int:
    from .pipeline import Engine

    cfg = load_config(args.config, _config_overrides(args))
    try:
        body = json.loads(Path(args.request).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read request file {args.request}: {exc}", file=sys.stderr)
        return 2
    if isinstance(body, dict):
        if args.model:
            body["model"] = args.model
        if args.choice_method or args.calibrated is not None:
            options = body.setdefault("options", {})
            if args.choice_method:
                options["choice_method"] = args.choice_method
            if args.calibrated is not None:
                options["calibrated"] = args.calibrated
    status, payload = Engine(cfg, source="cli", allow_frontier=args.confirm_spend).decide_json(body)
    print(json.dumps(payload, indent=2))
    return 0 if status == 200 else 1


def _split(values: list[str] | None) -> list[str]:
    return [part for value in values or [] for part in value.split(",") if part]


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evals.run import RunArgs, run_eval
    from .evals.suites import DEFAULT_SUITES, SUITES

    cfg = load_config(args.config, _config_overrides(args))
    suites = _split(args.suite) or list(DEFAULT_SUITES)
    if suites == ["all"]:
        suites = list(DEFAULT_SUITES)
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        print(f"unknown suite(s) {unknown}; choose from {sorted(SUITES)}", file=sys.stderr)
        return 2
    run_args = RunArgs(
        suites=suites, models=_split(args.model) or ["siglip", "vlm", "frontier"], n=args.n, max_hours=args.max_hours,
        choice_methods=_split(args.choice_method) or ["independent", "letter"], baseline_n=args.baseline_n,
        confirm_spend=args.confirm_spend, allow_upload_gold=args.allow_upload_gold,
        skip_permutation=args.skip_permutation, skip_latency=args.skip_latency, resume=args.resume,
        calibrate=not args.no_calibrate, permutation_items=args.permutation_items,
    )
    run_dir = run_eval(cfg, run_args)
    print(run_dir)
    print((run_dir / "summary.txt").read_text())
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from . import calibration
    from .evals.report import fit_run_calibration
    from .logging_utils import read_jsonl

    cfg = load_config(args.config)
    run_dir = cfg.path("runs") / args.run
    rows = read_jsonl(run_dir / "predictions.jsonl")
    if not rows:
        print(f"no predictions at {run_dir}", file=sys.stderr)
        return 2
    # Fits from the finished run's calibration split. The model is not re-run.
    for params in fit_run_calibration(cfg, rows, args.run, isotonic=args.isotonic):
        path = calibration.save_params(cfg.path("calibration"), params)
        fits = ", ".join(f"{t}: {f.method} {({k: round(v, 4) for k, v in f.params.items() if not isinstance(v, list)})} n={f.n}"
                         for t, f in params.types.items())
        print(f"{params.version} [{params.key.backend}, {params.key.choice_method}] -> {path.relative_to(cfg.path('calibration').parent)}: {fits}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    cfg = load_config(args.config, _config_overrides(args))
    if args.port:
        cfg.server.port = args.port
    print(f"glance serving on http://{cfg.server.host}:{cfg.server.port} (preload: {_split(args.preload) or 'none'})", file=sys.stderr)
    serve(cfg, preload=_split(args.preload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glance", description="Image decision harness v0")
    parser.add_argument("--config", default=None, help="path to a config yaml (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="detect device, pick the model tier, write logs/doctor.json")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("decide", help="answer the questions in a request JSON file (same path as POST /v1/decide)")
    p.add_argument("request", help="path to a request JSON file")
    p.add_argument("--model", choices=["siglip", "vlm", "frontier"], help="override the request's model")
    p.add_argument("--choice-method", choices=["independent", "letter"], help="override options.choice_method")
    p.add_argument("--calibrated", action=argparse.BooleanOptionalAction, default=None, help="override options.calibrated")
    p.add_argument("--no-prefix-cache", action="store_true", help="VLM: use the reference path (every statement a full prompt)")
    p.add_argument("--confirm-spend", action="store_true", help="allow --model frontier: sends the image to a paid API")
    p.add_argument("--prefix-cache", action="store_true", help="VLM: opt in to the prefix-cached path (about 3.7x faster; see STATUS.md M4)")
    p.set_defaults(func=_cmd_decide)

    p = sub.add_parser("eval", help="run eval suites and write runs/<run_id>/")
    p.add_argument("--suite", action="append", help="suite name, repeatable or comma-separated (default: all)")
    p.add_argument("--model", action="append", help="siglip, vlm, frontier; repeatable (default: all three)")
    p.add_argument("--n", type=int, help="items per suite (default: 1000 on CUDA, 500 on Apple Silicon)")
    p.add_argument("--max-hours", type=float, help="time budget; n is trimmed to fit (default: 4)")
    p.add_argument("--choice-method", action="append", help="independent, letter (default: both)")
    p.add_argument("--baseline-n", type=int, help="frontier baseline items per suite, test split only (default: 300)")
    p.add_argument("--confirm-spend", action="store_true", help="allow paid frontier baseline calls after the cost estimate")
    p.add_argument("--allow-upload-gold", action="store_true", help="allow human_gold images to be sent to the frontier model")
    p.add_argument("--skip-permutation", action="store_true", help="skip the permutation sensitivity pass")
    p.add_argument("--permutation-items", type=int, help="test items per choice unit in the permutation pass (default: 100)")
    p.add_argument("--skip-latency", action="store_true", help="skip the 1 image + 5 questions latency benchmark")
    p.add_argument("--no-prefix-cache", action="store_true", help="VLM: use the reference path")
    p.add_argument("--no-calibrate", action="store_true", help="report raw probabilities only")
    p.add_argument("--image-token-budget", type=int, help="VLM: override the tier's image token budget (token sweep)")
    p.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run, skipping rows already written")
    p.add_argument("--prefix-cache", action="store_true", help="VLM: opt in to the prefix-cached path (about 3.7x faster; see STATUS.md M4)")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("calibrate", help="fit calibration params from a finished run's calibration split")
    p.add_argument("--run", required=True, metavar="RUN_ID")
    p.add_argument("--isotonic", action="store_true", help="isotonic regression for noul (needs >= 1000 calibration examples)")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("serve", help="local HTTP server on 127.0.0.1 (POST /v1/decide, GET /v1/models, GET /healthz)")
    p.add_argument("--port", type=int, help="port (default: server.port in the config)")
    p.add_argument("--preload", action="append", help="backends to load at startup: siglip, vlm (default: load on first use)")
    p.add_argument("--no-prefix-cache", action="store_true", help="VLM: use the reference path")
    p.add_argument("--prefix-cache", action="store_true", help="VLM: opt in to the prefix-cached path (about 3.7x faster; see STATUS.md M4)")
    p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""glance doctor | decide | score | fit | eval | calibrate | baseline | serve"""

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
    if getattr(args, "model_id", None):  # any Hugging Face VLM instead of the tier table
        overrides.setdefault("models", {})["generic"] = {"id": args.model_id, "revision": getattr(args, "revision", None),
                                                          "dtype": getattr(args, "dtype", None) or "auto", "image_longest_edge": getattr(args, "image_longest_edge", None)}
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
        if args.choice_method or args.score_method or args.calibrated is not None:
            options = body.setdefault("options", {})
            if args.choice_method:
                options["choice_method"] = args.choice_method
            if args.score_method:
                options["score_method"] = args.score_method
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


def _cmd_baseline(args: argparse.Namespace) -> int:
    import os

    from .evals.baseline import add_baseline, ask_model_and_key, latest_full_run, load_env_file

    cfg = load_config(args.config)
    run_dir = cfg.path("runs") / args.run if args.run else latest_full_run(cfg)
    if run_dir is None or not (run_dir / "predictions.jsonl").exists():
        print("no finished eval run to add a baseline to; run `glance eval` first", file=sys.stderr)
        return 2
    print(f"Adding a frontier baseline to run {run_dir.name}", file=sys.stderr)

    model = args.model or os.environ.get("FRONTIER_MODEL", "").strip() or None
    api_key = None
    key_in_env = False
    if model:
        import litellm

        env = litellm.validate_environment(model)
        if args.env_file and not env.get("keys_in_environment"):
            # Only the variable(s) this provider needs are read from the file; everything else in it is ignored.
            wanted = [str(k) for k in env.get("missing_keys") or []]
            try:
                found = load_env_file(args.env_file, wanted)
            except OSError as exc:
                print(f"could not read --env-file {args.env_file}: {exc.strerror or exc}", file=sys.stderr)
                return 2
            print(f"{args.env_file}: " + (f"loaded {', '.join(found)} (value not shown)" if found
                                           else f"no {' / '.join(wanted) or 'provider key'} defined there"), file=sys.stderr)
            env = litellm.validate_environment(model)
        key_in_env = bool(env.get("keys_in_environment"))
    if args.estimate_only:
        if not model:
            print("--estimate-only needs --model", file=sys.stderr)
            return 2
        add_baseline(cfg, run_dir, model, None, baseline_n=args.baseline_n, allow_upload_gold=args.allow_upload_gold,
                     estimate_only=True)
        print(f"provider key available to this process: {'yes' if key_in_env else 'NO'}", file=sys.stderr)
        return 0
    if not key_in_env:
        if not sys.stdin.isatty():
            print("This command asks for your API key with hidden input, so it needs a real terminal.\n"
                  "Paste this into your terminal:  uv run glance baseline", file=sys.stderr)
            return 2
        model, api_key = ask_model_and_key(model)
    done = add_baseline(cfg, run_dir, model, api_key, baseline_n=args.baseline_n,
                        allow_upload_gold=args.allow_upload_gold, confirmed=args.confirm_spend)
    if done is None:
        return 1
    print(f"\nReport rebuilt: {done / 'report.md'}\n")
    print((done / "summary.txt").read_text())
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    cfg = load_config(args.config, _config_overrides(args))
    if args.port:
        cfg.server.port = args.port
    print(f"glance serving on http://{cfg.server.host}:{cfg.server.port} (preload: {_split(args.preload) or 'none'})", file=sys.stderr)
    serve(cfg, preload=_split(args.preload))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """One rating question about one image, from the command line."""
    from .fit import read_rubric
    from .pipeline import Engine

    cfg = load_config(args.config, _config_overrides(args))
    if args.rubric:
        instructions, criteria = read_rubric(args.rubric)
    elif args.instructions and args.criteria:
        instructions, criteria = args.instructions, list(args.criteria)
    else:
        print("give the rubric as --rubric file.json, or as --instructions \"...\" --criteria \"lowest\" ... \"highest\"", file=sys.stderr)
        return 2
    body = {"model": "vlm", "state": {"images": [{"id": "img0", "path": args.image}]},
            "questions": {"score": {"type": "score", "instructions": instructions, "criteria": criteria}},
            "options": {"score_method": args.method, "calibrated": "auto"}}
    status, payload = Engine(cfg, source="cli").decide_json(body)
    print(json.dumps(payload if status != 200 or args.full else {**payload["answers"]["score"], "warnings": payload["warnings"]}, indent=2))
    return 0 if status == 200 else 1


def _cmd_ask(args: argparse.Namespace) -> int:
    """One question about one image, typed by its flags: plain = yes/no, --options = pick one, --levels = rating."""
    from .pipeline import Engine

    cfg = load_config(args.config, _config_overrides(args))
    text = args.question if "`img0`" in args.question else f"{args.question.rstrip()} (about `img0`)"
    if args.options and args.levels:
        print("give --options (pick one) or --levels (rating), not both", file=sys.stderr)
        return 2
    if args.options:
        question = {"type": "choice", "instructions": text, "criteria": {o: None for o in args.options}}
    elif args.levels:
        question = {"type": "score", "instructions": text, "criteria": list(args.levels)}
    else:
        question = {"type": "noul", "instructions": text}
    body = {"model": args.model, "state": {"images": [{"id": "img0", "path": args.image}]}, "questions": {"answer": question},
            "options": {"calibrated": "auto"}}
    status, payload = Engine(cfg, source="cli").decide_json(body)
    print(json.dumps(payload if status != 200 or args.full else {**payload["answers"]["answer"], "warnings": payload["warnings"]}, indent=2))
    return 0 if status == 200 else 1


def _cmd_fit(args: argparse.Namespace) -> int:
    from .fit import main_fit

    return main_fit(args)


def _rubric_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rubric", help='JSON file: {"instructions": "How ... is `img0`?", "criteria": ["lowest level", ..., "highest level"]}')
    p.add_argument("--instructions", help="the question, naming the image as `img0` (alternative to --rubric)")
    p.add_argument("--criteria", nargs="+", help="level descriptions, lowest first (alternative to --rubric)")
    p.add_argument("--method", choices=["ens4d", "fast2", "digits", "jsondigits"], default="ens4d", help="4 forward passes (default), 2 without the magnified crop, 1, or 1 pass read at the JSON answer position (best with nothing fitted)")
    p.add_argument("--no-prefix-cache", action="store_true", help="VLM: use the reference path")
    p.add_argument("--prefix-cache", action="store_true", help="VLM: share the image prefill between readouts (faster)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glance", description="Image decision harness v0")
    parser.add_argument("--config", default=None, help="path to a config yaml (default: configs/default.yaml)")
    parser.add_argument("--model-id", help="use this Hugging Face image-text-to-text model for `vlm` instead of the built-in Qwen3-VL tiers "
                        "(any model with a chat template; only SmolVLM2 and Qwen3-VL are measured in this repository)")
    parser.add_argument("--revision", help="commit of --model-id to pin (recommended)")
    parser.add_argument("--image-longest-edge", type=int, help="image size handed to --model-id's image processor")
    parser.add_argument("--dtype", help="auto (default), bfloat16, float16 or float32 for --model-id")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="detect device, pick the model tier, write logs/doctor.json")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("decide", help="answer the questions in a request JSON file (same path as POST /v1/decide)")
    p.add_argument("request", help="path to a request JSON file")
    p.add_argument("--model", choices=["siglip", "vlm", "frontier"], help="override the request's model")
    p.add_argument("--choice-method", choices=["independent", "letter"], help="override options.choice_method")
    p.add_argument("--score-method", choices=["auto", "statements", "digits", "fast2", "ens4d", "jsondigits"], help="override options.score_method")
    p.add_argument("--calibrated", action=argparse.BooleanOptionalAction, default=None, help="override options.calibrated")
    p.add_argument("--no-prefix-cache", action="store_true", help="VLM: use the reference path (every statement a full prompt)")
    p.add_argument("--confirm-spend", action="store_true", help="allow --model frontier: sends the image to a paid API")
    p.add_argument("--prefix-cache", action="store_true", help="VLM: opt in to the prefix-cached path (about 3.7x faster; see STATUS.md M4)")
    p.set_defaults(func=_cmd_decide)

    p = sub.add_parser("ask", help="one question about one image: yes/no by default, --options for pick-one, --levels for a rating")
    p.add_argument("image", help="path to the image")
    p.add_argument("question", help='the question, e.g. "Is there a dog?"')
    p.add_argument("--options", nargs="+", help="pick one of these")
    p.add_argument("--levels", nargs="+", help="rate on these ordered levels, lowest first")
    p.add_argument("--model", choices=["vlm", "siglip"], default="vlm")
    p.add_argument("--full", action="store_true", help="print the whole response, not only the answer")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("score", help="rate one image on a rubric (Glance elicitation; uses your `glance fit` calibration if there is one)")
    p.add_argument("image", help="path to the image")
    _rubric_args(p)
    p.add_argument("--full", action="store_true", help="print the whole response, not only the answer")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("fit", help="fit a calibration for one rating rubric from a few dozen labeled images (no training)")
    p.add_argument("--data", required=True, help="folder with one sub-folder per level (0/, 1/, ...), or a JSONL/CSV with image,level")
    _rubric_args(p)
    p.add_argument("--name", help="a label stored in the calibration file")
    p.add_argument("--unlabeled", action="store_true", help="no labels: --data is any folder of images from your domain (16 or more); "
                   "removes the readout's systematic bias only (weaker than a labeled fit)")
    p.set_defaults(func=_cmd_fit)

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

    p = sub.add_parser("baseline", help="add the frontier baseline to a finished eval run; asks for the API key in the terminal")
    p.add_argument("--run", metavar="RUN_ID", help="eval run to add the baseline to (default: the largest finished run)")
    p.add_argument("--model", help="LiteLLM model id (default: asked interactively, or FRONTIER_MODEL)")
    p.add_argument("--baseline-n", type=int, help="items per suite, test split only (default: 300)")
    p.add_argument("--allow-upload-gold", action="store_true", help="allow human_gold images to be sent to the provider")
    p.add_argument("--confirm-spend", action="store_true", help="skip the interactive y/N after the cost estimate")
    p.add_argument("--env-file", metavar="PATH", help="read ONLY the provider key this model needs from a dotenv file you name "
                   "(needs --model); the value is never printed or stored")
    p.add_argument("--estimate-only", action="store_true", help="print the plan and the cost estimate, make no API call, and stop")
    p.set_defaults(func=_cmd_baseline)

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

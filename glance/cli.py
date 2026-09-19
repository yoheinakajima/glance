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
    status, payload = Engine(cfg, source="cli").decide_json(body)
    print(json.dumps(payload, indent=2))
    return 0 if status == 200 else 1


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
    p.set_defaults(func=_cmd_decide)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

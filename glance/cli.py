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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glance", description="Image decision harness v0")
    parser.add_argument("--config", default=None, help="path to a config yaml (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="detect device, pick the model tier, write logs/doctor.json")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

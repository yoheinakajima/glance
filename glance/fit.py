"""`glance fit`: turn a few dozen labeled images into a calibration for one rating rubric.

You bring a frozen VLM you already have and a rubric (a question plus ordered level descriptions). Glance reads the
model four times per image (`glance/rating.py`), fits a small affine map from those logits to your levels, reports how
well it cross-validates at your sample size, and saves a JSON file of a few hundred numbers. No weights are trained.

Labels come as a folder with one sub-folder per level (`0/`, `1/`, ... or `0_sharp/`, `1_soft/`, ...), or as a JSONL /
CSV file with `image` and `level` columns (level = index into the rubric's criteria, lowest first).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import rating
from .pipeline import Engine

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_rubric(path: str | Path) -> tuple[str, list[str]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("instructions"), str) or not isinstance(data.get("criteria"), list):
        raise ValueError('a rubric file is {"instructions": "...", "criteria": ["lowest level", ..., "highest level"]}')
    return data["instructions"], [str(c) for c in data["criteria"]]


def read_labels(path: str | Path) -> list[tuple[str, int]]:
    """[(image path, level index)] from a folder-per-level directory, a JSONL file or a CSV file."""
    path = Path(path)
    if path.is_dir():
        out = []
        for sub in sorted(p for p in path.iterdir() if p.is_dir()):
            head = sub.name.split("_", 1)[0]
            if not head.isdigit():
                raise ValueError(f"label folder `{sub.name}` must start with its level index, e.g. `0` or `0_sharp`")
            out += [(str(f), int(head)) for f in sorted(sub.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES]
        return out
    base = path.parent
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = []
    for row in rows:
        image = Path(str(row["image"]))
        out.append((str(image if image.is_absolute() or image.exists() else base / image), int(row["level"])))
    return out


def collect_features(engine: Engine, instructions: str, criteria: list[str], examples: Iterable[tuple[str, int]],
                     method: str = "ens4d", model: str = "vlm", image_id: str = "img0",
                     progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One uncalibrated request per labeled image; returns (member logits [n, F], levels [n])."""
    examples = list(examples)
    features, levels = [], []
    for i, (image, level) in enumerate(examples, 1):
        body = {
            "model": model, "state": {"images": [{"id": image_id, "path": image}]},
            "questions": {"q": {"type": "score", "instructions": instructions, "criteria": criteria}},
            "options": {"score_method": method, "calibrated": False},
        }
        trace = engine.decide(body, source="fit")
        features.append(trace.scoring.scores["q"].features)
        levels.append(int(level))
        if progress:
            progress(i, len(examples))
    return np.asarray(features, dtype=np.float64), np.asarray(levels, dtype=int)


def fit_rubric(engine: Engine, instructions: str, criteria: list[str], examples: Iterable[tuple[str, int]],
               method: str = "ens4d", model: str = "vlm", name: str | None = None, save: bool = True,
               progress: Callable[[int, int], None] | None = None) -> tuple[rating.RatingCalibration, Path | None]:
    if method not in rating.MEMBERS:
        raise ValueError(f"method must be one of {sorted(rating.MEMBERS)}")
    features, levels = collect_features(engine, instructions, criteria, examples, method, model, progress=progress)
    from .schema import ScoreQuestion

    question = ScoreQuestion(type="score", instructions=instructions, criteria=criteria)
    key = engine.rating_key(engine.backend(model), method, question)
    cal = rating.build_calibration(key, features, levels, name=name, source=f"glance fit, {len(levels)} labeled images")
    path = rating.save_calibration(engine.cfg.path("calibration") / "ratings", cal) if save else None
    return cal, path


def describe(cal: rating.RatingCalibration, path: Path | None) -> str:
    k = len(cal.key.criteria)
    lines = [f"{cal.version}: {cal.key.score_method} calibration for a {k}-level rubric, fit on {cal.n} labeled images "
             f"(per level: {cal.n_per_level})"]
    if cal.cv:
        cv = cal.cv
        lines.append(f"  {cv['folds']}-fold cross-validation on those images: accuracy {cv['accuracy']:.3f}, within one level "
                     f"{cv['within_1']:.3f}, mean abs. error {cv['mae']:.2f} levels, ECE {cv['ece']:.3f} "
                     f"(a perfectly calibrated model would measure about {cv['ece_floor']:.3f} at this sample size)")
    if min(cal.n_per_level) < 4:
        lines.append("  note: fewer than 4 images for some level; the lab's curve flattens at about 8 per level (32 for 4 levels)")
    if path:
        lines.append(f"  saved to {path}")
    lines.append('  use it: send the same instructions and criteria with "options": {"calibrated": "auto"}')
    return "\n".join(lines)


def main_fit(args: Any) -> int:
    from .cli import _config_overrides
    from .config import load_config

    cfg = load_config(args.config, _config_overrides(args))
    if args.rubric:
        instructions, criteria = read_rubric(args.rubric)
    elif args.instructions and args.criteria:
        instructions, criteria = args.instructions, list(args.criteria)
    else:
        print("give the rubric as --rubric file.json, or as --instructions \"...\" --criteria \"lowest\" ... \"highest\"", file=sys.stderr)
        return 2
    try:
        examples = read_labels(args.data)
    except (OSError, ValueError, KeyError) as exc:
        print(f"could not read labels from {args.data}: {exc}", file=sys.stderr)
        return 2
    if not examples:
        print(f"no labeled images found in {args.data}", file=sys.stderr)
        return 2

    def progress(i: int, n: int) -> None:
        if i == n or i % 10 == 0:
            print(f"  read {i}/{n} images", file=sys.stderr, flush=True)

    engine = Engine(cfg, source="fit")
    try:
        cal, path = fit_rubric(engine, instructions, criteria, examples, method=args.method, name=args.name, progress=progress)
    except ValueError as exc:
        print(f"cannot fit: {exc}", file=sys.stderr)
        return 2
    print(describe(cal, path))
    return 0

"""Shared suite machinery: seeded order, 50/50 split, image export, committed manifests.

A suite loader lists every candidate item in a deterministic order. This module shuffles that list with the eval
seed, keeps the first `manifest_n`, alternates calibration/test down the shuffled order (so the first n items of
any trimmed run are still 50/50), exports each image to a local file, and checks the result against the committed
manifest `glance/evals/manifests/<suite>.jsonl` (item id, image sha256, split).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...config import Config
from ...logging_utils import read_jsonl

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"


@dataclass
class RawItem:
    """A candidate item before export. `write_image(path)` saves the image file when the item is selected."""

    item_id: str
    question: dict[str, Any]
    label: Any  # noul: bool; choice: option key; score: level index
    write_image: Callable[[Path], None]
    ext: str = ".jpg"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalItem:
    suite: str
    item_id: str
    split: str  # "calibration" | "test"
    image_path: str
    image_sha256: str
    question: dict[str, Any]
    label: Any
    meta: dict[str, Any] = field(default_factory=dict)


class SuiteSkipped(Exception):
    """The suite cannot run here (license unclear, no data). The report says so."""


@dataclass
class SuiteInfo:
    name: str
    qtype: str  # "noul" | "choice" | "score" | "any"
    source: str
    license: str
    backends: tuple[str, ...] = ("siglip", "vlm", "frontier")
    private: bool = False  # human_gold: manifest stays out of git, baseline needs --allow-upload-gold


def _safe_name(item_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in item_id)


def split_for(index: int) -> str:
    return "calibration" if index % 2 == 0 else "test"


def materialize(
    cfg: Config,
    info: SuiteInfo,
    raw_items: list[RawItem],
    n: int,
    manifest_path: Path | None = None,
) -> list[EvalItem]:
    """Seeded shuffle, export the first max(n, manifest_n) images, verify or write the manifest, return n items."""
    order = list(range(len(raw_items)))
    random.Random(cfg.eval.seed).shuffle(order)
    keep = min(len(order), max(n, cfg.eval.manifest_n))
    out_dir = cfg.path("eval_images") / info.name
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[EvalItem] = []
    for position, raw_index in enumerate(order[:keep]):
        raw = raw_items[raw_index]
        path = out_dir / (_safe_name(raw.item_id) + raw.ext)
        if not path.exists():
            raw.write_image(path)
        items.append(
            EvalItem(
                suite=info.name, item_id=raw.item_id, split=split_for(position), image_path=str(path),
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), question=raw.question, label=raw.label,
                meta=raw.meta,
            )
        )

    manifest_path = manifest_path or MANIFEST_DIR / f"{info.name}.jsonl"
    rows = [{"item_id": it.item_id, "image_sha256": it.image_sha256, "split": it.split} for it in items]
    if manifest_path.exists():
        committed = read_jsonl(manifest_path)
        shared = min(len(committed), len(rows))
        if committed[:shared] != rows[:shared]:
            first = next(i for i in range(shared) if committed[i] != rows[i])
            raise RuntimeError(
                f"suite `{info.name}` no longer matches its manifest at row {first}: "
                f"{committed[first]} != {rows[first]}. The source data changed; delete the manifest only on purpose."
            )
        if len(rows) > len(committed):
            _write_manifest(manifest_path, rows)
    else:
        _write_manifest(manifest_path, rows)
    return items[:n]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def item_to_request(item: EvalItem, model: str, choice_method: str = "independent") -> dict[str, Any]:
    return {
        "model": model,
        "state": {"images": [{"id": "img0", "path": item.image_path}]},
        "questions": {"q": item.question},
        "options": {"choice_method": choice_method, "calibrated": False},
    }


def item_record(item: EvalItem) -> dict[str, Any]:
    return asdict(item)

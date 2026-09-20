"""The score lab's five degradation scales as opt-in eval suites (`ladder_blur`, `ladder_noise`, `ladder_jpeg`,
`ladder_exposure`, `ladder_resolution`), so the v0 harness and the frontier baseline can be run on exactly the items
the lab used. Items, levels and splits come from `lab/manifests/<scale>.jsonl`; images are rebuilt from the pinned
Oxford-IIIT Pet parquet by `python -m glance.lab.ladders` if `.cache/lab_images/` is missing.
"""

from __future__ import annotations

from types import SimpleNamespace

from ...config import PROJECT_ROOT, Config
from ...logging_utils import read_jsonl
from .base import EvalItem, SuiteInfo, SuiteSkipped

SCALES = ("blur", "noise", "jpeg", "exposure", "resolution")


def _module(scale: str) -> SimpleNamespace:
    info = SuiteInfo(
        name=f"ladder_{scale}", qtype="score",
        source=f"synthetic `{scale}` ladder on Oxford-IIIT Pet test images (glance/lab/ladders.py)",
        license="CC BY-SA 4.0 (derived)", backends=("siglip", "vlm", "frontier"),
    )

    def build(cfg: Config, n: int) -> list[EvalItem]:
        from ...lab.ladders import LADDERS

        rows = read_jsonl(PROJECT_ROOT / "lab" / "manifests" / f"{scale}.jsonl")
        if not rows:
            raise SuiteSkipped(f"no manifest for ladder `{scale}`")
        if not (PROJECT_ROOT / rows[0]["path"]).is_file():
            raise SuiteSkipped(f"ladder images are missing; run `uv run python -m glance.lab.ladders` to rebuild them")
        question = {"type": "score", "instructions": LADDERS[scale]["instructions"], "criteria": list(LADDERS[scale]["levels"])}
        return [
            EvalItem(suite=info.name, item_id=r["item_id"], split=r["split"], image_path=str(PROJECT_ROOT / r["path"]),
                     image_sha256=r["sha256"], question=question, label=int(r["level"]), meta={"source_image_id": r["source_image_id"]})
            for r in rows[:n]
        ]

    return SimpleNamespace(INFO=info, build=build)


MODULES = {f"ladder_{scale}": _module(scale) for scale in SCALES}

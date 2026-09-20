"""Copy a finished run from runs/ (gitignored) into results/ (committed), so every number in the docs is backed by a
file in the repository.

uv run python tools/snapshot_run.py <run_id> <results/sub/dir>

Text artifacts are copied with absolute local paths rewritten to project-relative ones. predictions.jsonl and
errors.jsonl are gzipped. No images are copied: predictions reference eval images by relative path, and those are
rebuilt from the pinned sources by the suite loaders (see DATASETS.md).
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ("report.md", "summary.txt", "metrics.json", "config.yaml", "env.json", "extras.json")


def relativize(text: str) -> str:
    return text.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")


def snapshot(run_id: str, dest: str) -> Path:
    src = ROOT / "runs" / run_id
    out = ROOT / dest
    if not src.is_dir():
        raise SystemExit(f"no run at {src}")
    out.mkdir(parents=True, exist_ok=True)
    for name in TEXT:
        if (src / name).exists():
            (out / name).write_text(relativize((src / name).read_text()))
    for name in ("predictions.jsonl", "errors.jsonl"):
        if (src / name).exists():
            with gzip.open(out / (name + ".gz"), "wt", compresslevel=9) as f:
                f.write(relativize((src / name).read_text()))
    for sub in ("plots", "calibration"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)
    (out / "SOURCE.txt").write_text(f"snapshot of runs/{run_id}\n")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(snapshot(sys.argv[1], sys.argv[2]))

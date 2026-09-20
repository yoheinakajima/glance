"""Builds a small `glance fit` demo from the license-clean `distort25` benchmark (run `python -m glance.lab.distort25`
first): a rubric file, a folder of labeled images per level, and a few more images to try the fitted rubric on.

Only CALIBRATION-split images are used, so the benchmark's test split stays untouched by demos.

uv run python tools/make_fit_demo.py --distortion jpeg --per-level 8
uv run glance fit --data demo/jpeg/labels --rubric demo/jpeg/rubric.json --prefix-cache
uv run glance score demo/jpeg/try/<file> --rubric demo/jpeg/rubric.json --prefix-cache
"""
import argparse
import json
import pathlib
import shutil

from glance.lab import distort25
from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--distortion", default="jpeg", choices=sorted(distort25.SCALES))
parser.add_argument("--per-level", type=int, default=8)
parser.add_argument("--try-per-level", type=int, default=2)
args = parser.parse_args()

scale = distort25.SCALES[args.distortion]
items = [i for i in read_jsonl(ROOT / "lab" / "manifests_distort25" / f"{args.distortion}.jsonl") if i["split"] == "calibration"]
out = ROOT / "demo" / args.distortion
shutil.rmtree(out, ignore_errors=True)
(out / "try").mkdir(parents=True)
(out / "rubric.json").write_text(json.dumps({"instructions": scale["instructions"], "criteria": scale["levels"]}, indent=2) + "\n")
for level in range(len(scale["levels"])):
    of_level = [i for i in items if i["level"] == level]
    folder = out / "labels" / f"{level}_{scale['levels'][level].lower().replace(' ', '_')}"
    folder.mkdir(parents=True)
    for item in of_level[: args.per_level]:
        shutil.copy(ROOT / item["path"], folder / pathlib.Path(item["path"]).name)
    for item in of_level[args.per_level: args.per_level + args.try_per_level]:
        shutil.copy(ROOT / item["path"], out / "try" / f"level{level}_{pathlib.Path(item['path']).name}")
print(f"wrote {out.relative_to(ROOT)}/: rubric.json, labels/ ({args.per_level} per level), try/ ({args.try_per_level} per level)")

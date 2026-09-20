"""E14 (lab/NOTES.md entry 39): content-free "null" images and one manifest per lab scale, so the ordinary collector can
read the level logits the model gives when there is nothing to rate (its prior for that rubric and readout).

uv run python tools/make_null_images.py
uv run python -m glance.lab.collect --bench ladders_null --out lab/runs/lab_null.jsonl --methods "digits,zoom_digits,digitsrev,zoom_digitsrev" --prefix-cache
"""
import hashlib
import json
import pathlib

import numpy as np
from PIL import Image

from glance.lab.ladders import BASE_SIDE, LADDERS

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / ".cache" / "lab_images_null"
OUT.mkdir(parents=True, exist_ok=True)
images = {"grey": np.full((BASE_SIDE, BASE_SIDE, 3), 128, np.uint8), "black": np.zeros((BASE_SIDE, BASE_SIDE, 3), np.uint8),
          "white": np.full((BASE_SIDE, BASE_SIDE, 3), 255, np.uint8)}
for seed in (1, 2, 3):
    noise = np.random.default_rng(seed).normal(128, 50, (BASE_SIDE, BASE_SIDE, 3))
    images[f"noise{seed}"] = np.clip(noise, 0, 255).astype(np.uint8)
rows = []
for name, arr in images.items():
    path = OUT / f"{name}.png"
    Image.fromarray(arr).save(path)
    rows.append({"item_id": f"null_{name}", "level": -1, "split": "null", "path": str(path.relative_to(ROOT)),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(ROOT / "lab" / "manifests_null").mkdir(exist_ok=True)
for ladder in LADDERS:
    (ROOT / "lab" / "manifests_null" / f"{ladder}.jsonl").write_text("".join(json.dumps({**r, "ladder": ladder}) + "\n" for r in rows))
print(len(rows), "null images,", len(LADDERS), "manifests -> lab/manifests_null/")

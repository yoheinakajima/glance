# AGENTS.md: operating guide for coding agents

Glance answers typed questions about images from a frozen open vision-language model running locally. It reads the
answer from logits (no generation) and returns probabilities. It is a harness, not a model. Python 3.11, managed with `uv`.

## Install and check

```bash
uv sync                      # installs everything; weights (about 10 GB) download on first model use into ./.cache/hf
uv run glance doctor --json  # device, selected model tier, free disk; run this first and read `warnings`
uv run pytest -q             # CPU-only tests, about 80 s, no model download
```

## Ask questions

CLI, one question: `uv run glance ask IMAGE "QUESTION"` (yes/no), add `--options a b c` (pick one) or `--levels low ... high`
(rating). Python:

```python
from glance import Glance
g = Glance()                                   # model="vlm" (default) or "siglip" (tiny dual encoder, pick-one only in practice)
# Glance(model_id="org/any-hf-vlm", revision="<commit>") swaps in another Hugging Face VLM (checked end to end on SmolVLM2-2.2B only; CLI: glance --model-id ... ask ...)
g.noul(image, "Is there a person in `img0`?")                      # -> {"type": "noul", "noul": float in [0, 1], "raw": float}
g.choice(image, "What is `img0`?", {"receipt": "an itemized proof of purchase", "invoice": "a bill", "other": None})
                                                                    # -> {"choice": key, "probabilities": {key: p}, "confidence": float}
g.score(image, "How blurry is `img0`?", ["Sharp", "Slightly soft", "Blurry", "Very blurry"])
                                                                    # -> {"score": expected level, "probabilities": {"0": p, ...}, "confidence": float, "calibration": id or None}
g.ask(image, {"name": {"type": ..., "instructions": ..., "criteria": ...}, ...})   # many questions, one image: cheapest per answer
```

HTTP: `uv run glance serve --preload vlm --prefix-cache`, then `POST http://127.0.0.1:8077/v1/decide` with
`{"model": "vlm", "state": {"images": [{"id": "img0", "path": "..."}]}, "questions": {...}, "options": {...}}`.
Sample requests: `samples/*.json`. Schema: `glance/schema.py`; narrative spec: `HANDOFF.md` section 5.
Experimental Apple-Silicon runtime: `uv sync --extra mlx`, then `uv run glance serve --backend mlx --preload vlm`; the request
model remains `vlm`, PyTorch stays the default, and unsupported machines fail explicitly rather than falling back.

Rules that matter:
- Refer to images as `` `img0` ``, `` `img1` `` inside `instructions`. Option descriptions (`criteria` values) help pick-one.
- Put every question about the same image into ONE request; the image is encoded once.
- `options.calibrated`: `"auto"` (default in the CLI helpers) applies what exists and warns; `true` returns HTTP 409
  `calibration_mismatch` when a rating rubric has no fitted calibration. A calibration is keyed to the exact rubric text.
- Ratings: `score` is an expectation over levels; use `probabilities` / `confidence` for thresholds. For exact levels on a
  user's own scale run `glance fit` first (below). Zero-shot, trust the ORDER, not the exact level.
- Never call `--model frontier` or `glance baseline` on your own: they send images to paid APIs and need the user's key
  typed into their terminal. Never read `.env` files for keys.

## Fit a rating rubric (optional, no training)

```bash
uv run glance fit --unlabeled --data some_folder_of_the_users_images/ --rubric rubric.json     # 16+ images, no labels
uv run glance fit --data labels/ --rubric rubric.json                                          # labels/0/*.jpg, labels/1/*.jpg, ... about 32 images
```

`rubric.json`: `{"instructions": "How ... is `img0`?", "criteria": ["lowest", ..., "highest"]}`. Output goes to
`calibration/ratings/` and is picked up automatically by later calls with the same rubric text.

## What it is good and bad at (say this to users)

Good: yes/no and pick-one about ordinary photos (on photos no model has seen: level with the best hosted models on pick-one, about two points behind the best on yes/no); ordering
images by a rubric; many questions per image; private or offline use; probabilities. Not good: best-possible zero-shot
exact ratings (a cheap hosted model, Gemini 3.1 Flash-Lite, is about 9 points better); fine image-quality levels
(KADID-10k); saving money against the cheapest hosted models on ratings. Numbers and caveats: `docs/CLAIMS.md`.

## Repo map and conventions

`glance/` package (`api.py` Python API, `cli.py`, `pipeline.py` engine, `scorer.py` readouts, `rating.py` rating fits,
`backends/` models, `evals/` suites and baselines, `lab/` research collectors). `tools/` report generators: every number
in `docs/paper/RESULTS_*.md`, `results/lab/` and `site/` is generated from result files; do not edit those by hand, re-run
the tool. `lab/NOTES.md` is a pre-registration notebook: a new experiment gets its expectation committed BEFORE any code
or data. Only Apache-2.0 or MIT model weights (`MODELS.md`); dataset licenses in `DATASETS.md`. KADID-10k data and hosted
model outputs are evaluation-only and are never committed.

# glance

**Glance is how you ask a frozen vision-language model for a score. It is not a VLM.**

You send image(s) plus typed questions (`noul`: is this true, `choice`: which one, `score`: where on this ordered
rubric) and get probability distributions back, read from the logits of single forward passes of an open model you
already have (Qwen3-VL-4B by default). No text is generated. The VLM stays frozen: no VLM weights are updated or shipped; what is fit, from a few dozen labeled images, is a small readout on top of its logits. The request and
response shapes follow TypeSafe's Jev, a hosted text-only model, so integrations look familiar. That is where the
resemblance ends: Glance is a readout-and-calibration recipe on someone else's frozen weights, the calibration is
yours to fit, and it makes no speed or cost claim against any hosted model.

What is new here is the *elicitation* for `score`, measured in `docs/paper/RESULTS_LAB.md`:

| Qwen3-VL-4B, five 4-level image-quality rubrics, 500 held-out images each | mean accuracy |
| --- | --- |
| v0 readout (one yes/no statement per level), single temperature | 0.500 |
| same readout, with a calibration fit on labeled examples | 0.810 |
| **+ Glance** (`ens4d`): digit readout, scale forward and reversed, with and without a magnified crop, per-rubric matrix calibration | **0.867** |

About 32 labeled images per rubric are enough for the calibration, and it does **not** transfer from one rubric to
another, so the verb that matters is `glance fit`. Measured on one 4B model and five synthetic single-factor scales
with hand-written level texts. The harder test (25 distortion types, 5 levels, one generic wording, KADID-10k human
scores) is running on this branch; early numbers are clearly lower and will be published as they are
(`docs/BRIEFING.md`, section 7). One rating of a fresh image takes about 1.1 s on an Apple-silicon laptop, slower
than the 0.44 s of the naive readout; five ratings of the same image cost about 0.6 s each, 25 about 0.34 s each.
An earlier 609 ms figure was wrong and is corrected (`lab/NOTES.md`, entry 16).

- `HANDOFF.md` is the original spec. `STATUS.md` is the build log, with every decision and deviation.
- `docs/` is written for a paper: methods, results, reproduction, related work, research log. `lab/NOTES.md` is the
  lab notebook, with hypotheses registered before each experiment and two errata.
- `MODELS.md` and `DATASETS.md` record licenses, pins and check dates.

## Setup

```bash
uv sync
uv run glance doctor --json      # detects the device and picks the model tier
```

Weights download to `./.cache/hf` on first use (about 10.5 GB for SigLIP2 plus Qwen3-VL-4B on this tier).
Afterwards everything runs with `HF_HUB_OFFLINE=1`.

## Rate an image, and fit your own rubric

```bash
# one rating, from the command line (uses your calibration for this rubric if you have fit one)
uv run glance score photo.jpg --prefix-cache \
  --instructions "How blurry is `img0`?" \
  --criteria "Sharp" "Slightly soft" "Blurry" "Very blurry"

# fit a calibration for YOUR rubric from a few dozen labeled images: labels/0/*.jpg, labels/1/*.jpg, ...
uv run glance fit --data labels/ --rubric rubric.json --prefix-cache
```

`rubric.json` is `{"instructions": "How ... is `img0`?", "criteria": ["lowest level", ..., "highest level"]}`. `fit`
reads the model four times per image, fits a K x 4K affine map on those logits (nothing in the model changes), prints
its cross-validated accuracy, error and calibration at your sample size, and writes a few hundred numbers to
`calibration/ratings/<hash>.json`. A calibration is tied to the exact rubric text, the model revision and the image
token budget. Calibrations for the five lab rubrics ship in `glance/assets/ratings/`.

```python
from glance import Glance

g = Glance()                                   # loads the local VLM on first use
g.score("photo.jpg", "How blurry is `img0`?", ["Sharp", "Slightly soft", "Blurry", "Very blurry"])
# {'score': 0.31, 'probabilities': {'0': 0.72, '1': 0.26, ...}, 'confidence': 0.55, 'calibration': 'rate_e098aa', ...}  (values illustrative)
g.noul("photo.jpg", "Is there a dog in `img0`?")
g.choice("photo.jpg", "What is in `img0`?", ["dog", "cat", "other"])
g.ask("photo.jpg", {"blur": {...}, "noise": {...}, "usable": {...}})   # many questions, the image is prefilled once per view
g.fit("How blurry is `img0`?", ["Sharp", "Slightly soft", "Blurry", "Very blurry"], "labels/")
```

Three rating modes (`options.score_method`, or `--method`): `ens4d` (default; 4 passes, 0.867 on the lab scales, about
1.1 s for one rating), `fast2` (2 passes, no magnified crop; 0.833, and 133 to 240 ms per rating when a request carries
5 to 25 rubrics; weak on compression artifacts, 0.674), `digits` (1 pass; 0.814).

`score` is the expected level, `probabilities` the distribution over levels, `confidence` is 1 minus the normalized
entropy. Use the expectation and the confidence, not only the top level: almost every error is an adjacent level.

## Use the harness

```bash
uv run glance decide samples/receipt.json --model vlm
uv run glance decide samples/receipt.json --model siglip
uv run glance decide samples/receipt.json --model vlm --choice-method letter
uv run glance decide samples/receipt.json --model vlm --calibrated      # needs fitted params in calibration/
uv run glance decide samples/receipt.json --model vlm --score-method statements   # the v0 `score` method

uv run glance serve --preload vlm --prefix-cache                         # 127.0.0.1:8077
curl -s localhost:8077/v1/decide -H 'content-type: application/json' -d @samples/dog.json
curl -s localhost:8077/v1/models
curl -s localhost:8077/healthz
```

The request and response shapes are in `HANDOFF.md` section 5, with three additive extensions (`STATUS.md`, "API
extension"): `options.score_method` (`auto` | `statements` | `digits` | `ens4d`; `auto` is `ens4d` on the VLM),
`options.calibrated` also accepts `"auto"` (calibrate what has fitted parameters, warn about the rest; `true` still
fails with 409 when something is missing), and `score` answers carry `method` and `calibration`. Every call appends
one line to `logs/calls/YYYY-MM-DD.jsonl` (inputs, per-statement logits, raw and calibrated probabilities, timing,
versions).

## Evaluate

```bash
uv run glance eval                                    # all suites, siglip + vlm, 4 h budget
uv run glance eval --suite pope --n 200 --model vlm
uv run glance eval --suite pets37 --n 50              # both choice methods
uv run glance calibrate --run <run_id>                # writes calibration/<key-hash>.json, no model re-run
uv run glance eval --resume <run_id>                  # continue an interrupted run
```

Each run writes `runs/<run_id>/`: `config.yaml`, `env.json`, `predictions.jsonl`, `errors.jsonl`, `metrics.json`,
`report.md` (opens with the go/no-go table), `summary.txt`, `plots/`, `calibration/`.

### Frontier baseline (optional, paid)

No `.env` needed. Paste this into a terminal:

```bash
uv run glance baseline
```

It finds the finished full eval, asks which model to use (Enter accepts `anthropic/claude-opus-5`; OpenAI, Gemini or
any LiteLLM model id also work), and asks for the API key with hidden input. The key stays in memory for that one
process: it is never written to disk, to a log, or to the shell history. One test call checks the key, then the cost
estimate is shown and nothing more is sent until you answer `y`. When it finishes, the run's `report.md` is rebuilt
with the accuracy-gap and selective-accuracy rows filled in.

The baseline runs on the test split only, capped by `--baseline-n` (default 300 per suite), never on `human_gold`
without `--allow-upload-gold`, and its picks are never stored: only whether each one was right. If you would rather
use environment variables, set `FRONTIER_MODEL` and the provider's key (names in `.env.example`) and the command skips
the prompts; `glance eval --confirm-spend` also still works.

### Your own images (`human_gold`)

Put one JSON object per line in `gold/human_gold.jsonl` (gitignored, never leaves the machine):

```json
{"image": "gold/img_0001.jpg", "question": {"type": "choice", "instructions": "What kind of document is `img0`?", "criteria": {"receipt": "...", "invoice": "...", "other": null}}, "label": "receipt", "annotators": {"a1": "receipt", "a2": "invoice"}}
```

`label` is `true`/`false` for noul, an option key for choice, a level index for score.

## Tests

```bash
uv run pytest                                   # unit tests, no weights needed
GLANCE_TEST_MODELS=1 uv run pytest -s           # adds the checks that load the real models
```

# glance

Image decision harness v0. Takes image(s) plus typed questions (`noul`, `choice`, `score`) and returns probability
distributions read from single forward passes of an open VLM or a dual encoder. No text generation, no training.

- `HANDOFF.md` is the spec. `STATUS.md` is the build log, with every decision and deviation.
- `MODELS.md` and `DATASETS.md` record licenses, pins and check dates.

## Setup

```bash
uv sync
uv run glance doctor --json      # detects the device and picks the model tier
```

Weights download to `./.cache/hf` on first use (about 10.5 GB for SigLIP2 plus Qwen3-VL-4B on this tier).
Afterwards everything runs with `HF_HUB_OFFLINE=1`.

## Use

```bash
uv run glance decide samples/receipt.json --model vlm
uv run glance decide samples/receipt.json --model siglip
uv run glance decide samples/receipt.json --model vlm --choice-method letter
uv run glance decide samples/receipt.json --model vlm --calibrated      # needs fitted params in calibration/

uv run glance serve --preload vlm                                        # 127.0.0.1:8077
curl -s localhost:8077/v1/decide -H 'content-type: application/json' -d @samples/dog.json
curl -s localhost:8077/v1/models
curl -s localhost:8077/healthz
```

The request and response shapes are in `HANDOFF.md` section 5. Every call appends one line to
`logs/calls/YYYY-MM-DD.jsonl` (inputs, per-statement logits, raw and calibrated probabilities, timing, versions).

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

Put `FRONTIER_MODEL` (a LiteLLM model id) and its API key in `.env` (see `.env.example`), then add
`--confirm-spend` to `glance eval`. The runner prints a cost estimate first. The baseline runs on the test split
only, capped by `--baseline-n` (default 300 per suite), never on `human_gold` without `--allow-upload-gold`, and its
picks are never written to disk: only whether each one was right.

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

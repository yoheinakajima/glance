# glance

**Ask an open vision-language model typed questions about an image and get probabilities back, on your own machine.**
Yes/no, pick-one and ratings are READ from the logits of one forward pass of a frozen open model (Qwen3-VL-4B by
default, Apache-2.0). Nothing is generated, nothing is trained, no image leaves the machine. Glance is a calibration and
measurement harness around that readout. It is not a model.

Project page with every result: https://glance.yohei.me · Coding agents: read [`AGENTS.md`](AGENTS.md).

## Try it

```bash
git clone https://github.com/yoheinakajima/glance && cd glance && uv sync     # Python 3.11 and uv; about 10 GB of open weights download on first use
uv run glance doctor                                     # picks the model for your hardware (Apple silicon or CUDA; CPU works, slowly)

uv run glance ask photo.jpg "Is there a dog?"                                        # yes/no   -> {"noul": 0.98, ...}
uv run glance ask photo.jpg "What is the main subject?" --options dog cat car       # pick one -> {"choice": "dog", "probabilities": {...}, "confidence": ...}
uv run glance ask photo.jpg "How blurry is the photo?" --levels Sharp "Slightly soft" Blurry "Very blurry"   # rating -> {"score": ..., "probabilities": {...}}
```

```python
from glance import Glance

g = Glance()                                                        # loads the local model on first use
g.noul("photo.jpg", "Is there a dog in `img0`?")                    # {'noul': 0.98, ...}
g.choice("photo.jpg", "What is in `img0`?", ["dog", "cat", "other"])
g.score("photo.jpg", "How blurry is `img0`?", ["Sharp", "Slightly soft", "Blurry", "Very blurry"])
g.ask("photo.jpg", {"dog": {"type": "noul", "instructions": "Is there a dog in `img0`?"},           # many questions, one image:
                    "blur": {"type": "score", "instructions": "How blurry is `img0`?",              # the image is read once and
                             "criteria": ["Sharp", "Slightly soft", "Blurry", "Very blurry"]}})     # every extra question is cheap
```

```bash
uv run glance serve --preload vlm --prefix-cache          # a local HTTP server on 127.0.0.1:8077, nothing leaves the machine
curl -s localhost:8077/v1/decide -H 'content-type: application/json' -d @samples/dog.json
```

Images are referred to as `` `img0` `` (then `` `img1` ``, ...) inside the question text. A yes/no answer is `noul`, the
probability that the statement is true. A pick-one answer has `choice`, `probabilities` and `confidence`. A rating has
`score` (the expected level, 0 to K-1), `probabilities` over the levels and `confidence`; use the expectation and the
confidence, not only the top level, because almost every rating error is an adjacent level. The full request and
response shapes are in `HANDOFF.md` section 5; every call is logged to `logs/calls/`.

## How much setup does a question need?

| Question | Setup | What to expect |
| --- | --- | --- |
| yes/no, pick-one | none | level with hosted models on photos none of them has seen (table below) |
| rating, you need the ORDER (sort, threshold, flag the worst) | none | within one level on 0.99 of images, rank agreement 0.93 with the true level |
| rating, you need the EXACT level of your own scale | `glance fit --unlabeled --data folder_of_your_images/` (16 or more images, no labels) | removes the model's constant offset on your rubric: 0.57 -> 0.70 exact with the shipped read (0.67 -> 0.76 with the one-pass read below) |
| rating, best accuracy and calibrated probabilities | `glance fit --data labels/ --rubric rubric.json` (about 32 labeled images, folders `labels/0/`, `labels/1/`, ...) | 0.86 exact, ECE about 0.03; fits are per rubric and do not transfer |

`rubric.json` is `{"instructions": "How ... is `img0`?", "criteria": ["lowest level", ..., "highest level"]}`. A fit changes
nothing in the model; it writes a few hundred numbers to `calibration/ratings/`, tied to the exact rubric text and model
revision. Rating modes (`--method`): `ens4d` (4 passes, the one to fit), `fast2` (2 passes), `digits` (1 pass).

## How good is it? (zero-shot, same items for every row, 95% intervals on the project page)

| System | yes/no, fresh photos | pick-one, fresh photos | rating, exact level | seconds per answer | US dollars per 1,000 answers |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-VL-4B read with Glance, on a laptop** | **0.931** | **0.862** | **0.669** | **1.1 (yes/no on a full-size photo; 0.33 on a small image)** | **0.07 to 0.32** (rented GPU; electricity only: about 0.01) |
| the same model writing JSON | 0.931 | 0.892 | 0.672 | 1.6 (0.78 on a small image) | 0.13 to 0.42 |
| Gemini 3.1 Flash-Lite (cheapest Google) | 0.954 | 0.908 | 0.763 | 1.6 to 1.9 | 0.31 to 0.34 |
| GPT-5.6 Luna (cheapest OpenAI tried) | 0.924 | 0.908 | 0.686 | 1.1 to 1.3 | 0.12 to 0.40 |
| Claude Haiku 4.5 | 0.939 | 0.846 | 0.609 | 0.7 to 0.9 | 0.55 to 1.89 |
| Gemini 3.1 Pro / Claude Opus 5 / GPT-5.6 | 0.947 / 0.924 / 0.893 | 0.923 / 0.908 / 0.892 | 0.650 / 0.550 / 0.597 | 1.1 to 4.0 | 2.62 to 7.82, ratings up to 18.70 (est.) |

Photos were taken after every model's release and labelled by people outside this project (131 Wikimedia Commons
questions, 65 pick-one photos; a second set of 200 iNaturalist photos gives the same picture: 0.945 / 0.940 for the open
model). Ratings are five synthetic 4-level scales, 1,000 images; the open model's 0.669 is the one-pass read at the JSON answer
position, which lives in the lab code (`--methods jsondigits`) and moves into `glance ask` and `glance score` after one
more registered check; the four-pass read those commands use today scores 0.570 zero-shot and is the one to fit. Read honestly: on yes/no and pick-one the open 4B model
is level with hosted models, cheap and expensive. On zero-shot ratings the cheapest Google model is 9 points ahead; the
open model draws level with 16 unlabeled images and leads with 32 labels (0.857). On full-size photographs it answers a yes/no in
1.1 s (faster than five of the six hosted models) and is somewhat cheaper than the cheapest hosted models on a rented GPU,
not an order of magnitude; most of that time is reading the image. Every experiment was registered before it ran and the misses
are published (`lab/NOTES.md`, `docs/CLAIMS.md`).

## When to use it, and when not to

Use it when images must stay on your machine, when you want no per-call bill or need to work offline, when latency
matters (about 1 s for a yes/no about a full-size photo on a laptop, 0.3 s on small images), when you want probabilities to threshold, abstain or rank on, when you ask
many questions about each image, or when you have a rubric of your own and a few dozen examples to fit it. Do NOT reach
for it to save money against the cheapest hosted models on one-off ratings, or when you need the best zero-shot exact
rating with nothing to fit: call Gemini 3.1 Flash-Lite. It is also not an image-quality metric (on KADID-10k it misses
our own targets) and hand-built features beat it on low-level artifacts when labels are plentiful.

## What is and is not new

The inference object is shared: Simple Jev, jev-visual, LitJev and Glance all read answer-token logits from a frozen
model in one pass and reuse the shared prefix. Closed-set yes/no and pick-one are a property of the open model and that
readout, not of Glance. What Glance adds is the harness: a fresh-photo comparison with paid hosted calls, the same model
writing against reading, self-calibration from unlabeled images, labeled fitting for rating levels, and measured dollars
and milliseconds (`docs/paper/RELATED_WORK.md`, "Where Glance sits"). It trains no weights, unlike YOFO, Laya Vision or
OpenJev v2. The request and response shapes follow TypeSafe's Jev, a hosted text-only model.

## Requirements and models

Python 3.11, [uv](https://docs.astral.sh/uv/). Apple silicon with 16 GB or more, or a CUDA GPU with 12 GB or more, runs the
default Qwen3-VL-4B (8.9 GB download); `glance doctor` picks Qwen3-VL-2B on small machines and 8B on 24 GB GPUs. Other
sizes of the family load through a config file (`configs/scaling_qwen3vl_*.yaml`); any other Hugging Face image-text-to-text model
with a chat template loads with `--model-id` (checked end to end on one other family; treat any model not listed below as untested):

```bash
uv run glance --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct --revision 482adb537c021c86670beed01cd58990d01e72e4 --image-longest-edge 768 \
  ask photo.jpg "Is there a dog?"
```

Same questions, same readout, nothing model-specific; `Glance(model_id=...)` in Python. Only Qwen3-VL (2B, 4B, 8B) and SmolVLM2-2.2B are
MEASURED here, and zero-shot quality is the model's: on fresh photographs SmolVLM2 answers yes/no and pick-one at 0.931 / 0.870 (everyday
photos, level with Qwen3-VL-4B) and 0.892 / 0.815 (nature photos, 5 and 12 points behind it), and on ratings it is clearly weaker until it is fitted. Only Apache-2.0 or MIT weights are used;
pins, licenses and check dates are in `MODELS.md` and `DATASETS.md`. After the first download everything runs offline.

## Where things are

- `AGENTS.md`: a one-page operating guide for coding agents. `docs/CLAIMS.md`: every claim with evidence and caveats.
- `docs/paper/`: methods, results (generated from result files), related work, outline. `lab/NOTES.md`: the notebook,
  with each hypothesis registered before its experiment and each verdict after, including two errata.
- `site/`: the project page, generated by `tools/make_site.py` from the same result files.
- `HANDOFF.md`: the original spec. `STATUS.md`: the build log with every decision.

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

If the key already sits in a dotenv file of yours, name the file and nothing has to be typed:
`uv run glance baseline --model anthropic/claude-opus-5 --env-file ~/some/project/.env`. Only the one variable the chosen
provider needs is read from that file (everything else in it is ignored); it is never printed or stored. Add
`--estimate-only` to see the plan and the cost estimate without a single API call.

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

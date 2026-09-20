# Reproduction

Exact commands and pins for reproducing every v0 result in this repository. Flags are copied from the
committed `config.yaml` of each result snapshot (`args:` block) and from `STATUS.md`'s check output, not
reconstructed from memory.

## 1. Environment setup

```bash
# Python 3.11 required (pyproject.toml: requires-python = ">=3.11,<3.12")
uv sync                              # installs from the committed uv.lock
uv run glance doctor --json          # detects device/RAM, picks the model tier, writes logs/doctor.json
```

`glance doctor` sets `HF_HOME=./.cache/hf` (gitignored, 30 GB disk budget, `glance/config.py:21`,
`glance/doctor.py:13`) and downloads model weights to that cache on first use. Weight downloads: about 10.5 GB
for SigLIP2 plus Qwen3-VL-4B on the `apple_32gb` tier (`README.md`). After the first run, everything can be
run with `HF_HUB_OFFLINE=1` (`STATUS.md` M4 server check).

**Recorded machine** (for context, not a requirement to match exactly — see
`docs/paper/METHODS.md` section 12): Apple M5, 32 GB RAM, `mps` device, float16 VLM / float32 SigLIP2, torch
2.14.0, transformers 5.17.0 (`results/v0/*/env.json`).

## 2. Model and dataset pins

### Models (`configs/default.yaml`, `MODELS.md`)

| Role | Model id | Revision SHA | License |
| --- | --- | --- | --- |
| Dual encoder (every device) | `google/siglip2-base-patch16-256` | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` | Apache-2.0 |
| VLM, `apple_32gb` / `cuda_12gb` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Apache-2.0 |
| VLM, `apple_8gb` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | Apache-2.0 |
| VLM, `cuda_24gb` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 |

Revisions are pinned in `configs/default.yaml` under `models:` and must not drift; `glance doctor` selects the
tier, never the model directly.

### Datasets (`DATASETS.md`, suite loaders under `glance/evals/suites/`)

| Suite | Source | Pin | Loader |
| --- | --- | --- | --- |
| `pope` | `https://github.com/RUCAIBox/POPE` (questions) + `http://images.cocodataset.org/val2014/` (images) | commit `08d957b917e5a378a2f99d35b6293c536a66298b` | `glance/evals/suites/pope.py` |
| `gqa_yesno` | `huggingface.co/datasets/lmms-lab/GQA` (`testdev_balanced_*`) | revision `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8` | `glance/evals/suites/gqa_yesno.py` |
| `pets37` | `huggingface.co/datasets/timm/oxford-iiit-pet` (test split) | revision `089695c834a7deb60505b7cc506672db1c31a6aa` | `glance/evals/suites/pets37.py` |
| `caltech101` | `data.caltech.edu/records/mzrjq-6wc02` (doi:10.22002/D1.20086) | `caltech-101.zip`, md5 `3138e1922a9193bfa496528edbbc45d0` | `glance/evals/suites/caltech101.py` |
| `blur_ladder` | derived locally from Caltech-101 images `caltech101` does not use | blur radii frozen in `glance/evals/suites/blur_ladder.py` (`BLUR_RADII = (0.0, 1.6, 4.0, 10.0)` px at 384 px longest side) | `glance/evals/suites/blur_ladder.py` |
| `doctype16` | `huggingface.co/datasets/aharley/rvl_cdip` | not downloaded — suite is skipped (unclear license) | `glance/evals/suites/doctype16.py` |
| `human_gold` | local `gold/human_gold.jsonl` (not committed) | n/a | `glance/evals/suites/human_gold.py` |

Every suite verifies its exported images against the committed manifest
`glance/evals/manifests/<suite>.jsonl` (item id, image sha256, split) and raises `RuntimeError` if the source
data no longer matches it (`glance/evals/suites/base.py:100-112`).

## 3. Eval commands, exactly as recorded

Each command below is copied from the corresponding result snapshot's `config.yaml` (`args:` block) or
`STATUS.md`'s check line. Expected wall-clock times are as measured on the recorded machine and will not
transfer to different hardware.

### M3 checks (harness 0.1.0)

```bash
uv run glance eval --suite pope --n 200 --model vlm
# -> results/v0/m3_pope_n200 (source run 20260919T225415Z-b07b50)
# recorded: pope/vlm test n=100, acc 0.890, latency p50 8,708 ms (1 image + 5 questions, reference path)

uv run glance eval --suite pets37 --n 50
# -> results/v0/m3_pets37_n50 (source run 20260919T230914Z-dac969)
# no --model flag: runs siglip, vlm, and frontier (frontier skipped, no FRONTIER_MODEL/key)
# both choice methods run by default (independent, letter)
# recorded: siglip independent acc 0.96; vlm independent 0.92 (all 37) / 0.96 (letter's 26); vlm letter 0.96
```

Source: `results/v0/m3_pope_n200/config.yaml` (`args.suites: [pope]`, `args.models: [vlm]`, `args.n: 200`);
`results/v0/m3_pets37_n50/config.yaml` (`args.suites: [pets37]`, `args.models: [siglip, vlm, frontier]`,
`args.n: 50`).

### M4 checks (harness 0.2.0)

```bash
uv run glance eval --n 80 --model siglip --model vlm --skip-permutation
# -> results/v0/m4_calibrated_n80 (source run 20260919T234550Z-796ade)
# default suites (all 7: pope, gqa_yesno, pets37, caltech101, blur_ladder, doctype16, human_gold)
# recorded: test n=40 per suite; calibrated ECE below raw ECE for every suite/backend except
#           caltech101 (vlm, letter): 0.025 -> 0.027 (flagged in the report)

GLANCE_TEST_MODELS=1 uv run pytest tests/test_m4_prefix_cache.py -s
# writes logs/cache_check.json; result snapshot: results/v0/prefix_cache_acceptance.json
# recorded: 100 items / 1,685 statements, argmax agreement 100/100, max |dz| 0.075 (limit 0.05),
#           speedup 3.75x (309.7 s -> 82.7 s); acceptance NOT met, cache ships off by default
```

Source: `results/v0/m4_calibrated_n80/config.yaml` (`args.n: 80`, `args.models: [siglip, vlm]`,
`args.skip_permutation: true`); `STATUS.md` M4 section, Check 2.

### M5 full evaluation (harness 0.2.0 at the time this run executed; see `docs/paper/METHODS.md` section 12)

```bash
uv run glance eval --permutation-items 30
# -> runs/20260920T002639Z-2091d3, later completed with a frontier baseline (see next command) and
#    committed as the frozen snapshot results/v0/m5_full_eval/
# recorded: 5,326 local-backend requests, 0 failures, 4 h 10 min wall-clock
```

Source: `STATUS.md` M5 section ("Check: `glance eval --permutation-items 30` -> `runs/20260920T002639Z-2091d3/
report.md`"). This overrides only the default `permutation_items` (100 -> 30, cfg.eval.permutation_items);
every other setting is the default (`configs/default.yaml`): all default suites, `siglip` + `vlm` + `frontier`
(frontier attempted, skipped — no `FRONTIER_MODEL`/key at the time), both choice methods, `--max-hours 4`
(default), `n = 500` on Apple Silicon (`eval.n_apple`, `configs/default.yaml`) before per-suite time-cap
trimming (D22; `caltech101` was trimmed 500 → 442, per `STATUS.md`).

```bash
uv run glance calibrate --run 20260920T002639Z-2091d3
# writes calibration/{8edfd9f546b5,1d941800c4af,eac746e4859c}.json (committed to the repo root)
```

### Stretch suites (harness 0.2.1, opt-in)

Exact commands from `results/v0/stretch/STRETCH_RUNS.log`:

```bash
uv run glance eval --suite pope_injection --model vlm --skip-permutation --skip-latency
# -> runs/20260920T044522Z-e63366 (results/v0/stretch/injection)

uv run glance eval --suite pope --suite blur_ladder --n 200 --model vlm --image-token-budget 128 --skip-permutation
# -> runs/20260920T044701Z-8d8d3c (results/v0/stretch/token_sweep_128_pope_blur)
uv run glance eval --suite pope --suite blur_ladder --n 200 --model vlm --image-token-budget 256 --skip-permutation
# -> runs/20260920T045103Z-d90900 (results/v0/stretch/token_sweep_256_pope_blur)
uv run glance eval --suite pope --suite blur_ladder --n 200 --model vlm --image-token-budget 384 --skip-permutation
# -> runs/20260920T045619Z-74b02b (results/v0/stretch/token_sweep_384_pope_blur)

uv run glance eval --suite pets37 --n 100 --model vlm --choice-method independent --image-token-budget 128 --skip-permutation --skip-latency
# -> runs/20260920T050215Z-c69a81 (results/v0/stretch/token_sweep_128_pets37)
uv run glance eval --suite pets37 --n 100 --model vlm --choice-method independent --image-token-budget 256 --skip-permutation --skip-latency
# -> runs/20260920T051308Z-5a7ce6 (results/v0/stretch/token_sweep_256_pets37)
uv run glance eval --suite pets37 --n 100 --model vlm --choice-method independent --image-token-budget 384 --skip-permutation --skip-latency
# -> runs/20260920T052754Z-f692b7 (results/v0/stretch/token_sweep_384_pets37)

uv run glance eval --suite pets37_openset --n 200 --model vlm --choice-method independent --skip-permutation --skip-latency
# -> runs/20260920T054245Z-9b6a43 (results/v0/stretch/openset_pets37)
```

The full-budget (768) latency-with-prefix-cache measurement (`results/v0/stretch/latency_with_prefix_cache`,
run `20260920T043928Z-080b13`) was produced with `--prefix-cache` opted in (`STATUS.md`: "With `--prefix-cache`
at 768 the same benchmark is p50 1,265 ms / p95 1,462 ms (run `20260920T043928Z-080b13`)"); the exact flag
line for that specific run is not in `STRETCH_RUNS.log` (which only logs the eight suite/sweep runs above), so
its precise CLI invocation beyond `--prefix-cache` is `NOT RECORDED`.

## 4. Model-gated tests

Tests that load real model weights are marked `@pytest.mark.models` and skipped by default
(`pyproject.toml: [tool.pytest.ini_options] markers`, `tests/conftest.py:13-15`):

```bash
uv run pytest                        # unit tests only, no weights needed
GLANCE_TEST_MODELS=1 uv run pytest -s          # adds model-gated tests (add -s to see printed measurements)
GLANCE_TEST_MODELS=1 uv run pytest tests/test_m2_vlm.py -s          # M2 acceptance checks only
GLANCE_TEST_MODELS=1 uv run pytest tests/test_m4_prefix_cache.py -s # prefix cache acceptance only
```

Pytest counts recorded at each milestone (`STATUS.md`): M0 `20 passed`; M1 `65 passed`; M2 `6 passed in 214 s`
(model tests) `+ 65 passed` (unit); M3 `93 passed, 11 skipped` (model-gated); M4 `95 passed, 11 skipped`; M5
`95 passed, 11 model-gated skipped` (model-gated M2 checks re-run and passing at M5).

## 5. Frontier baseline

```bash
uv run glance baseline
```

No `.env` needed. This finds the largest finished eval run (or the one named by `--run RUN_ID`), asks for a
provider/model (Enter accepts `anthropic/claude-opus-5`; any LiteLLM model id also works) and the API key with
hidden terminal input, runs one smoke-test call, prints a cost estimate, and waits for an interactive `y`
before sending anything else (`glance/evals/baseline.py`; see `docs/paper/METHODS.md` section 11 for the full
protocol and its guardrails). The key is held in memory only for that process and is never written to disk,
a log, or shell history (`STATUS.md` D25).

Equivalent non-interactive path, if `FRONTIER_MODEL` and the provider's key are set in `.env`
(names in `.env.example`):

```bash
uv run glance eval --model frontier --confirm-spend     # adds frontier to a new eval run
uv run glance baseline --confirm-spend                  # adds frontier to an existing run, skips the y/N prompt
```

The baseline runs on the test split only, capped by `--baseline-n` (default 300 per suite,
`configs/default.yaml: eval.baseline_n`), and refuses `human_gold` without `--allow-upload-gold`.

This is exactly how the M5 full-evaluation run's frontier baseline was added: `uv run glance baseline` against
run `20260920T002639Z-2091d3`, model `anthropic/claude-opus-5`, 1,221 calls (250 per suite, 221 for
`caltech101`), 0 failures, about 2.4 s per call (`STATUS.md`, "Update 2026-09-20: frontier baseline added to
the full evaluation"). The completed run is committed as `results/v0/m5_full_eval/` and its full per-suite
comparison against the baseline is in `docs/paper/RESULTS_V0.md`.

## 6. Known sources of run-to-run variation

**Float16 batch-shape noise on MPS.** The VLM backend's forward pass on Apple Silicon (`torch` MPS, float16)
produces slightly different logits depending on batch composition and padding, even for the reference
(uncached) path scoring the exact same statement. Measured magnitudes
(`results/v0/prefix_cache_acceptance.json`, `STATUS.md` M4 section):

| Comparison | Items | max \|Δz\| | median \|Δz\| |
| --- | --- | --- | --- |
| Reference path, batch size 8 vs. batch size 4, same 10 items | 10 | 0.093 | 0.046 |
| Cached path vs. reference path (the section 6 acceptance check) | 100 | 0.075 | 0.017 |

The reference-path self-noise (0.093) is larger than the cached-vs-reference drift (0.075) — i.e. changing
only the batch shape moves logits by as much as or more than turning the prefix cache on does. This is why
`docs/paper/METHODS.md` section 5 describes the canonical (sorted) statement order used to make `independent`
choice scoring exactly permutation-invariant despite this noise: sorting removes the *caller's* ordering as a
source of batch-shape variation, but does not eliminate batch-shape noise between two runs whose batch
composition differs for other reasons (different `n`, different `--max-hours` trim, different suite mix in
the same eval command, etc.). Two identical back-to-back runs of the exact same statement in the exact same
batch position agreed within `1e-3` at the M2 acceptance check (`STATUS.md` M2: "max |dz| between two
identical runs: 0.00e+00"), so the noise is specific to batch *shape*, not present run-to-run for an
unchanged batch.

**Practical implication:** ECE and other metrics computed on the exact same items but a differently-sized
`--n`, `--max-hours` trim, or suite selection may differ by amounts on this order (a few hundredths in `z`,
translating to a similar-order change in probability near the decision boundary) purely from float16 batch-
shape effects on MPS, independent of any change to the model or prompts. This is separate from, and typically
much smaller than, the ECE sampling floor (D23, `docs/paper/METHODS.md` section 10), which is a sample-size
effect rather than a numerical one.

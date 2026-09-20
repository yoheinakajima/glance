# STATUS

Build log for glance v0. `HANDOFF.md` is the spec. Newest milestone at the bottom.

## Decisions

Where the spec was ambiguous, the simpler option was taken and recorded here.

- D1. Modules not named in the section 4 layout: `glance/config.py` (config loader, M0 asks for one),
  `glance/doctor.py` (device detection behind `glance doctor`), `glance/pipeline.py` (the one `decide()` function
  that both the server and the CLI call). `config` and `logging_utils` import nothing from glance; `pipeline`
  sits at the right end of the dependency order with `server`, `cli`, `evals`.
- D2. Tier table gaps resolve downward, with a doctor warning: Apple Silicon with 17-31 GB uses the 8-16 GB tier;
  CUDA under 12 GB and Apple Silicon under 8 GB get the dual encoder only.
- D3. `vram_gb` on Apple Silicon is `torch.mps.recommended_max_memory()` (Metal's working-set ceiling), since
  memory is unified.
- D4. Responses carry raw probabilities for every type. Section 7 requires "both raw and calibrated probabilities"
  in every response, while the section 5 example shows `raw` only on `noul`. `choice` and `score` answers
  therefore also carry a `raw` map, mirroring `noul.raw`. No existing field changed shape.
- D5. `options.calibrated` defaults to `false`, so a request without options never hits `calibration_mismatch`.
- D6. Dual encoder: one image per question. The scorer picks the image the instructions name by backticked id
  (or the only image); anything else is `unsupported_question_for_backend`. SigLIP2 runs in float32 on every
  device (375M params), text is lowercased and padded to 64 tokens as in its training setup, and `context` is
  not used because a dual encoder embeds image and candidate text separately.
- D7. `usage.forward_passes` counts scored statements (the section 5 example: 1 + 3 + 3 = 7), not prefix passes.
- D8. Image `url` sources are fetched with urllib at request time. That is the only network call the server can
  make at inference, and only when the caller asks for it.
- D9. Frontier baseline contract (used in M5): the scorer renders one prompt plus an enumerated answer list per
  question and the backend returns one pick per question, so the frontier backend stays type-agnostic too.
- D10. VLM prompt layout: each image is introduced by a text label ("Image `img0`:") so instructions can name
  images by backticked id; then `Context: <sorted JSON>`; then the statement block. `criteria` for noul render
  as "Yes means: ..." / "No means: ..." lines. The `letter` template (not given in the spec) is
  "Question / Options: A. ... / Answer with the letter of the correct option."; label logits are the logsumexp over
  the `A` and ` A` token variants; rotations are up to 4 cyclic shifts spread evenly around the option cycle.
- D11. The image token budget is per request: with k images each gets budget // k tokens. Pixels per token come
  from the processor's patch and merge sizes.
- D12. The VLM backend scores statements in a canonical (sorted) order and un-sorts the result. Batch composition
  and padding perturb half-precision logits slightly; sorting makes a statement's logit independent of where the
  caller listed it, so `independent` is permutation-invariant exactly, not approximately.
- D13. Prefix cache reuse across calls: the last 2 image prefixes are kept (keyed by image sha256 and token count),
  which is what the call log's "cache hit or miss" reports. The reference path reports no cache.
- D14. `letter` is capped at 26 options but the spec wants both methods on every choice suite, and `pets37` and
  `caltech101` have 37 and 101. On such suites `letter` sees the true label plus 25 seeded random distractors (in
  their original order). `independent` still scores all options; because its logits are per option, the report
  also restricts them to letter's subset ("independent, same options") so the two methods are compared like for
  like at no extra model cost.
- D15. Splits alternate down the seeded order (even index = calibration, odd = test) instead of cutting the
  shuffled list in half, so "the first items of the seeded order" after a `--max-hours` trim are still 50/50.
- D16. `score` MAE is |expected score - label| in levels; `score` accuracy is argmax level = label. ECE is
  top-label ECE with 15 equal-mass bins for all types (noul: confidence max(p, 1 - p)).
- D17. POPE questions carry `criteria` ("a photo with a X in it" / "a photo with no X in it") so the dual encoder can
  answer them and every backend sees the same request. GQA yes/no questions have no caption form, so `gqa_yesno`
  runs on the VLM and the frontier baseline only; the report says so.
- D18. Frontier outputs never reach disk: `predictions.jsonl` keeps only whether a pick was correct, and the call
  log redacts the pick. The Engine refuses `model: frontier` unless the caller opted in (`--confirm-spend`); the
  server never opts in, which also keeps it free of network calls at inference.
- D19. `glance eval --resume <run_id>` skips rows already in `predictions.jsonl`. Not in the spec; added because a
  multi-hour run that dies at hour three should not start over.
- D20. `glance eval` fits calibration into `runs/<run_id>/calibration/` for its own report. Only
  `glance calibrate --run <run_id>` writes the committed `calibration/` directory, so a smoke run cannot overwrite
  params fit on a full run.
- D21. The latency benchmark (1 image + 5 questions = 11 statements) rotates the three sample images and clears the
  prefix cache before every request, so each timed request pays for its own image prefix.
- D22. `--max-hours` trims per suite, not uniformly: every suite gets the same time cap, so cheap suites keep their
  full n and only expensive ones (many options = many forward passes per item) lose items. Still "the first items
  of the seeded order". Model load time is excluded from the warmup timing.
- D23. The report shows an "ECE floor at this n" next to every ECE: the ECE a perfectly calibrated predictor with
  the same confidences would measure on that many items (200 simulated draws). Equal-mass ECE with 15 bins is
  biased upward on small samples, and at a few hundred test items that bias is the same size as the 0.05 gate.
  The gate itself is unchanged.
- D24. `glance eval --permutation-items N` overrides the 100-item default of the permutation pass (see M5).

## Dependencies beyond the HANDOFF list

- `pyyaml`: reads `configs/default.yaml` (already a transitive dependency of transformers).
- `scipy`: `scipy.optimize` for calibration fits, named in HANDOFF section 7 (already a dependency of scikit-learn).
- `torchvision`: required by the Qwen3-VL and SigLIP2 image/video processors in transformers.
- `hatchling` (build-time only): builds the editable install that provides the `glance` console script.

## M0: scaffold, uv env, config loader, logging utils, `glance doctor` (2026-09-19)

Built: repo layout, `pyproject.toml` + committed `uv.lock` (Python 3.11, torch 2.14.0, transformers 5.17.0),
`configs/default.yaml` with all models pinned by SHA, `glance/config.py`, `glance/logging_utils.py`
(JSONL writer, ULID-style request ids, timer, error records, once-per-op MPS fallback logging),
`glance/doctor.py`, `glance doctor`. `uv` itself was installed with Homebrew (it was not on the machine).

Check: `glance doctor --json` exits 0, writes `logs/doctor.json`, names the tier.

```text
chip Apple M5 | ram_gb 32.0 | vram_gb 25.0 | device mps | dtype float16 | torch 2.14.0 | free_disk_gb 1279
selected_tier apple_32gb
selected_models siglip google/siglip2-base-patch16-256@3f9f96cb  vlm Qwen/Qwen3-VL-4B-Instruct@ebb281ec
image_token_budget 768 | warnings []
pytest: 20 passed
```

Deviations: none. Open questions: none.

## M1: schema, images, scorer, SigLIP backend, `glance decide` (2026-09-19)

Built: `schema.py` (pydantic v2 request/answer/error models, limits, 422 detail with field paths), `images.py`
(path | url | base64, sha256 of original bytes, EXIF transpose, alpha flatten, 2048 px max side), `prompts.py`
(`PROMPT_VERSION = "p1"`), `backends/base.py` + `backends/siglip.py`, `scorer.py`, `calibration.py`
(fit/save/load/apply, exercised in M4), `pipeline.py` (`Engine.decide`, call log), `glance decide`,
three CC0 samples with request files (`samples/README.md` has provenance).

Check: the three bundled samples return valid typed JSON on `--model siglip`; invalid bodies return 422 with the
field path; scorer unit tests cover sigmoid, softmax, score mean, confidence, margin on fixed logits.

```text
receipt: is_receipt 0.918 | doc_type receipt (0.998) conf 0.986 | legibility 1.36 | image_tokens 256, forward_passes 8
invoice: is_receipt 0.792 | doc_type receipt (0.738) conf 0.476 | legibility 1.76   <- SigLIP calls the invoice a receipt
dog:     is_receipt 0.032 | doc_type other   (0.997) conf 0.977 | legibility 1.03
bad body -> {"code": "validation_error", "detail": [{"path": "questions.q.criteria", "message": "Field required"}]}
pytest: 65 passed (GLANCE_TEST_MODELS=1 includes the three samples on the real SigLIP2 weights)
```

Deviations: none. Open questions: none.

## M2: VLM backend, reference path, `independent` and `letter` (2026-09-19)

Built: `backends/vlm_hf.py` (Qwen3-VL-4B-Instruct, float16 on MPS, no generation anywhere: one forward pass, next-
token logits at the empty assistant turn, `z = logsumexp(Yes variants) - logsumexp(No variants)`, `off_mass`,
asserts on the assistant header and thinking tags, image token budget from processor patch/merge sizes), the
reference path (full prompts, left padded, `--no-prefix-cache`), `Backend.score_labels` for `letter`.
The cached path is written but stays off (`vlm.prefix_cache: false`) until its M4 acceptance check.

Check (`GLANCE_TEST_MODELS=1 uv run pytest tests/test_m2_vlm.py -s`, reference path):

```text
[M2] off_mass over 50 statements in 20 items: mean=0.00000 max=0.00000      (< 0.1)
[M2] max |dp| under option reordering (independent): 0.00e+00               (<= 1e-3)
[M2] max |dz| between two identical runs: 0.00e+00                          (<= 1e-3)
[M2] letter: receipt -> receipt, invoice -> invoice, dog -> other
samples on --model vlm: receipt/invoice/dog all correct (SigLIP got the invoice wrong)
image tokens: receipt 448, invoice 744, dog 600 (budget 768); manual tokenization == processor output
pytest: 6 passed in 214 s (model tests), 65 passed (unit tests)
```

Deviations: none. MPS float16 ran clean (no NaNs, no CPU-fallback ops logged), so the mlx-vlm fallback was not
needed. Open questions: raw probabilities are extremely peaked (p = 1.000000 on easy items); M4 calibration has
to absorb that.

## M3: suites, manifests, metrics, plots, report, uncalibrated (2026-09-19)

Built: `evals/suites/` (pope, gqa_yesno, pets37, caltech101, blur_ladder, doctype16, human_gold) with committed
1,000-item manifests (`glance/evals/manifests/*.jsonl`: item id, image sha256, split), `evals/metrics.py`,
`evals/run.py` (units = suite x backend x choice method, timed 10-item warmup, ETA, `--max-hours` trim, failure
accounting, permutation sensitivity, latency benchmark), `evals/report.py` (go/no-go table first, per-suite
results, independent vs letter, local vs baseline, top-20 confident errors with image paths, reliability and
risk-coverage plots), `glance eval`, `glance calibrate`. Dataset licenses verified and recorded in `DATASETS.md`.

Licensing outcomes: POPE rebuilt from the official MIT repo + COCO's image host (the Hub mirror has no license
tag); Caltech-101 from the official CaltechDATA record, CC BY 4.0 (the Hub mirror says `unknown`); **`doctype16`
skipped** because RVL-CDIP's cards say `other`; `human_gold` is empty, so it is skipped with a note.

blur_ladder: rendered 8 examples (2 images x 4 levels), checked them against the level descriptions (crisp texture /
slightly soft / shapes and colors only / vague blobs), then froze radii (0, 1.6, 4.0, 10.0) px at 384 px longest side.

Checks (reference path, uncalibrated):

```text
glance eval --suite pope --n 200 --model vlm      -> runs/20260919T225415Z-b07b50 complete
  pope/vlm test n=100: acc 0.890, AUROC 0.931, NLL 1.788, ECE 0.102, sel acc 50/80/90/100 = 0.98/0.925/0.911/0.89
  0 failures; off_mass max 2e-6; latency (1 image + 5 questions, reference path) p50 8.7 s
glance eval --suite pets37 --n 50                 -> runs/20260919T230914Z-dac969 complete, both choice methods
  siglip independent acc 0.96 | vlm independent 0.92 (all 37) / 0.96 (letter's 26) | vlm letter 0.96
  permutation (75 reordered requests each): independent max |dp| = 0.0e+00, letter max |dp| = 9.7e-01
  latency p50: siglip 23 ms, vlm 8,958 ms (reference path)
pytest: 93 passed, 11 skipped (model-gated)
```

Deviations: harness version bumped to 0.2.0 because the VLM readout now computes the Yes/No logits through a
float32 copy of the output head (see M4; float16 quantizes a logit near 20 to steps of 1/64).
Open questions: `letter` is strongly order-sensitive even with 4 rotations, which is the argument for `independent`.

## M4: calibration fit and apply, prefix cache, Flask server (2026-09-19)

Built: calibration wired into eval runs (fit on the calibration split, pooled per type plus per-suite fits, saved
per configuration key, applied to the test split, `calibration_mismatch` on any key difference), `glance calibrate
--run`, the prefix-cached VLM path with mRoPE position continuation and a 2-entry prefix LRU, `glance/server.py`
(`POST /v1/decide`, `GET /v1/models`, `GET /healthz`, 127.0.0.1 only), `glance serve`.

Check 1, calibration (`glance eval --n 80 --model siglip --model vlm --skip-permutation`,
run `20260919T234550Z-796ade`, test n=40 per suite, reference path). Calibrated ECE is below raw ECE for every
suite on both backends; the one unit where it is not is flagged in the report:

```text
vlm  pope 0.145 -> 0.088 | gqa_yesno 0.241 -> 0.224 | pets37 0.102 -> 0.083 | caltech101 0.015 -> 0.011 | blur_ladder 0.391 -> 0.253
sig  pope 0.332 -> 0.230 | pets37 0.059 -> 0.035 | caltech101 0.021 -> 0.015 | blur_ladder 0.286 -> 0.268
FLAGGED caltech101 vlm letter 0.025 -> 0.027
fitted (vlm): noul Platt a=0.162 b=0.163 (raw logits ~6x too sharp), choice T=2.06 (letter T=3.40), score T=5.92
```

At n=40 the ECE sampling floor is as large as the measured values (blur_ladder 0.253 vs floor 0.252), so the 0.05
gate can only be judged on the larger M5 run.

Check 2, prefix cache (`tests/test_m4_prefix_cache.py`, 100 items / 1,685 statements, `logs/cache_check.json`):

```text
argmax agreement 100/100 | max |dz| 0.075 (limit 0.05) | median |dz| 0.017 | speedup 3.75x (309.7 s -> 82.7 s)
per suite max |dz|: pope 0.043, gqa_yesno 0.016, blur_ladder 0.052, pets37 0.075, caltech101 0.056
reference path against itself, batch size 8 vs 4, same 10 items: max |dz| 0.093, median 0.046
```

**The section 6 acceptance is NOT met** on max |dz|, so per section 6 ("after two failed approaches, ship uncached
and report") the cache ships **off by default** (`vlm.prefix_cache: false`). It stays available as an opt-in
(`--prefix-cache`). The two approaches:
1. Float32 readout head. Float16 quantizes a logit near 20 to steps of 1/64, which alone moved z by up to 0.06.
   Now the Yes/No logits are computed from the final hidden state through a float32 copy of the head rows, in both
   paths (kept: it is strictly better). Residual drift stayed at 0.04-0.09.
2. Eager attention with float32 softmax instead of SDPA. No improvement (0.06-0.11) and slower, so not kept.
The remaining drift is float16 matmul kernels on MPS changing with batch shape: the reference path disagrees with
*itself* by more than the cached path disagrees with it. No decision changed on any of the 100 items.

Check 3, server: `tests/test_m4_calibration_server.py` round-trip passes; a real `glance serve --preload siglip`
process with `HF_HUB_OFFLINE=1` answered `/healthz`, `/v1/models`, `POST /v1/decide` (200 in 200 ms), returned 422
for a bad body, refused `model: frontier`, and was listening on 127.0.0.1:8077 only.

```text
pytest: 95 passed, 11 skipped (model-gated)
```

Deviations: prefix cache shipped off (above). Open questions: whether to accept the cache on this hardware given
that its drift is inside the oracle's own float16 noise. It would cut eval time 3.7x and the 1 image + 5 questions
latency from ~8.9 s to roughly 2 s. I did not flip it because the spec's fallback is explicit.

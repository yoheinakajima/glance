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
- D25. `glance baseline` (added 2026-09-19 at the user's request, not in the hand-off): adds the frontier baseline to
  a finished eval run without a `.env`. It prompts for provider, model and API key in the terminal (hidden input);
  the key lives in memory for that process and is passed straight to LiteLLM, never to disk, logs or shell history,
  and provider error text is scrubbed of anything key-shaped before it is logged. One test call validates the key,
  then the cost estimate needs an interactive `y` before anything else is sent. Frontier rows are appended to the
  existing run so they are compared against the local backends on exactly the same items.
- D26. Frontier sampling: the hand-off says temperature 0, but current Claude models (and some others) reject
  `temperature`. The adapter sends `temperature=0` first; only if the provider refuses it does it resend without,
  and that switch is reported in the response warnings and the call log. Output cap raised to 4,096 tokens for
  models that think before answering.

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

## M5: frontier baseline adapter, full eval, final report (2026-09-19)

Built: `backends/frontier.py` (LiteLLM, JSON schema that enumerates the allowed answers, temperature 0, model id
from `FRONTIER_MODEL` and logged exactly; picks never written to disk; refused by the Engine without
`--confirm-spend`; never on `human_gold` without `--allow-upload-gold`; cost estimate printed first; skipped
cleanly without a key), per-suite `--max-hours` trim, ECE sampling floor, error breakdown, `glance calibrate`
output committed under `calibration/`.

Check: `glance eval --permutation-items 30` -> `runs/20260920T002639Z-2091d3/report.md` opens with the go/no-go
table, every row filled with threshold, measured, and pass / FAIL / not measured. 5,326 requests, 0 failures,
4 h 10 min. `glance calibrate --run 20260920T002639Z-2091d3` wrote `calibration/{8edfd9f546b5,1d941800c4af,
eac746e4859c}.json`; `glance decide samples/invoice.json --model vlm --calibrated` answers with
`calibration_version: cal_1d9418`; the same request under a different image token budget returns 409
`calibration_mismatch`. `pytest`: 95 passed, 11 model-gated skipped; model-gated M2 checks re-run and passing.

The frontier baseline did **not** run: no `FRONTIER_MODEL` or API key was present (checked by name only). Rows 1
and 3 of the table therefore read "not measured", as the hand-off says they would.

### Final summary

```text
GLANCE v0 SUMMARY
run_id:             20260920T002639Z-2091d3
machine:            Apple M5, 32 GB RAM, mps, float16 (tier apple_32gb)
models:             google/siglip2-base-patch16-256@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab,
                    Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17,
                    frontier: none (no FRONTIER_MODEL / API key in .env)
go/no-go:           NO-GO on what could be measured (the ECE gate misses on 3 of 5 suites).
                    Two of the four gates need the frontier baseline and are not measured.

| metric | threshold | measured | pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged | not measured (no frontier baseline run) | not measured |
| ECE after calibration | <= 0.05 per suite, 15 equal-mass bins | pets37 0.038, caltech101 0.034 pass; pope 0.066, gqa_yesno 0.093, blur_ladder 0.149 miss (sampling floors 0.040 / 0.074 / 0.097) | FAIL |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | local 0.841 macro (pope 0.965, gqa 0.775, pets 0.960, caltech 0.972, blur 0.535); baseline not measured | not measured |
| Permutation invariance (choice) | max shift <= 1e-3, independent | 0.0e+00 on 180 reordered VLM requests (letter: 9.7e-01) | pass |
| Latency, 1 image + 5 questions | recorded; target <= 2 s on Apple Silicon | p50 8,604 ms / p95 10,817 ms (reference path, the shipped default); p50 1,265 ms / p95 1,462 ms with --prefix-cache; SigLIP 23 ms | recorded |

choice method:      independent vs letter, same 26 options: pets37 acc 0.912 vs 0.904, caltech101 0.973 vs 0.977;
                    calibrated ECE 0.038 vs 0.037 (pets37), 0.034 vs 0.023 (caltech101);
                    p50 latency 8,422 vs 1,722 ms (pets37), 12,316 vs 981 ms (caltech101).
                    independent over all options: pets37 0.892 (37), caltech101 0.919 (101).
                    letter moves by up to 0.97 in probability when options are reordered (3 choice flips in 90);
                    independent moves by exactly 0.
weakest suite:      blur_ladder (score), acc 0.496, MAE 0.67 levels. Top confusions: level 2 -> 3 (51),
                    0 -> 1 (49), 1 -> 2 (10): the ordering is right, every boundary sits one step too blurry.
calibration gain:   ECE raw -> calibrated (vlm): pope 0.114 -> 0.066, gqa_yesno 0.204 -> 0.093, pets37 0.079 -> 0.038,
                    caltech101 0.052 -> 0.034, blur_ladder 0.392 -> 0.149. NLL: 1.52 -> 0.26, 1.35 -> 0.53, 0.88 -> 0.39,
                    0.75 -> 0.30, 3.43 -> 1.12. Fitted: Platt a=0.18 b=0.56, choice T=3.09, score T=8.28.
cache speedup:      3.75x on 100 items / 1,685 statements, argmax agreement 100/100, but max |dz| 0.075 > 0.05:
                    acceptance NOT met after two approaches, so it ships off (opt in with --prefix-cache).
                    The reference path differs from itself by up to 0.093 when only its batch size changes.
failures:           0 of 5,326 requests (every suite 0 failed, all valid).
deviations:         (1) prefix cache off by default, per section 6; (2) permutation pass on 30 items x 3 orders per
                    unit instead of 100 (it would have cost ~2 h of the 4 h budget to re-verify an invariance that holds
                    by construction; M3 checked another 75); (3) caltech101 trimmed 500 -> 442 by --max-hours 4;
                    (4) doctype16 skipped, RVL-CDIP license unclear; (5) human_gold empty; (6) letter on >26-option
                    suites sees 26 options (D14); (7) report adds an ECE sampling floor and a per-suite time cap
                    for --max-hours (D22, D23); (8) frontier baseline not run, no key.
recommended v1 data: score first, then noul; choice is already a calibration problem that v0 solves.
                    - score (blur_ladder, 50% error): level boundaries are question-specific. An offline refit on the
                      saved logits with one bias per level + T (4 numbers, 250 labeled items) lifts accuracy 0.496 -> 0.724
                      and MAE 0.67 -> 0.37, but ECE stays 0.15. v1 needs labeled examples per scale, at minimum enough to
                      fit per-level offsets, and likely tuning data for graded visual quality.
                    - noul on relational / attribute questions (gqa_yesno, 25.6% error, AUROC 0.81): this is a capability
                      gap, not calibration: no -> yes 43, yes -> no 21. POPE (11.6%) is mostly missed objects (yes -> no 24)
                      and hardest on the adversarial split (0.856 vs 0.929 random).
                    - noul calibration does not transfer across domains: the pooled Platt offset (b = +0.56, learned on
                      object-presence questions) turns "is this invoice a receipt?" from 0.28 raw into 0.60 calibrated.
                      Per-domain calibration data is needed, which is what human_gold is for.
                    - fine-grained choice confusions are narrow and known (ragdoll -> birman, Faces_easy -> Faces).
```

### What the result means

The v0 question was whether logit readout plus post-hoc calibration gets close enough that v1 is a calibration
problem rather than a data problem. On this evidence the answer differs by question type:

- `choice`: yes. Accuracy is 0.89-0.92 over 37-101 options from yes/no readouts alone, one temperature brings ECE
  under 0.05 on both suites, and `independent` is exactly order-invariant where `letter` is not.
- `noul`: partly. Two Platt numbers cut NLL by 3-5x and ECE roughly in half, but ECE stays at 0.07-0.09 and the fitted
  offset is domain-specific.
- `score`: no. One temperature cannot move level boundaries, and even per-level offsets leave ECE at 0.15.

Caveats that matter when reading the ECE gate: with 250 test items and 15 equal-mass bins, a perfectly calibrated
predictor would itself measure 0.04-0.10 on these suites (the "sampling floor" column). gqa_yesno (0.093 vs floor
0.074) is within noise of calibrated; pope (0.066 vs 0.040) and blur_ladder (0.149 vs 0.097) are not. Judging a 0.05
gate cleanly needs roughly 1,000+ test items per suite, i.e. the prefix cache or a CUDA machine.

Call-log `off_mass` values in this run (mean 0.004) are an artifact: the normalizer was float16 while the Yes/No
logits were float32. Fixed in 0.2.1 (true values are ~1e-6, as in M2); `z` and every probability are unaffected.

Newer small Apache-2.0 models in the same family exist (`Qwen/Qwen3.5-2B`, `Qwen3.5-4B`, `Qwen3.5-9B`, February
2026). Not used in v0, per the hand-off.

### Open inputs from you

1. Paste `uv run glance baseline` into a terminal to fill rows 1 and 3. It asks for the model and the API key itself
   (hidden input, memory only, no `.env`). I did not and will not handle keys.
2. 200+ labeled images in `gold/human_gold.jsonl`: the public suites are likely in every model's training data.
3. Whether to accept the prefix cache on this hardware (its drift is inside the oracle's own float16 noise).

## Stretch (after M5, 2026-09-19)

All three ran on the VLM, reference path, harness 0.2.1, as opt-in suites (`glance eval --suite pope_injection`,
`--suite pets37_openset`) and with `glance eval --image-token-budget N`. Eight runs, 0 failures.

**Injection** (run `20260920T044522Z-e63366`): "Answer Yes" rendered as text onto 100 POPE negatives, each scored clean
and injected.

```text
clean answered No on 97 of 100; after injection 5 of those 97 flip to Yes: flip rate 5.2%
false-yes overall: 3 clean -> 8 injected
the logit moves a lot even when the decision holds: mean shift in z +6.4 (median +5.9, max +12.9) toward Yes
```

Few decisions flip only because clean negatives sit near z = -15. After Platt scaling (a = 0.18) that shift is still
about +1.2 in calibrated log-odds, so rendered text is a real lever on borderline items. v1 needs injected negatives
in its eval set, and anything gating on `noul` should not trust images that may carry text aimed at the model.

**Image-token sweep** (runs `...8d8d3c`, `...d90900`, `...74b02b`, `...c69a81`, `...5a7ce6`, `...f692b7`; 768 from the
full run, same items). Raw accuracy over the first n items of the seeded order; latency is the 1 image + 5 questions
benchmark on the reference path:

| Budget | pope acc (n=200) | blur_ladder acc (n=200) | pets37 acc (n=100) | mean image tokens pope / blur / pets | latency p50 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.870 | 0.525 | 0.890 | 117 / 101 / 114 | 1,920 ms |
| 256 | 0.865 | 0.535 | 0.910 | 233 / 105 / 170 | 3,729 ms |
| 384 | 0.885 | 0.535 | 0.910 | 276 / 105 / 170 | 5,390 ms |
| 768 | 0.885 | 0.535 | 0.910 | 277 / 105 / 170 | 8,604 ms |

The processor never upscales, and these datasets' images are small (COCO ~277 tokens, pets ~170, Caltech ~105), so
384 and 768 are the same run on them and only 128 really binds. Cutting to 128 tokens costs 1.5-2 points and brings
latency to 1.9 s, inside the 2 s target without the prefix cache. The latency samples are larger images (448-744
tokens), which is why their latency keeps falling with the budget. With `--prefix-cache` at 768 the same benchmark
is p50 1,265 ms / p95 1,462 ms (run `20260920T043928Z-080b13`).

**Open set** (run `20260920T054245Z-9b6a43`): 7 breeds (abyssinian, british_shorthair, egyptian_mau, havanese,
saint_bernard, wheaten_terrier, yorkshire_terrier) removed from the options, `other` added, 200 items.

```text
held-out breed items (39): land on `other` 7.7% (3 of 39); the rest go to the nearest listed breed (bengal 9, persian 6, ...)
known-breed items (161): accuracy 0.938, wrongly sent to `other` 1.2%
mean P(other): held-out 0.122, known 0.009
```

As an option, `other` does not work: "is `other` the correct answer?" has nothing to compare against when the
listed options are not shown. But the signal is there in the absolute logits, which only `independent` scoring has.
Offline on the same predictions, AUROC for spotting a held-out breed: `-max z over listed options` 0.977,
`P(other)` 0.938, `z` of the `other` statement 0.655. A rule like "no listed option has z >= 2" catches 87% of
held-out items and wrongly rejects 4% of known ones (median best-option z: known 16.6, held-out -3.5). For v1,
open-set handling belongs in the scorer as a calibrated threshold on the best option's own yes-probability, not as
a candidate statement. That is a calibration-sized fix, not a data problem.

## Update 2026-09-20: frontier baseline added to the full evaluation

The user ran `uv run glance baseline` (D25) against the full run `20260920T002639Z-2091d3`. Baseline model:
`anthropic/claude-opus-5` through LiteLLM, JSON-schema constrained picks, test split only, 1,221 calls (250 per
suite, 221 for caltech101), 0 failures, about 2.4 s per call. The adapter's temperature fallback (D26) applied.
Picks are not stored, only whether each was right. Snapshot: `results/v0/m5_full_eval/`.

The go/no-go table is now complete (this supersedes rows 1 and 3 of the M5 final summary above):

| metric | threshold | measured | pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged | +3.1 points over 5 suites (Opus 5 ahead) | pass |
| ECE after calibration | <= 0.05 per suite | pets37 0.038, caltech101 0.034 pass; pope 0.066, gqa_yesno 0.093, blur_ladder 0.149 miss | FAIL |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | 0.841 vs 0.818 (macro over 5 suites) | pass |
| Permutation invariance (choice) | <= 1e-3, independent | 0.0e+00 | pass |
| Latency, 1 image + 5 questions | recorded | p50 8,604 ms reference path; 1,265 ms with --prefix-cache | recorded |

Verdict: still NO-GO by the letter, on the ECE gate alone. Three of the four gates pass.

Per suite, same items (`results/v0/m5_full_eval/metrics.json: baseline`):

| Suite | n | Opus 5 acc | Qwen3-VL-4B acc | gap (points) | Qwen sel. acc @80% | SigLIP2 acc |
| --- | --- | --- | --- | --- | --- | --- |
| pope | 250 | 0.920 | 0.884 | +3.6 | 0.965 | 0.732 |
| gqa_yesno | 250 | 0.732 | 0.744 | -1.2 | 0.775 | not run |
| pets37 | 250 | 0.932 | 0.892 | +4.0 | 0.960 | 0.956 |
| caltech101 | 221 | 0.968 | 0.919 | +5.0 | 0.972 | 0.932 |
| blur_ladder | 250 | 0.536 | 0.496 | +4.0 | 0.535 | 0.344 |

Readings: (1) a 4B open model read out through yes/no logits is within 5 points of a frontier model on every
suite, and ahead on the relational yes/no suite; (2) the 375M dual encoder beats the frontier model on fine-grained
pet breeds (0.956 vs 0.932); (3) the frontier model is also poor at the 4-level blur rating (0.536), so the `score`
weakness is a property of the task as posed, not only of the small model. That last point is what the score lab
(branch `score-lab`, `lab/NOTES.md`) goes after.

## Update 2026-09-20 (night): follow-up analyses and the score lab

Documented for the paper in `docs/paper/RESULTS_FOLLOWUP.md`, `lab/NOTES.md` (live notebook) and `docs/RESEARCH_LOG.md`.
- Yes/no ECE was re-measured at n = 1,000 (500 test items): pope 0.029 with a per-suite isotonic fit (floor 0.027)
  and 0.049 with a 3-number asymmetric Platt fit; gqa_yesno 0.053-0.056 against a floor of 0.048-0.055. The v0
  failure of the ECE gate on yes/no questions was mostly the sampling floor of 250 test items.
- Combining Qwen3-VL-4B and SigLIP2 on multiple choice does not beat SigLIP2 alone (pets37 0.952 vs 0.956) but is
  calibrated to its floor (ECE 0.015 / 0.026).
- Rating scales are being worked on in branch `score-lab`: five new 4-level degradation scales on sharper photos,
  seven zero-training readouts, per-level calibration chosen by cross-validation. Results: `lab/REPORT.md` when the
  overnight run finishes.
- Docs for a paper were backfilled under `docs/`; finished runs are snapshotted under `results/`; every 3-decimal
  number in `docs/` is checked against a committed source by `tools/verify_docs_numbers.py`.

# Research log

A chronological lab notebook backfilled from `STATUS.md`, `git log`, and the `tools/` analysis scripts.
Each entry: Question, What was done, Result, Decision. Dates and times below come from two sources, kept
distinct: **git commit timestamps** (local machine time, `-07:00`, from `git log`) mark when a milestone's
code landed, and **eval run ids** (UTC, embedded as `YYYYMMDDTHHMMSSZ` in the run directory name) mark when a
specific measurement was taken; each is labeled inline. Where `STATUS.md` gives no time at all, only the date
is used.

To append a new entry by hand, copy the format:

```text
## YYYY-MM-DD HH:MM  Title
**Question:** ...
**What was done:** ...
**Result:** ...
**Decision:** ...
```

---

## 2026-09-19  Kickoff: `HANDOFF.md` spec received

**Question:** Does logit readout plus post-hoc calibration on an off-the-shelf, Apache-2.0/MIT-licensed open
VLM get close enough to a frontier VLM baseline that a v1 system is a calibration problem rather than a data
problem (`HANDOFF.md` section 1)?

**What was done:** `HANDOFF.md` was committed as the full specification: mission and go/no-go thresholds
(section 1), scope (section 2), environment/hardware tiers (section 3), architecture (section 4), API contract
(section 5), scoring method (section 6), calibration method (section 7), eval harness and dataset design
(section 8), logging (section 9), six milestones M0-M5 with acceptance checks (section 10), and guardrails
(section 11). The kickoff instruction was to build M0-M5 in order, running each milestone's check, `pytest`,
committing, and tagging `m<N>` before continuing (`HANDOFF.md` section 12).

**Result:** Not applicable (planning artifact, no measurement yet).

**Decision:** Build in the order M0 → M1 → M2 → M3 → M4 → M5, stopping only on an explicit "Stop and ask
before" condition from `HANDOFF.md` section 11, and recording any spec ambiguity as a numbered decision (D1,
D2, ...) in `STATUS.md`.

## 2026-09-19 15:25  M0: scaffold, `uv` env, config loader, `glance doctor`

**Question:** What device tier does the target machine get, and does the scaffolded harness pass its own
acceptance check?

**What was done:** Repo layout; `pyproject.toml` + committed `uv.lock` (Python 3.11, torch 2.14.0,
transformers 5.17.0); `configs/default.yaml` with every model pinned by revision SHA; `glance/config.py`;
`glance/logging_utils.py` (JSONL writer, ULID-style request ids, timer, once-per-op MPS-fallback logging);
`glance/doctor.py`; the `glance doctor` command. `uv` itself was installed via Homebrew (not present on the
machine). Decisions D1-D3 were recorded in `STATUS.md`: module placement not named in the spec's layout
(`config.py`, `doctor.py`, `pipeline.py`); tier-table gaps resolve to the smaller tier with a warning; Apple
Silicon `vram_gb` is read from `torch.mps.recommended_max_memory()`.

**Result:** `glance doctor --json` exits 0, writes `logs/doctor.json`. Measured: `chip Apple M5 | ram_gb 32.0
| vram_gb 25.0 | device mps | dtype float16 | torch 2.14.0 | free_disk_gb 1279`; `selected_tier apple_32gb`;
`selected_models siglip google/siglip2-base-patch16-256@3f9f96cb, vlm Qwen/Qwen3-VL-4B-Instruct@ebb281ec`;
`image_token_budget 768`; `warnings []`. `pytest: 20 passed`. (`STATUS.md`, M0 section.)

**Decision:** No deviations, no open questions. Proceed to M1. Commit `bb3cb09` ("M0: scaffold, uv env, config
loader, logging utils, glance doctor"), tagged `m0`.

## 2026-09-19 15:35  M1: schema, images, scorer, SigLIP backend, `glance decide`

**Question:** Does the typed request/response schema validate correctly, and can the dual-encoder backend
answer the three bundled sample images end to end?

**What was done:** `schema.py` (pydantic v2 request/answer/error models with field-path-level 422 detail);
`images.py` (path/url/base64 loading, sha256 of original bytes, EXIF transpose, alpha flatten, 2048 px max
side); `prompts.py` (`PROMPT_VERSION = "p1"`); `backends/base.py` + `backends/siglip.py`; `scorer.py`;
`calibration.py` (fit/save/load/apply, exercised starting at M4); `pipeline.py` (`Engine.decide`, call log);
`glance decide`; three CC0 sample images with request files (`samples/README.md`).

**Result:** All three samples returned valid typed JSON on `--model siglip`: `receipt: is_receipt 0.918 |
doc_type receipt (0.998) conf 0.986 | legibility 1.36`; `invoice: is_receipt 0.792 | doc_type receipt (0.738)
conf 0.476` (SigLIP calls the invoice a receipt — a labeled miss); `dog: is_receipt 0.032 | doc_type other
(0.997) conf 0.977 | legibility 1.03`. An invalid request body returned `422` with
`{"code": "validation_error", "detail": [{"path": "questions.q.criteria", "message": "Field required"}]}`.
`pytest: 65 passed` (`GLANCE_TEST_MODELS=1` includes the three samples on real SigLIP2 weights).
(`STATUS.md`, M1 section.)

**Decision:** No deviations, no open questions. Commit `c7a292f`, tagged `m1`.

## 2026-09-19 15:45  M2: VLM backend, reference path, `independent` and `letter`

**Question:** Does the VLM's raw logit readout stay clean (`off_mass < 0.1`), and is the `independent` choice
method already permutation-invariant on the reference (uncached) forward path?

**What was done:** `backends/vlm_hf.py` (Qwen3-VL-4B-Instruct, float16 on MPS, one forward pass per statement,
next-token logits read at the empty assistant turn, `z = logsumexp(Yes variants) - logsumexp(No variants)`,
`off_mass`, asserts on the assistant header and absence of thinking tags, image token budget derived from the
processor's patch/merge sizes); the reference path (full prompts, left-padded, `--no-prefix-cache`);
`Backend.score_labels` for `letter`. The cached path was written but left off (`vlm.prefix_cache: false`)
pending its M4 acceptance check.

**Result** (`GLANCE_TEST_MODELS=1 uv run pytest tests/test_m2_vlm.py -s`, reference path):
`off_mass` over 50 statements in 20 items: mean 0.00000, max 0.00000 (< 0.1 threshold); max `|Δp|` under
option reordering (`independent`): `0.00e+00` (≤ 1e-3 threshold); max `|Δz|` between two identical runs:
`0.00e+00` (≤ 1e-3 threshold); `letter`: receipt → receipt, invoice → invoice, dog → other; all three samples
correct on `--model vlm` (SigLIP had gotten the invoice wrong at M1); image tokens receipt 448, invoice 744,
dog 600 (budget 768), matching manual tokenization. `pytest: 6 passed in 214 s` (model tests) `+ 65 passed`
(unit tests). (`STATUS.md`, M2 section.)

**Decision:** No deviations. MPS float16 ran clean (no NaNs, no logged CPU-fallback ops), so the `mlx-vlm`
4-bit fallback described in `HANDOFF.md` section 3 was never triggered. Open question raised (resolved at
M3/M4): raw probabilities are extremely peaked (`p = 1.000000` on easy items), so M4's calibration has to
absorb that. Commit `3f11679`, tagged `m2`.

## 2026-09-19 16:36  M3: suites, manifests, metrics, plots, report (uncalibrated)

**Question:** With every suite's licensing verified and its dataset recast into typed questions, do the
uncalibrated per-suite eval and report pipelines complete on real data?

**What was done:** `evals/suites/` (`pope`, `gqa_yesno`, `pets37`, `caltech101`, `blur_ladder`, `doctype16`,
`human_gold`) with committed 1,000-item manifests; `evals/metrics.py`; `evals/run.py` (seeded units, timed
10-item warmup with ETA, `--max-hours` trim, failure accounting, permutation sensitivity, latency benchmark);
`evals/report.py` (go/no-go table, per-suite results, `independent` vs `letter`, local vs baseline, top-20
confident errors, reliability/risk-coverage plots); `glance eval`, `glance calibrate`. Every dataset license
was verified against its own card before download and recorded in `DATASETS.md`: POPE rebuilt from the
official MIT repo plus COCO's image host (the Hub mirror carries no license tag); Caltech-101 from the
official CaltechDATA record, CC BY 4.0 (the Hub mirror says `unknown`); `doctype16` **skipped** (RVL-CDIP's
cards say `other`); `human_gold` skipped (empty). `blur_ladder`'s 8 example renders were checked against the
level descriptions before its blur radii (0, 1.6, 4.0, 10.0 px at 384 px longest side) were frozen.

**Result:** Check run `uv run glance eval --suite pope --n 200 --model vlm` → `runs/20260919T225415Z-b07b50`
(22:54:15 UTC): `pope/vlm` test n=100, acc 0.890, AUROC 0.931, NLL 1.788, ECE 0.102, selective accuracy
50/80/90/100 = 0.98/0.925/0.911/0.89; 0 failures; `off_mass` max 2e-6; latency (1 image + 5 questions,
reference path) p50 8.7 s. Check run `uv run glance eval --suite pets37 --n 50` →
`runs/20260919T230914Z-dac969` (23:09:14 UTC): siglip independent acc 0.96; vlm independent 0.92 over all 37
options / 0.96 restricted to letter's 26; vlm letter 0.96; permutation (75 reordered requests each):
independent max `|Δp|` = `0.0e+00`, letter max `|Δp|` = `9.7e-01`; latency p50 siglip 23 ms, vlm 8,958 ms.
`pytest: 93 passed, 11 skipped` (model-gated). (`STATUS.md`, M3 section; cross-checked against
`results/v0/m3_pope_n200/report.md` and `results/v0/m3_pets37_n50/report.md`.)

**Decision:** Deviation: harness version bumped to 0.2.0 because the VLM readout was changed mid-milestone to
compute Yes/No logits through a float32 copy of the output head (float16 quantizes a logit near 20 to steps of
1/64). Open question: `letter` stayed strongly order-sensitive even with 4 rotations — the argument for
`independent` as the default. Commit `58bb3b5`, tagged `m3`.

## 2026-09-19 17:26  M4: calibration fit/apply, prefix cache, Flask server

**Question:** Does post-hoc calibration lower ECE on every suite's test split, does the prefix-cached VLM
forward path meet its accuracy-preservation acceptance criterion, and does the local HTTP server round-trip
correctly?

**What was done:** Calibration wired into eval runs (fit on the calibration split, pooled per question type
plus per-suite fits for comparison, saved per configuration key, applied to the test split,
`calibration_mismatch` on any key mismatch); `glance calibrate --run`; the prefix-cached VLM path with mRoPE
position continuation and a 2-entry prefix LRU; `glance/server.py` (`POST /v1/decide`, `GET /v1/models`,
`GET /healthz`, bound to `127.0.0.1` only); `glance serve`.

**Result — Check 1, calibration** (`glance eval --n 80 --model siglip --model vlm --skip-permutation` →
`runs/20260919T234550Z-796ade`, 23:45:50 UTC, test n=40/suite, reference path): calibrated ECE was below raw
ECE for every suite on both backends except one flagged unit — `caltech101` (vlm, letter): 0.025 → 0.027.
vlm ECE raw→cal: pope 0.145→0.088, gqa_yesno 0.241→0.224, pets37 0.102→0.083, caltech101 0.015→0.011,
blur_ladder 0.391→0.253. siglip: pope 0.332→0.230, pets37 0.059→0.035, caltech101 0.021→0.015,
blur_ladder 0.286→0.268. Fitted (vlm): noul Platt a=0.162 b=0.163, choice T=2.06 (letter T=3.40), score T=5.92.
At n=40 the ECE sampling floor was as large as the measured value for `blur_ladder` (0.253 vs. floor 0.252),
so the 0.05 gate could not yet be judged reliably.

**Result — Check 2, prefix cache** (`tests/test_m4_prefix_cache.py`, 100 items / 1,685 statements,
`logs/cache_check.json`, snapshot `results/v0/prefix_cache_acceptance.json`): argmax agreement 100/100, max
`|Δz|` 0.075 (limit 0.05), median `|Δz|` 0.017, speedup 3.75x (309.7 s → 82.7 s); per-suite max `|Δz|`: pope
0.043, gqa_yesno 0.016, blur_ladder 0.052, pets37 0.075, caltech101 0.056. Reference path against itself
(batch size 8 vs. 4, same 10 items): max `|Δz|` 0.093, median 0.046 — larger than the cache-vs-reference
drift.

**Result — Check 3, server:** `tests/test_m4_calibration_server.py` round-trip passed; a real `glance serve
--preload siglip` process with `HF_HUB_OFFLINE=1` answered `/healthz`, `/v1/models`, `POST /v1/decide` (200 in
200 ms), returned 422 for a bad body, refused `model: frontier`, and listened on `127.0.0.1:8077` only.
`pytest: 95 passed, 11 skipped`. (`STATUS.md`, M4 section.)

**Decision:** The section 6 acceptance was **not met** (0.075 > 0.05 limit). Two fixes were tried: (1) the
float32 readout head (kept — "strictly better", residual drift 0.04-0.09); (2) eager attention with float32
softmax instead of SDPA (not kept — no improvement, 0.06-0.11, and slower). Per `HANDOFF.md` section 6 ("after
two failed approaches, ship uncached and report"), the prefix cache ships **off by default**
(`vlm.prefix_cache: false`), available opt-in via `--prefix-cache`. Commit `55177cc`, tagged `m4`.

## 2026-09-19 17:33  Report refinements: ECE sampling floor, error breakdown, opt-in stretch suites

**Question:** How should a 0.05 ECE gate be judged reliably when a suite's test split is only a few hundred
items, given that equal-mass binned ECE is known to be biased upward at small n?

**What was done:** Added `ece_noise_floor` (D23): for each unit, simulate 200 draws of `correctness ~
Bernoulli(confidence)` using the test split's own observed confidences (i.e. what a *perfectly calibrated*
predictor with the same confidence values would measure), and report the mean simulated ECE next to every
measured ECE. Also added the per-suite error-breakdown table, opt-in stretch suites (`pope_injection`,
`pets37_openset`), and the `--image-token-budget` override flag used later for the token sweep.

**Result:** Infrastructure change; no new accuracy/calibration numbers at this commit (`STATUS.md` describes
this work folded into the M4 narrative; commit `2ea52c5`, "Report: ECE sampling floor, error breakdown; opt-in
stretch suites; token budget flag", not separately tagged).

**Decision:** Report every ECE next to its sampling floor rather than relaxing the 0.05 gate itself.

## 2026-09-19 21:45  M5: frontier baseline adapter, full evaluation, calibration params, final report

**Question:** With the full default suite set at Apple-Silicon scale (n=500 before trimming), does the go/no-
go table fill in cleanly, and what does the evidence say about each question type?

**What was done:** `backends/frontier.py` (LiteLLM, JSON schema enumerating allowed answers, temperature 0
with a fallback for providers that reject it, model id from `FRONTIER_MODEL` logged exactly, picks never
written to disk, refused without `--confirm-spend`, refused on `human_gold` without `--allow-upload-gold`,
cost estimate printed first, skipped cleanly without a key); per-suite `--max-hours` trim (D22); the ECE
sampling floor now applied throughout; error breakdown; `glance calibrate` output committed under
`calibration/`.

**Result:** Check `glance eval --permutation-items 30` → `runs/20260920T002639Z-2091d3` (00:26:39 UTC,
2026-09-20): report opens with the go/no-go table, every row filled with threshold/measured/pass-or-FAIL/not-
measured. 5,326 requests, 0 failures, 4 h 10 min wall-clock. `glance calibrate --run 20260920T002639Z-2091d3`
wrote `calibration/{8edfd9f546b5,1d941800c4af,eac746e4859c}.json`. `pytest: 95 passed, 11 model-gated
skipped` (M2 checks re-run and passing). The frontier baseline did **not** run: no `FRONTIER_MODEL` or API key
was present. Go/no-go: ECE gate FAILed on 3 of 5 suites (pope 0.066 vs. floor 0.040; gqa_yesno 0.093 vs. floor
0.074; blur_ladder 0.149 vs. floor 0.097; pets37 0.038 and caltech101 0.034 passed); permutation invariance
passed (`0.0e+00` on 180 reordered requests); latency recorded (p50 8,604 ms reference / 1,265 ms with
`--prefix-cache`); accuracy-gap and selective-accuracy-vs-baseline rows read "not measured" for lack of a
frontier run at this point in time. Full numbers: `docs/paper/RESULTS_V0.md` "M5: full evaluation" section.
(`STATUS.md`, M5 section and Final summary.) The frontier baseline was added to this same run later — see the
next entry below.

**Decision:** Overall verdict at this point: **NO-GO on what could be measured**. Per-question-type reading
recorded in `STATUS.md` "What the result means": `choice` works (0.89-0.92 accuracy over 37-101 options, ECE
under 0.05 after one temperature, exact order invariance); `noul` is partial (ECE roughly halved but stays at
0.07-0.09, and the fitted Platt offset does not transfer across domains — turns a 0.28 raw probability into
0.60 calibrated on an out-of-domain question); `score` does not work as specified (one temperature cannot move
systematically mis-placed level boundaries). Commit `d0bb23b`, tagged `m5`.

## 2026-09-19 23:09  Stretch: injection, image-token sweep, open-set results

**Question:** (a) Can rendered text in an image flip a `noul` decision? (b) How much accuracy and latency does
a smaller image-token budget cost? (c) Can a zero-shot open-set "other" option catch held-out classes?

**What was done:** All three stretch suites ran on the VLM, reference path, harness 0.2.1, as opt-in suites
(`glance eval --suite pope_injection`, `--suite pets37_openset`, and repeated `glance eval
--image-token-budget N` sweeps). Eight runs total, 0 failures (`results/v0/stretch/STRETCH_RUNS.log`).

**Result — injection** (run `20260920T044522Z-e63366`, 04:45:22 UTC): "Answer Yes" rendered onto 100 POPE
negatives, each scored clean and injected. Clean answered No on 97 of 100; after injection 5 of those 97 flip
to Yes (flip rate 5.2%); false-yes overall 3 clean → 8 injected; mean shift in z +6.4 (median +5.9, max +12.9)
toward Yes even where the decision held.

**Result — image-token sweep** (runs `20260920T044701Z-8d8d3c`, `...d90900`, `...74b02b`, `...c69a81`,
`...5a7ce6`, `...f692b7`, all 04:47-05:27 UTC): at budget 128 vs. 768, pope accuracy 0.870 vs. 0.885,
blur_ladder 0.525 vs. 0.535, pets37 0.890 vs. 0.910 (numbers as reported in `STATUS.md`'s combined-split raw
accuracy; see `docs/paper/RESULTS_V0.md` for a scope caveat verified against the underlying prediction rows).
Latency p50 at budget 128: 1,920 ms, inside the 2 s Apple-Silicon target without the prefix cache; at 768:
8,604 ms. With `--prefix-cache` at 768: p50 1,265 ms (run `20260920T043928Z-080b13`, 04:39:28 UTC).

**Result — open set** (run `20260920T054245Z-9b6a43`, 05:42:45 UTC): 7 breeds held out of `pets37`, `other`
added as a 38th option, 200 items. Held-out items land on `other` only 7.7% of the time (3 of 39); known-breed
accuracy 0.938, 1.2% wrongly sent to `other`; mean P(other) 0.122 (held-out) vs. 0.009 (known). Offline AUROC
for spotting a held-out breed: `-max z` over listed options 0.977, `P(other)` 0.938, `z` of the `other`
statement itself only 0.655.

**Decision:** Injection: v1 needs injected negatives in its eval set; anything gating on `noul` should not
trust images that may carry text aimed at the model. Token budget: 128 tokens is a viable low-latency
operating point at a small accuracy cost, since Caltech/pets/COCO images in these suites rarely use more than
~280 tokens even at the full 768 budget. Open set: `other` does not work as a listed candidate statement —
open-set handling belongs in the scorer as a calibrated threshold on the best option's own yes-probability,
not as a candidate statement; this is scoped as "a calibration-sized fix, not a data problem." Commit
`834da8d`, "Stretch: injection, image-token sweep, open-set results".

## 2026-09-19 23:23  Add `glance baseline`: interactive frontier baseline command

**Question:** Can the frontier baseline be run without asking the user to put an API key in a `.env` file on
disk?

**What was done:** `glance/evals/baseline.py` and the `glance baseline` CLI command (D25, added at the user's
request, not in `HANDOFF.md`): finds the largest finished eval run, prompts for provider/model and the API key
with hidden terminal input, runs one smoke-test call before showing a cost estimate, waits for an interactive
`y` before spending further, appends frontier rows to the existing run so they are compared against the local
backends on exactly the same items, and rebuilds that run's report. The key lives only in memory for the
process's lifetime and is never written to disk, a log, or shell history; provider error text is scrubbed of
anything key-shaped before being logged (D25). D26 was also recorded: since current frontier models (including
some Claude models) reject an explicit `temperature=0`, the adapter retries once without it and reports the
switch as a warning.

**Result:** Not run against a real provider in this session — this is a capability addition, awaiting the
user to supply a model choice and API key interactively (`STATUS.md` Final summary, "Open inputs from you",
item 1: "Paste `uv run glance baseline` into a terminal to fill rows 1 and 3. It asks for the model and the
API key itself... I did not and will not handle keys.").

**Decision:** Ship the command; leave the two baseline-dependent go/no-go rows as "not measured" until the
user runs it. Commit `f12360a`, "Add glance baseline: interactive frontier baseline with an in-memory key, no
.env".

## 2026-09-20  Frontier baseline added to the full evaluation

**Question:** With a frontier baseline now available, do the two remaining go/no-go rows (accuracy gap,
selective accuracy vs. baseline) pass, and how close is the 4B open model to a frontier model in practice?

**What was done:** The user ran `uv run glance baseline` (D25) against the M5 run
`20260920T002639Z-2091d3`. Model: `anthropic/claude-opus-5` through LiteLLM, JSON-schema-constrained picks,
test split only, up to 300 items per suite. The adapter's temperature fallback (D26) applied. Picks were not
stored, only whether each was right; the run's report was rebuilt afterward. The completed run was then
committed as a frozen snapshot at `results/v0/m5_full_eval/` (`report.md`, `metrics.json`, `summary.txt`,
`config.yaml`, `env.json`, `extras.json`, `plots/`, `calibration/`, gzipped predictions).

**Result:** 1,221 baseline calls (250 per suite except `caltech101`'s 221), 0 failures, about 2.4 s per call.
Go/no-go table completed (`results/v0/m5_full_eval/metrics.json: go_no_go`): accuracy gap **+3.1 points**
macro-averaged over 5 suites (pass, ≤5 threshold) — per suite: pope +3.6, gqa_yesno -1.2 (the 4B model ahead
here), pets37 +4.0, caltech101 +5.0, blur_ladder +4.0. Selective accuracy at 80% coverage **0.841 vs. baseline
full-coverage accuracy 0.818** (pass). Permutation invariance and latency rows unchanged from the original M5
check. ECE gate still FAILs on 3 of 5 suites (pope 0.066, gqa_yesno 0.093, blur_ladder 0.149, all over 0.05).
Overall verdict (`results/v0/m5_full_eval/metrics.json: verdict`): **NO-GO**, on the ECE gate alone — three of
the four judged gates now pass. Notably, the frontier model itself scored only **0.536** on `blur_ladder`
(`results/v0/m5_full_eval/report.md`), and SigLIP2 (375M parameters) **beat** the frontier model on `pets37`
(0.956 vs. 0.932). (`STATUS.md`, "Update 2026-09-20: frontier baseline added to the full evaluation" section.)

**Decision:** The v0 mission question reads differently with the baseline in hand: for `choice` and, more
narrowly, `noul`, the 4B open model plus calibration is close enough to a frontier model (within the 5-point
accuracy-gap gate, and ahead of it on selective accuracy) that further work there is a calibration problem, not
a data problem, exactly as the mission asked. `score` is the exception on both counts: it fails its own ECE
gate and both models — 4B open and frontier — struggle with the same 4-level task, which points toward the
task formulation itself, not only the small model, being the next thing to fix (this is the stated motivation
for the score lab, next entry).

## 2026-09-20  Score lab began on branch `score-lab`

The v0 evaluation above identified `score` (ordinal-scale) questions as the weakest result: `blur_ladder`
reached accuracy 0.496, MAE 0.67 levels, calibrated ECE 0.149, with a systematic one-step-too-blurry boundary
error rather than random noise (`docs/paper/RESULTS_V0.md` M5 section; `STATUS.md` Final summary). A separate,
in-progress experiment — the "score lab" — started on this date on branch `score-lab` to explore alternative
prompting and readout methods for ordinal-scale questions without any model training: `independent` (the v0
method, unchanged), `cumulative` (show the whole scale, ask K-1 "is it step k or higher?" questions),
`digits` (show a numbered scale, read next-token logits over the digits in one pass), and
`anchors_cumulative` / `anchors_digits` (the same two, with reference images of known level included in the
request). Method definitions live in `glance/lab/score_methods.py`; supporting modules are
`glance/lab/collect.py`, `glance/lab/ladders.py`, `glance/lab/analyze.py`, `glance/lab/extras.py`. Per the
scope of this documentation pass, the score lab's *results* are not documented here — only that it started, on
this date, on this branch, with these method definitions. Its own running notebook is `lab/NOTES.md`, whose
format mirrors this log (dated entries, Question/hypotheses registered before data collection, per its own
first entry).

Later entries in this file, if any, are appended by hand by the experimenter as work continues on the main
branch; the score-lab work itself is logged separately in `lab/NOTES.md`.

## 2026-09-20  Documentation pass, review, and a dropped side analysis

**Question.** Can the repository carry a paper on its own?

**What was done.** Everything in `docs/` was backfilled from `STATUS.md`, the code and the committed result snapshots
by a cheaper assistant model, then reviewed by the experimenter. `runs/` is gitignored, so every finished run was
first copied into `results/` with `tools/snapshot_run.py` (absolute paths rewritten to relative, predictions
gzipped), and the ad-hoc probe and analysis scripts used during v0 were saved under `tools/`.

**Result.** `tools/verify_docs_numbers.py` confirms that every three-decimal number in `docs/` appears in a committed
source. The review still found sentence-level errors that a number check cannot catch, all in `OUTLINE.md`, now
corrected: "two of five suites exceed the ECE gate" (three do), "`independent` matches or exceeds `letter`" (0.973 vs
0.977 on caltech101, so "on par"), "calibration halves ECE" (reductions are 34-54% on the binary and multiple-choice
suites), "a 128-token budget nearly halves latency" (8,604 ms to 1,920 ms), and "confident POPE negatives" (the 97
negatives the model had answered No on).

**Decision.** Treat `docs/` as a draft that must be re-checked claim by claim before submission; numbers are
traceable, wording is not guaranteed. An escalation ("answer locally when confident, send the rest to the frontier
model") analysis was started and then dropped at the project owner's direction; its script and plots were removed
to keep the repository focused on the model itself.

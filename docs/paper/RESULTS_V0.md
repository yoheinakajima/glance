# Results: v0

All numbers below are copied from the cited files. Where a number appears in both `STATUS.md` and a committed
result snapshot under `results/v0/`, both are cited and they were cross-checked against each other as part of
this documentation pass (see the final report-back for the list of spot-checked numbers and any discrepancy
found).

**Full v0 evaluation run, now complete.** `runs/20260920T002639Z-2091d3` (the M5 full-eval run) was, for part
of this documentation pass, still being extended by a frontier-baseline process, so an earlier draft of this
document restricted every number from that run to what `STATUS.md` had already recorded. The frontier baseline
(`anthropic/claude-opus-5`, 1,221 calls, 0 failures) has since finished and the completed run is committed as
a frozen snapshot at `results/v0/m5_full_eval/` (`report.md`, `metrics.json`, `summary.txt`, `config.yaml`,
`env.json`, `extras.json`, `plots/`, `calibration/`, gzipped predictions). Every number below cites either
that snapshot or `STATUS.md`'s "M5", "Final summary", and "Update 2026-09-20: frontier baseline added to the
full evaluation" sections; this document still does not read `runs/20260920T002639Z-2091d3/predictions.jsonl`
directly (only the committed snapshot's gzipped copy was consulted, and only where cited). The three post-hoc
calibration files `calibration/1d941800c4af.json`, `calibration/8edfd9f546b5.json`,
`calibration/eac746e4859c.json` at the repo root, and the identical copies under
`results/v0/m5_full_eval/calibration/`, are the frozen output of `glance calibrate --run
20260920T002639Z-2091d3`; both locations agree with each other and with `STATUS.md` everywhere checked.

## M0: scaffold and device detection

Source: `STATUS.md`, M0 section.

```text
chip Apple M5 | ram_gb 32.0 | vram_gb 25.0 | device mps | dtype float16 | torch 2.14.0 | free_disk_gb 1279
selected_tier apple_32gb
selected_models  siglip google/siglip2-base-patch16-256@3f9f96cb  vlm Qwen/Qwen3-VL-4B-Instruct@ebb281ec
image_token_budget 768 | warnings []
pytest: 20 passed
```

## M1: schema, images, scorer, SigLIP backend

Source: `STATUS.md`, M1 section.

| Sample | is_receipt | doc_type (confidence) | legibility |
| --- | --- | --- | --- |
| receipt.jpg | 0.918 | receipt (0.998), conf 0.986 | 1.36 |
| invoice.jpg | 0.792 | receipt (0.738), conf 0.476 — SigLIP calls the invoice a receipt | 1.76 |
| dog.jpg | 0.032 | other (0.997), conf 0.977 | 1.03 |

`image_tokens 256, forward_passes 8` (receipt). Invalid body → `{"code": "validation_error", "detail":
[{"path": "questions.q.criteria", "message": "Field required"}]}`. `pytest: 65 passed`
(`GLANCE_TEST_MODELS=1` includes the three samples on real SigLIP2 weights).

## M2: VLM backend, reference path, `independent`/`letter`

Source: `STATUS.md`, M2 section.

```text
off_mass over 50 statements in 20 items: mean=0.00000 max=0.00000      (< 0.1 threshold)
max |dp| under option reordering (independent): 0.00e+00               (<= 1e-3 threshold)
max |dz| between two identical runs: 0.00e+00                          (<= 1e-3 threshold)
letter: receipt -> receipt, invoice -> invoice, dog -> other
samples on --model vlm: receipt/invoice/dog all correct (SigLIP got the invoice wrong)
image tokens: receipt 448, invoice 744, dog 600 (budget 768)
pytest: 6 passed in 214 s (model tests), 65 passed (unit tests)
```

No deviations. MPS float16 ran clean (no NaNs, no CPU-fallback ops logged), so the `mlx-vlm` fallback planned
in `HANDOFF.md` section 3 was never needed.

## M3: suites, manifests, metrics, plots, report (uncalibrated)

### `pope`, n=200

Source: `results/v0/m3_pope_n200/report.md` (run `20260919T225415Z-b07b50`), cross-checked against
`STATUS.md` M3 section.

| Suite | Backend | Method | Test n | Acc | AUROC | NLL | Brier | ECE | Sel acc 50/80/90/100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pope | vlm | statement | 100 | 0.890 | 0.931 | 1.788 | 0.108 | 0.102 | 0.980 / 0.925 / 0.911 / 0.890 |

0 of 200 failures. `off_mass` mean 0.00000, p95 0.00000. Latency (1 image + 5 questions, reference path):
p50 8,708 ms, p95 11,029 ms (`results/v0/m3_pope_n200/report.md`; `STATUS.md` gives p50 8.7 s for the same
check). Most common confusion: yes → no (11 of the 11 test errors).

### `pets37`, n=50, both choice methods

Source: `results/v0/m3_pets37_n50/report.md` (run `20260919T230914Z-dac969`), cross-checked against
`STATUS.md` M3 section.

| Suite | Backend | Method | Test n | Acc | AUROC | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| pets37 | siglip | independent | 25 | 0.960 | 0.917 | 0.041 |
| pets37 | vlm | independent | 25 | 0.920 | 0.765 | 0.051 |
| pets37 | vlm | letter | 25 | 0.960 | 0.882 | 0.039 |

`independent` restricted to letter's same 26-of-37 option subset also reaches 0.960 (`independent (same
options)` column, `results/v0/m3_pets37_n50/report.md`), matching `STATUS.md`'s "vlm independent 0.92 (all 37)
/ 0.96 (letter's 26)". Permutation sensitivity (75 reordered requests per unit):

| Unit | max abs Δp | mean abs Δp | Choice flips |
| --- | --- | --- | --- |
| pets37 · siglip · independent | 0.0e+00 | 0.0e+00 | 0 |
| pets37 · vlm · independent | 0.0e+00 | 0.0e+00 | 0 |
| pets37 · vlm · letter | 9.7e-01 | 4.1e-02 | 3 |

Latency p50: siglip 23 ms, vlm independent 9,621 ms, vlm letter 1,858 ms (per-unit throughput table,
`results/v0/m3_pets37_n50/report.md`); the 1-image-5-question benchmark gives vlm p50 8,958 ms, p95 11,247 ms.

**Deviation noted at this milestone:** the harness version was bumped to 0.2.0 because the VLM readout was
changed to compute the Yes/No logits through a float32 copy of the output head (`STATUS.md`, M3 "Deviations":
"float16 quantizes a logit near 20 to steps of 1/64"). **Open question raised:** `letter` is strongly order-
sensitive even with 4 rotations, which is the stated argument for `independent` (`STATUS.md`, M3 "Open
questions").

## M4: calibration, prefix cache, server

### Calibration check, n=80 per suite (test n=40)

Source: `results/v0/m4_calibrated_n80/report.md` (run `20260919T234550Z-796ade`), cross-checked against
`STATUS.md` M4 section.

| Suite | Backend | Method | Test n | Acc raw | ECE raw → cal | ECE floor at n |
| --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 40 | 0.325 | 0.286 → 0.268 | 0.231 |
| blur_ladder | vlm | statement | 40 | 0.525 | 0.391 → 0.253 | 0.252 |
| caltech101 | siglip | independent | 40 | 0.975 | 0.021 → 0.015 | 0.019 |
| caltech101 | vlm | independent | 40 | 0.950 | 0.015 → 0.011 | 0.034 |
| caltech101 | vlm | letter | 40 | 0.975 | 0.025 → 0.027 | 0.008 |
| gqa_yesno | vlm | statement | 40 | 0.725 → 0.800 | 0.241 → 0.224 | 0.200 |
| pets37 | siglip | independent | 40 | 0.950 | 0.059 → 0.035 | 0.031 |
| pets37 | vlm | independent | 40 | 0.875 | 0.102 → 0.083 | 0.059 |
| pets37 | vlm | letter | 40 | 0.875 | 0.115 → 0.105 | 0.071 |
| pope | siglip | statement | 40 | 0.600 → 0.750 | 0.332 → 0.230 | 0.196 |
| pope | vlm | statement | 40 | 0.850 | 0.145 → 0.088 | 0.111 |

At this n (40 test items per suite), the ECE sampling floor is comparable to the measured value for
`blur_ladder` (0.253 measured vs 0.252 floor), so the 0.05 gate could only be judged reliably on the larger M5
run (`STATUS.md`: "the ECE sampling floor is as large as the measured values ... so the 0.05 gate can only be
judged on the larger M5 run").

**Flagged**: calibrated ECE was not below raw ECE for `caltech101` (vlm, letter: 0.025 → 0.027)
(`results/v0/m4_calibrated_n80/report.md`).

**Fitted parameters** (pooled, this n=80 run; source: `results/v0/m4_calibrated_n80/calibration/1d941800c4af.json`,
which matches `results/v0/m4_calibrated_n80/report.md`'s Calibration table and `STATUS.md`'s "fitted (vlm)"
line):

| Backend | Choice method | Type | Method | Params | n (cal split) | NLL before → after | ECE before → after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlm | independent | noul | platt | a=0.1623, b=0.1632 | 80 | 1.101 → 0.405 | 0.163 → 0.109 |
| vlm | independent | choice | temperature | T=2.0626 | 80 | 0.213 → 0.160 | 0.043 → 0.035 |
| vlm | independent | score | temperature | T=5.924 | 40 | 2.449 → 1.003 | 0.265 → 0.328 |
| vlm | letter | choice | temperature | T=3.3955 | 80 | 0.401 → 0.145 | 0.036 → 0.034 |
| siglip | independent | noul | platt | a=0.8196, b=1.652 | 40 | 0.832 → 0.525 | 0.331 → 0.236 |
| siglip | independent | choice | temperature | T=0.5658 | 80 | 0.103 → 0.077 | 0.039 → 0.021 |
| siglip | independent | score | temperature | T=4.7167 | 40 | 1.698 → 1.330 | 0.345 → 0.299 |

`STATUS.md`'s summary line "fitted (vlm): noul Platt a=0.162 b=0.163 (raw logits ~6x too sharp), choice T=2.06
(letter T=3.40), score T=5.92" matches these to the reported precision.

### Prefix cache acceptance check

Source: `results/v0/prefix_cache_acceptance.json`, cross-checked against `STATUS.md` M4 section (Check 2)
and `results/v0/stretch/*/report.md` "Prefix cache check" lines, which all quote the same figures.

| Metric | Value | Limit |
| --- | --- | --- |
| Items | 100 | — |
| Statements | 1,685 | — |
| Argmax agreement | 100/100 | — |
| Max \|Δz\| | 0.075 | 0.05 |
| Median \|Δz\| | 0.017 | — |
| Speedup | 3.75x (309.7 s → 82.7 s) | — |

Per-suite max `|Δz|`: pope 0.043, gqa_yesno 0.016, blur_ladder 0.052, pets37 0.075, caltech101 0.056
(`results/v0/prefix_cache_acceptance.json: per_suite.*.max_abs_dz`).

**The section 6 acceptance was NOT met** (max `|Δz|` 0.075 > the 0.05 limit), so per `HANDOFF.md` section 6
("after two failed approaches, ship uncached and report") the cache ships **off by default**
(`vlm.prefix_cache: false` in `configs/default.yaml`), available opt-in via `--prefix-cache`.

**Reference path's own batch-size self-noise**, measured on the same 10 items, batch size 8 vs 4:
max `|Δz|` 0.093, median `|Δz|` 0.046 (`results/v0/prefix_cache_acceptance.json:
reference_self_noise_bs8_vs_bs4`). This exceeds the cached-path-vs-reference drift (0.075), i.e. the
uncached reference path disagrees with *itself* under a pure batch-shape change by more than the cached path
disagrees with the reference — evidence the remaining drift is float16 matmul kernel behavior on MPS, not a
correctness bug in the cache design (`STATUS.md` M4 section).

**Two attempted fixes** (`STATUS.md` M4 section):

1. **Float32 readout head** (kept — described as "strictly better", now used on both forward paths; see
   `docs/paper/METHODS.md` section 2). Float16 quantized a logit near 20 to steps of 1/64, which alone moved
   `z` by up to 0.06. Residual drift after this fix stayed at 0.04-0.09.
2. **Eager attention** with float32 softmax instead of SDPA (not kept). No improvement (0.06-0.11 residual
   drift) and slower. Probe script: `tools/probe_eager_attention.py`.

No decision changed on any of the 100 acceptance-check items despite the drift (`STATUS.md`). With the cache
on, the 1-image-5-question latency benchmark measured p50 1,265 ms / p95 1,462 ms versus p50 ~8.9 s uncached
(`results/v0/stretch/latency_with_prefix_cache/report.md`, run `20260920T043928Z-080b13`), i.e. roughly the
3.75x eval-time speedup carries over to single-request latency.

### Server round-trip

Source: `STATUS.md` M4 section, Check 3. `tests/test_m4_calibration_server.py` round-trip passed; a real
`glance serve --preload siglip` process with `HF_HUB_OFFLINE=1` answered `/healthz`, `/v1/models`,
`POST /v1/decide` (200 in 200 ms), returned 422 for a bad body, refused `model: frontier`, and listened on
`127.0.0.1:8077` only. `pytest: 95 passed, 11 skipped (model-gated)`.

## M5: full evaluation, calibration, final report

Run `20260920T002639Z-2091d3`: 5,326 local-backend requests, 0 failures, 4 h 10 min wall-clock
(`STATUS.md` M5 section). The frontier baseline was added afterward (see next subsection); the run is now
complete and frozen as `results/v0/m5_full_eval/` (`report.md`, `metrics.json`, `summary.txt`, `config.yaml`,
`env.json`, `extras.json`, `plots/`, `calibration/`, gzipped predictions). Numbers in this section cite that
snapshot and/or `STATUS.md`.

### Frontier baseline addition

Source: `STATUS.md`, "Update 2026-09-20: frontier baseline added to the full evaluation" section, cross-checked
against `results/v0/m5_full_eval/report.md` and `results/v0/m5_full_eval/metrics.json`. The user ran
`uv run glance baseline` (D25) against this run. Baseline model: `anthropic/claude-opus-5` through LiteLLM,
JSON-schema-constrained picks, test split only, **1,221 calls** (250 per suite except caltech101's 221, which
matches that suite's `--max-hours`-trimmed test-split size), **0 failures**, about 2.4 s per call
(`STATUS.md`). The adapter's temperature fallback (D26) applied. Picks are not stored, only whether each was
right (`docs/paper/METHODS.md` section 11).

### Go/no-go table

Source: `results/v0/m5_full_eval/metrics.json: go_no_go, verdict`, identical to `STATUS.md`'s "Update
2026-09-20" section and `results/v0/m5_full_eval/report.md`'s opening table.

| Metric | Threshold | Measured | Pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | ≤ 5 points, macro-averaged | **+3.1 points over 5 suites** (Claude Opus 5 ahead on average) | **pass** |
| ECE after calibration | ≤ 0.05 per suite, 15 equal-mass bins | worst 0.149 (blur_ladder, sampling floor 0.097); over threshold: blur_ladder, gqa_yesno, pope | FAIL |
| Selective accuracy at 80% coverage | ≥ baseline full-coverage accuracy | **0.841 vs baseline 0.818** (macro over 5 suites) | **pass** |
| Permutation invariance (choice, independent) | max shift ≤ 1e-3 | 0.0e+00 (pets37) | pass |
| Latency, 1 image + 5 questions | recorded; target ≤ 2 s on Apple Silicon | p50 8,604 ms / p95 10,817 ms (reference path, shipped default); p50 1,265 ms / p95 1,462 ms with `--prefix-cache`; SigLIP 23 ms | recorded |

**Verdict: `results/v0/m5_full_eval/metrics.json: verdict` = "NO-GO"** — still NO-GO by the letter of the
go/no-go rule (any failing gate is a NO-GO), but on the ECE gate alone: three of the four judged gates now
pass (`STATUS.md`: "Verdict: still NO-GO by the letter, on the ECE gate alone. Three of the four gates
pass.").

### Per-suite comparison against the frontier baseline

Source: `results/v0/m5_full_eval/metrics.json: baseline` (full precision) and
`results/v0/m5_full_eval/report.md` "Local against the frontier baseline" table (matches `STATUS.md`'s
"Update 2026-09-20" per-suite table to 3 decimal places).

| Suite | n | Claude Opus 5 acc | SigLIP2 acc | Qwen3-VL-4B acc | Qwen gap (points) | Qwen sel. acc @80% |
| --- | --- | --- | --- | --- | --- | --- |
| pope | 250 | 0.920 | 0.732 | 0.884 | +3.6 | 0.965 |
| gqa_yesno | 250 | 0.732 | not run (no criteria for the dual encoder) | 0.744 | -1.2 | 0.775 |
| pets37 | 250 | 0.932 | 0.956 | 0.892 | +4.0 | 0.960 |
| caltech101 | 221 | 0.968 | 0.932 | 0.919 | +5.0 | 0.972 |
| blur_ladder | 250 | **0.536** | 0.344 | 0.496 | +4.0 | 0.535 |

"Gap (points)" is `frontier accuracy - local accuracy`, positive means the frontier model is ahead; this
matches `results/v0/m5_full_eval/report.md`'s "Local against the frontier baseline" table exactly (e.g. pope
vlm: `92.0 - 88.4 = +3.6`). The macro mean of the vlm gap column (+3.6, -1.2, +4.0, +5.0, +4.0) is
`+3.08 ≈ +3.1` points, matching the go/no-go row above. SigLIP2 (the 375M-parameter dual encoder) **beats**
the frontier model on fine-grained pet-breed classification (0.956 vs. 0.932, gap -2.4) and on POPE object
presence at a much larger negative gap (`+18.8`, i.e. far behind on `pope`) — the dual encoder is far weaker on
`pope` and `blur_ladder` (gaps +18.8 and +19.2) but competitive to better on `pets37` and `caltech101` (gaps
-2.4 and +3.6).

**The frontier model itself scored 0.536 on `blur_ladder`** (`results/v0/m5_full_eval/report.md`: `blur_ladder
| frontier | pick | 250 | 0.536`), barely above the 4B open model's 0.496 and far from a ceiling. This is
direct evidence that the systematic level-boundary error on this suite (see "Weakest suite" below) is at least
partly a property of the 4-level ordinal task as posed, not solely a small-model capability gap (`STATUS.md`:
"the frontier model is also poor at the 4-level blur rating (0.536), so the `score` weakness is a property of
the task as posed, not only of the small model").

### `independent` vs `letter` (same 26 options)

Source: `results/v0/m5_full_eval/report.md` "`independent` against `letter`" table, matching `STATUS.md`
Final summary.

| Suite | Acc independent (same options) | Acc letter | ECE cal independent | ECE cal letter | p50 ms independent | p50 ms letter |
| --- | --- | --- | --- | --- | --- | --- |
| pets37 | 0.912 | 0.904 | 0.038 | 0.037 | 8,422 | 1,722 |
| caltech101 | 0.973 | 0.977 | 0.034 | 0.023 | 12,316 | 981 |

`independent` over all options (not restricted to letter's 26): pets37 0.892 (37 options), caltech101 0.919
(101 options) (`results/v0/m5_full_eval/report.md`, "Acc independent (all options)" column, matching the
per-suite table above). `letter` moves by up to 0.97 in probability when options are reordered (3 choice flips
in 90 reordered requests, `pets37 · vlm · letter` row); `independent` moves by exactly 0 on all six
suite/backend/method permutation units run at M5, including `caltech101 · vlm · independent` and
`caltech101 · siglip · independent` (`results/v0/m5_full_eval/report.md`, "Permutation sensitivity" table —
this table is fuller than the single `pets37` row `STATUS.md`'s prose quotes, and confirms `0.0e+00` on every
`independent` unit, not only `pets37`).

### Weakest suite and calibration gain

Source: `results/v0/m5_full_eval/report.md` per-suite table and `STATUS.md` Final summary (both agree).

Weakest suite: `blur_ladder` (score), accuracy 0.496, MAE 0.67 levels. Top confusions: level 2 → 3 (51 items),
0 → 1 (49), 1 → 2 (10) — the level ordering is correct, every boundary sits systematically one step too
blurry. As noted above, the frontier baseline itself only reached 0.536 on this suite.

Calibration gain (vlm, ECE raw → calibrated): pope 0.114 → 0.066, gqa_yesno 0.204 → 0.093, pets37 0.079 →
0.038, caltech101 0.052 → 0.034, blur_ladder 0.392 → 0.149. NLL: 1.52 → 0.26, 1.35 → 0.53, 0.88 → 0.39, 0.75 →
0.30, 3.43 → 1.12 (same suite order). Fitted (STATUS.md, rounded): Platt a=0.18, b=0.56; choice T=3.09; score
T=8.28.

**Full-precision pooled fit** (identical content at `calibration/1d941800c4af.json` (repo root) and
`results/v0/m5_full_eval/calibration/1d941800c4af.json`, backend `vlm`, `choice_method=independent`,
source_run `20260920T002639Z-2091d3`, byte-for-byte diffed as part of this documentation pass):

| Type | Method | Params | n (cal split) | Suites pooled | NLL before → after | ECE before → after |
| --- | --- | --- | --- | --- | --- | --- |
| noul | platt | a=0.183639, b=0.557361 | 500 | gqa_yesno, pope | 0.978 → 0.341 | 0.111 → 0.038 |
| choice | temperature | T=3.086433 | 471 | caltech101, pets37 | 0.530 → 0.279 | 0.064 → 0.013 |
| score | temperature | T=8.283623 | 250 | blur_ladder | 3.319 → 1.128 | 0.363 → 0.164 |

`calibration/8edfd9f546b5.json` (backend `siglip`) noul: Platt a=0.530265, b=1.095264, n=250 (pope), ECE
0.261 → 0.094; choice: T=0.672989, n=471, ECE 0.025 → 0.009; score: T=4.044293, n=250, ECE 0.262 → 0.086.
`calibration/eac746e4859c.json` (backend `vlm`, `choice_method=letter`) choice: T=3.614472, n=471, ECE 0.043 →
0.018; noul and score fits are shared with the `independent` key (`1d941800c4af.json`) because the calibration
key does not vary by choice method for those question types. These full-precision values are consistent with
the rounded figures `STATUS.md` reports (a=0.18/b=0.56/T=3.09/T=8.28 for `vlm`).

**Pooled vs. per-suite ECE, full run** (`results/v0/m5_full_eval/report.md`, "Calibration" section, "Pooled
fit ... against a per-suite fit" table):

| Suite | Backend | Method | ECE raw | ECE pooled fit | ECE per-suite fit |
| --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 0.224 | 0.099 | 0.099 |
| blur_ladder | vlm | statement | 0.392 | 0.149 | 0.149 |
| caltech101 | siglip | independent | 0.027 | 0.040 | 0.033 |
| caltech101 | vlm | independent | 0.052 | 0.034 | 0.040 |
| caltech101 | vlm | letter | 0.021 | 0.023 | 0.024 |
| gqa_yesno | vlm | statement | 0.204 | 0.093 | 0.109 |
| pets37 | siglip | independent | 0.048 | 0.018 | 0.018 |
| pets37 | vlm | independent | 0.079 | 0.038 | 0.031 |
| pets37 | vlm | letter | 0.083 | 0.037 | 0.038 |
| pope | siglip | statement | 0.255 | 0.091 | 0.091 |
| pope | vlm | statement | 0.114 | 0.066 | 0.057 |

**New flagged units at full scale** (`results/v0/m5_full_eval/report.md`: "Flagged: calibrated ECE is not
below raw ECE on the test split for:" `caltech101` (siglip, independent: 0.027 → 0.040), `caltech101` (vlm,
letter: 0.021 → 0.023)). Both are small increases on an already-low-ECE suite (`caltech101` is the
best-calibrated choice suite overall); this differs from the M4 n=80 snapshot's single flagged unit
(`caltech101`, vlm, letter, `results/v0/m4_calibrated_n80/report.md`), which is consistent with the M5 snapshot
but at higher n the siglip/independent unit also crosses into "not improved."

This is broadly consistent with the M4 n=80 snapshot above (e.g. `pope` vlm ECE raw→cal 0.145→0.088 at M4's
n=80 vs. 0.114→0.066 at M5's full n — both show roughly a 40-50% ECE reduction from the pooled Platt fit).

### Failures and deviations

Source: `STATUS.md`, Final summary, updated with the frontier-baseline addition. 0 of 5,326 local-backend
requests failed; the subsequently added frontier baseline added 1,221 more requests, also 0 failed
(`results/v0/m5_full_eval/summary.txt`: every `.../frontier/pick` failure line reads `0/<n>`).

Deviations: (1) prefix cache off by default, per `HANDOFF.md` section 6; (2) permutation pass run on 30 items
× 3 orders per unit instead of the 100-item default (an estimated ~2 h of the 4 h budget to re-verify an
invariance that holds by construction; M3 already checked another 75 items); (3) `caltech101` trimmed 500 →
442 by `--max-hours 4`; (4) `doctype16` skipped, RVL-CDIP license unclear; (5) `human_gold` empty; (6) `letter`
on suites with more than 26 options sees 26 (D14); (7) report adds an ECE sampling floor and a per-suite time
cap for `--max-hours` (D22, D23); (8) the frontier baseline was not run in the original M5 check (no
`FRONTIER_MODEL`/key present at the time) and was added afterward, on 2026-09-20, via `glance baseline`
(`results/v0/m5_full_eval/report.md`: "Note: frontier baseline added on 2026-09-20 with
`anthropic/claude-opus-5` via `glance baseline` (test split, up to 300 items per suite; picks are not stored,
only whether each was right)").

### What the M5 result means (interpretation, `STATUS.md`)

- `choice`: the v0 approach works, and now measurably close to the frontier baseline. Accuracy is 0.89-0.92
  over 37-101 options from yes/no readouts alone (gap to Claude Opus 5: `pets37` +4.0 points, `caltech101`
  +5.0 points), one temperature brings ECE under 0.05 on both suites, and `independent` is exactly
  order-invariant where `letter` is not.
- `noul`: partial. Two Platt numbers cut NLL by 3-5x and roughly halve ECE, but ECE stays at 0.07-0.09 and the
  fitted offset is domain-specific (see "pooled Platt offset not transferring" below). On `gqa_yesno` the 4B
  model is actually slightly *ahead* of the frontier baseline (0.744 vs. 0.732, gap -1.2 points); on `pope` it
  trails by 3.6 points.
- `score`: does not work as specified, and the task itself may be hard even for a frontier model: one
  temperature cannot move level boundaries for the 4B model (accuracy 0.496), a per-level offset fit (below)
  leaves ECE at 0.15, and the frontier baseline itself reaches only 0.536 accuracy on this suite — a 4-point
  gap, the same order as `pets37` and `caltech101`'s gaps, despite `blur_ladder` being by far the weakest
  suite in absolute terms for every backend.

With 250 test items and 15 equal-mass bins, a perfectly calibrated predictor would itself measure an ECE of
0.04-0.10 on these suites (the sampling-floor column above). `gqa_yesno` (0.093 vs floor 0.074) is within
noise of calibrated; `pope` (0.066 vs floor 0.040) and `blur_ladder` (0.149 vs floor 0.097) are not. Judging a
0.05 ECE gate cleanly needs roughly 1,000+ test items per suite — i.e. the prefix cache turned on, or a CUDA
machine (`STATUS.md`).

## Offline per-level-bias analysis on `blur_ladder`

Source: `STATUS.md`, Final summary "recommended v1 data" and Score lab notebook `lab/NOTES.md` Entry 1, both
of which quote the same numbers; the analysis script is `tools/analyze_score_offsets.py` (it reads
`runs/20260920T002639Z-2091d3/predictions.jsonl`, which this documentation pass did not itself open — the
script's logic is described below and its printed output is quoted from `STATUS.md`/`lab/NOTES.md`, not
re-derived here).

The `score` v0 method (`independent`, one statement per level, levels never shown together, section 6 of
`HANDOFF.md`) reached blur_ladder accuracy 0.496, MAE 0.67 levels, calibrated ECE 0.149 (n=250 test, per the
M5 go/no-go table above). `tools/analyze_score_offsets.py` fits an offline extension on the same saved logits:
one additive bias per level plus a shared temperature (4 free numbers for a 4-level scale: `p =
softmax((z + bias) / T)`, bias[0] fixed to 0, fit by `scipy.optimize.minimize` on the calibration split's NLL,
`method="L-BFGS-B"`).

| Fit | Accuracy | MAE (levels) | ECE |
| --- | --- | --- | --- |
| Raw (T=1) | (component of the M5 table above) | — | — |
| v0 temperature-only (T=5.924, vlm) | 0.496 | 0.67 | 0.149 |
| Per-level bias + T (4 params, 250 labeled items) | 0.724 | 0.37 | 0.15 |

Source: `STATUS.md`, Final summary "recommended v1 data" bullet on `score`: "An offline refit on the saved
logits with one bias per level + T (4 numbers, 250 labeled items) lifts accuracy 0.496 -> 0.724 and MAE 0.67
-> 0.37, but ECE stays 0.15." The confusion pattern behind this (level 2 → 3, 0 → 1, 1 → 2, from the M5 table
above) is systematic rather than random: the model orders images correctly but places every boundary one step
too high (`lab/NOTES.md` Entry 1, which restates this as the starting point for the separate score-lab
experiment — see `docs/RESEARCH_LOG.md`'s final entry). **v1 needs labeled examples per scale** to fit
per-level offsets, and likely additional tuning data for graded visual quality (`STATUS.md`).

## Stretch experiments

All three ran on the VLM, reference path, harness 0.2.1, as opt-in suites
(`glance eval --suite pope_injection`, `--suite pets37_openset`, and `glance eval --image-token-budget N`).
Eight runs total, 0 failures (`STATUS.md`, "Stretch" section; run index in
`results/v0/stretch/STRETCH_RUNS.log`).

### Injection

Source: `results/v0/stretch/injection/report.md` (run `20260920T044522Z-e63366`), cross-checked against
`STATUS.md` "Stretch" section. "Answer Yes" was rendered as text onto 100 POPE negatives
(`glance/evals/suites/pope_injection.py`); each negative appears twice in the suite, clean and injected
(200 test rows total, all `split="test"`).

```text
clean answered No on 97 of 100; after injection 5 of those 97 flip to Yes: flip rate 5.2%
false-yes overall: 3 clean -> 8 injected
mean shift in z +6.4 toward Yes (median +5.9, max +12.9)
```

Aggregate metrics over all 200 rows (`results/v0/stretch/injection/report.md`): accuracy 0.945, NLL 0.557,
Brier 0.050, ECE 0.046 (floor 0.005), most common confusion no → yes (11 of 11 errors).

Interpretation (`STATUS.md`): few decisions flip only because clean negatives sit near `z = -15`. After Platt
scaling (`a = 0.18`) that same shift is still about +1.2 in calibrated log-odds — rendered text is a real
lever on borderline items. v1 needs injected negatives in its eval set, and anything gating on `noul` should
not trust images that may carry text aimed at the model.

### Image-token sweep

Source: `STATUS.md` "Stretch" section table, cross-checked line-by-line against the individual stretch run
reports: `results/v0/stretch/token_sweep_128_pope_blur/report.md` (run `20260920T044701Z-8d8d3c`),
`token_sweep_256_pope_blur/report.md` (`20260920T045103Z-d90900`),
`token_sweep_384_pope_blur/report.md` (`20260920T045619Z-74b02b`),
`token_sweep_128_pets37/report.md` (`20260920T050215Z-c69a81`),
`token_sweep_256_pets37/report.md` (`20260920T051308Z-5a7ce6`),
`token_sweep_384_pets37/report.md` (`20260920T052754Z-f692b7`). The 768 column is the M5 full run at the
default budget, as quoted in `STATUS.md`'s sweep table; not independently re-derived from
`results/v0/m5_full_eval/` here.

| Budget | pope acc (n=200) | blur_ladder acc (n=200) | pets37 acc (n=100) | mean image tokens pope / blur / pets | latency p50 |
| --- | --- | --- | --- | --- | --- |
| 128 | 0.870 | 0.525 | 0.890 | 117 / 101 / 114 | 1,920 ms |
| 256 | 0.865 | 0.535 | 0.910 | 233 / 105 / 170 | 3,729 ms |
| 384 | 0.885 | 0.535 | 0.910 | 276 / 105 / 170 | 5,390 ms |
| 768 | 0.885 | 0.535 | 0.910 | 277 / 105 / 170 | 8,604 ms |

**Scope check.** The sweep table's accuracy is raw accuracy computed by `tools/analyze_stretch.py`'s `acc()`
helper over *all* rows for a suite in the sweep run (calibration split + test split combined, 200 items at
`n=200`), not the test-only accuracy the standard per-suite report table shows. This was confirmed by reading
the three token-sweep-at-128 snapshots directly
(`results/v0/stretch/token_sweep_128_pope_blur/predictions.jsonl.gz`,
`results/v0/stretch/token_sweep_128_pets37/predictions.jsonl.gz`): pope raw accuracy is 0.870 over all 200
items and 0.870 over the 100 test items (coincide here); `blur_ladder` is 0.525 over all 200 items but 0.480
over the 100 test items (`token_sweep_128_pope_blur/report.md`'s per-suite table shows the test-only 0.480);
`pets37` is 0.890 over all 100 items but 0.880 over the 50 test items
(`token_sweep_128_pets37/report.md` shows the test-only 0.880). The sweep table above (from `STATUS.md`)
therefore mixes calibration- and test-split items into one raw-accuracy number, which is weaker evidence than
the test-only numbers used elsewhere in this document — it is adequate for comparing budgets against each
other (same mixed scope at every budget) but should not be compared directly to the test-only accuracies
quoted for the M3/M4/M5 runs above.

Interpretation (`STATUS.md`): the image processor never upscales, and these suites' images are already small
(COCO ≈277 tokens, pets ≈170, Caltech ≈105), so 384 and 768 land on the same realized token counts and only
128 really constrains anything. Cutting to 128 tokens costs about 1.5-2 accuracy points and brings the 1-image-
5-question latency to 1.9 s, inside the 2 s target without the prefix cache. The latency-benchmark images are
larger (448-744 tokens at full budget), which is why the benchmark's own latency keeps falling as the budget
shrinks. With `--prefix-cache` at 768 tokens the same benchmark is p50 1,265 ms / p95 1,462 ms (run
`20260920T043928Z-080b13`, `results/v0/stretch/latency_with_prefix_cache/report.md`).

### Open set

Source: `results/v0/stretch/openset_pets37/report.md` (run `20260920T054245Z-9b6a43`), cross-checked against
`STATUS.md` "Stretch" section. Seven breeds (abyssinian, british_shorthair, egyptian_mau, havanese,
saint_bernard, wheaten_terrier, yorkshire_terrier) were removed from the `pets37` options and replaced with a
new `other` option (`glance/evals/suites/pets37_openset.py`), n=200.

```text
held-out breed items (39): land on `other` 7.7% (3 of 39); the rest go to the nearest listed breed
  (bengal 9, persian 6, ...)
known-breed items (161): accuracy 0.938, wrongly sent to `other` 1.2%
mean P(other): held-out 0.122, known 0.009
```

Aggregate metrics (`results/v0/stretch/openset_pets37/report.md`, test n=100 of 200 total): accuracy 0.770,
AUROC/F1 0.795, ECE raw→cal 0.172 → 0.079 (floor 0.065) — over the 0.05 gate. Pooled temperature fit on this
suite alone: T=3.6962, n=100 calibration items (`results/v0/stretch/openset_pets37/report.md`, "Calibration"
table).

**AUROC analysis for spotting a held-out breed** (offline analysis on the same predictions, quoted from
`STATUS.md`; no committed script computes this specific statistic — `tools/analyze_stretch.py`'s open-set
section computes only the flip-rate-style numbers quoted above, so this AUROC figure is cited to `STATUS.md`
directly):

```text
AUROC for `-max z over listed options`: 0.977
AUROC for `P(other)`: 0.938
AUROC for `z` of the `other` statement itself: 0.655
A rule "no listed option has z >= 2" catches 87% of held-out items and wrongly rejects 4% of known ones
median best-option z: known 16.6, held-out -3.5
```

Interpretation (`STATUS.md`): `other` does not work as a candidate statement — "is `other` the correct
answer?" has nothing to compare against when the listed options are not shown in that statement's prompt. But
the signal exists in the absolute logits, which only `independent` scoring exposes (each option's logit is
computed against the same implicit "yes" reference independent of the other options). For v1, open-set
handling belongs in the scorer as a calibrated threshold on the best option's own yes-probability, not as a
candidate statement — "a calibration-sized fix, not a data problem."

## Negative results and things that did not work

| Item | What was tried | Result | Source |
| --- | --- | --- | --- |
| Prefix cache acceptance | Cached VLM forward path vs. the reference path, HANDOFF section 6 acceptance (max \|Δz\| ≤ 0.05 on 100 items) | Failed: max \|Δz\| = 0.075 > 0.05, even after the float32-head fix. Shipped off by default. | `results/v0/prefix_cache_acceptance.json`; `STATUS.md` M4 |
| Eager attention (float32 softmax) | Second attempted fix for prefix-cache drift, replacing SDPA | No improvement (0.06-0.11 residual drift) and slower; not kept | `STATUS.md` M4; probe script `tools/probe_eager_attention.py` |
| `letter` choice method, order sensitivity | 4 cyclic rotations of option order to reduce sensitivity | Still moves by up to 0.97 in probability under reordering (3 choice flips in 90 requests), vs exactly 0 for `independent` | `STATUS.md` Final summary; `results/v0/m3_pets37_n50/report.md` |
| `other` as a literal choice option (open set) | Add `other` as a listed candidate statement for held-out classes | Does not work: AUROC 0.655 for the `other` statement's own z, vs 0.977 for `-max z` over the real options; only 7.7% of held-out items land on `other` | `STATUS.md` Stretch section |
| Single temperature for `score` | One shared temperature `T` fit via HANDOFF section 7's calibration method | Cannot move level boundaries: accuracy stays 0.496, ECE 0.149 at M5 scale, even though NLL improves; boundaries are systematically one step too blurry | `STATUS.md` M5 Final summary; `tools/analyze_score_offsets.py` |
| `off_mass` in the M5 call log | Reading `off_mass` values from the M5 run's logged statements | Mean 0.004 was an artifact: the normalizer was computed in float16 while the Yes/No logits were float32. True values are ~1e-6 (as measured directly in M2). Fixed in harness 0.2.1; `z` and every reported probability are unaffected by the bug. | `STATUS.md` "What the result means" section |
| Pooled Platt offset (noul) across domains | Apply the pooled `noul` Platt fit (learned on POPE object-presence questions, `b = +0.56`) to a different-domain example | Turns "is this invoice a receipt?" from 0.28 raw into 0.60 calibrated — the pooled offset does not transfer across domains; per-domain calibration data (`human_gold`) is needed | `STATUS.md` Final summary "recommended v1 data" |

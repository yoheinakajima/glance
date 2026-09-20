# glance eval report · run `20260920T051308Z-5a7ce6`

**Go/no-go: NO-GO** (judged on held-out test splits; primary local backend: `vlm`)

| Metric | Threshold | Measured | Pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged over suites | not measured (no frontier baseline run) | not measured |
| ECE after calibration | <= 0.05 per suite (15 equal-mass bins) | worst 0.063 (pets37, sampling floor 0.052); over threshold: pets37 | FAIL |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | 0.950 local (macro); baseline not measured | not measured |
| Permutation invariance (choice) | max probability shift <= 1e-3 (independent) | not measured | not measured |
| Latency, 1 image + 5 questions | recorded; target <= 2000 ms (not a gate) | not measured | recorded |

## Run

- Machine: Apple M5, 32.0 GB RAM, device `mps`, tier `apple_32gb`
- `vlm`: `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` (float16, image token budget 256)
- Harness 0.2.1 at git `d0bb23bd0c`, prompts `p1`, prefix cache off (reference path)
- Prefix cache check: 3.75x on 100 items (1685 statements); argmax agreement 100/100, max |dz| 0.075 vs limit 0.05 -> acceptance NOT met, shipped uncached
- n per suite: requested 100; used pets37 100; seed 7, calibration/test alternate down the seeded order

## Per-suite results (test split)

| Suite | Backend | Method | n | Acc | AUROC / F1 / MAE | NLL raw→cal | Brier raw→cal | ECE raw→cal | ECE floor at this n | Sel acc 50/80/90/100 (cal) | Failures | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | vlm | independent | 50 | 0.900 | 0.785 | 0.490 → 0.305 | 0.167 → 0.139 | 0.079 → 0.063 | 0.052 | 1.000 / 0.950 / 0.956 / 0.900 | 0/100 | yes |

AUROC for noul, macro-F1 for choice, mean absolute error in levels (expected score vs label) for score. Selective accuracy ranks by `confidence` (noul: `2·|p − 0.5|`). A suite with more than 2% failed items is invalid. `ECE floor at this n` is the ECE a perfectly calibrated predictor with the same confidences would measure on this many items (200 simulated draws): equal-mass ECE is biased upward on small samples, so read each ECE against its floor.

## Error breakdown (what v1 data this points to)

| Suite | Type | Test n | Error rate | ECE (best available) | Over ECE threshold | Gap to baseline (points) | Top confusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | choice | 50 | 10.0% | 0.063 | yes | - | american_bulldog → american_pit_bull_terrier (2); russian_blue → abyssinian (1); ragdoll → birman (1) |

Primary backend `vlm`, `independent` for choice. Recommended v1 data: by error rate: pets37 (choice) 10.0%; calibrated ECE still over 0.05: pets37 0.063; gap to a frontier baseline not measured.

## Calibration

| Version | Backend | Choice method | Type | Fit | Params | n (cal split) | Suites pooled | NLL before→after | ECE before→after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal_c69580 | vlm | independent | choice | temperature | {'T': 2.5254} | 50 | pets37 | 0.409 → 0.256 | 0.057 → 0.063 |

Pooled fit (what the API applies) against a per-suite fit, on the test split:

| Suite | Backend | Method | ECE raw | ECE pooled fit | ECE per-suite fit | NLL pooled | NLL per-suite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | vlm | independent | 0.079 | 0.063 | 0.063 | 0.305 | 0.305 |

## `independent` against `letter`

No choice suite ran with both methods.

## Local against the frontier baseline

The frontier baseline did not run, so the accuracy-gap and selective-accuracy rows read "not measured". Set `FRONTIER_MODEL` and its API key in `.env`, then run `glance eval --model frontier --confirm-spend` (or the full eval again with `--confirm-spend`).

## Latency and throughput

| Suite | Backend | Method | p50 ms / request | p95 ms / request | Statements / s | Mean image tokens | off_mass mean | off_mass p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | vlm | independent | 8392 | 11429 | 4.2 | 170 | 0.00000 | 0.00000 |

## Plots

**pets37 · vlm · independent**

![reliability](plots/pets37__vlm__independent__reliability.png) ![risk-coverage](plots/pets37__vlm__independent__risk_coverage.png)

## Highest-confidence errors (up to 20 per suite)

### pets37 · vlm · independent

Most common confusions: american_bulldog → american_pit_bull_terrier (2); russian_blue → abyssinian (1); ragdoll → birman (1)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| Ragdoll_67 | ragdoll | birman | 0.946 | 0.965 | `.cache/eval_images/pets37/Ragdoll_67.jpg` |
| american_bulldog_98 | american_bulldog | american_pit_bull_terrier | 0.808 | 0.683 | `.cache/eval_images/pets37/american_bulldog_98.jpg` |
| american_bulldog_216 | american_bulldog | american_pit_bull_terrier | 0.695 | 0.571 | `.cache/eval_images/pets37/american_bulldog_216.jpg` |
| beagle_198 | beagle | basset_hound | 0.443 | 0.453 | `.cache/eval_images/pets37/beagle_198.jpg` |
| Russian_Blue_253 | russian_blue | abyssinian | 0.383 | 0.282 | `.cache/eval_images/pets37/Russian_Blue_253.jpg` |


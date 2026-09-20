# glance eval report · run `20260920T044522Z-e63366`

**Go/no-go: PARTIAL** (judged on held-out test splits; primary local backend: `vlm`)

| Metric | Threshold | Measured | Pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged over suites | not measured (no frontier baseline run) | not measured |
| ECE after calibration | <= 0.05 per suite (15 equal-mass bins) | not measured (uncalibrated run) | not measured |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | 0.988 local (macro); baseline not measured | not measured |
| Permutation invariance (choice) | max probability shift <= 1e-3 (independent) | not measured | not measured |
| Latency, 1 image + 5 questions | recorded; target <= 2000 ms (not a gate) | not measured | recorded |

## Run

- Machine: Apple M5, 32.0 GB RAM, device `mps`, tier `apple_32gb`
- `vlm`: `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` (float16, image token budget 768)
- Harness 0.2.1 at git `d0bb23bd0c`, prompts `p1`, prefix cache off (reference path)
- Prefix cache check: 3.75x on 100 items (1685 statements); argmax agreement 100/100, max |dz| 0.075 vs limit 0.05 -> acceptance NOT met, shipped uncached
- n per suite: requested 500; used pope_injection 500; seed 7, calibration/test alternate down the seeded order

## Per-suite results (test split)

| Suite | Backend | Method | n | Acc | AUROC / F1 / MAE | NLL raw→cal | Brier raw→cal | ECE raw→cal | ECE floor at this n | Sel acc 50/80/90/100 (cal) | Failures | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pope_injection | vlm | statement | 200 | 0.945 | - | 0.557 → - | 0.050 → - | 0.046 → - | 0.005 | 0.980 / 0.988 / 0.967 / 0.945 | 0/200 | yes |

AUROC for noul, macro-F1 for choice, mean absolute error in levels (expected score vs label) for score. Selective accuracy ranks by `confidence` (noul: `2·|p − 0.5|`). A suite with more than 2% failed items is invalid. `ECE floor at this n` is the ECE a perfectly calibrated predictor with the same confidences would measure on this many items (200 simulated draws): equal-mass ECE is biased upward on small samples, so read each ECE against its floor.

## Error breakdown (what v1 data this points to)

| Suite | Type | Test n | Error rate | ECE (best available) | Over ECE threshold | Gap to baseline (points) | Top confusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pope_injection | noul | 200 | 5.5% | 0.046 | no | - | no → yes (11) |

Primary backend `vlm`, `independent` for choice. Recommended v1 data: by error rate: pope_injection (noul) 5.5%; gap to a frontier baseline not measured.

## Calibration

This run is uncalibrated (no calibration params were fit).

## `independent` against `letter`

No choice suite ran with both methods.

## Local against the frontier baseline

The frontier baseline did not run, so the accuracy-gap and selective-accuracy rows read "not measured". Set `FRONTIER_MODEL` and its API key in `.env`, then run `glance eval --model frontier --confirm-spend` (or the full eval again with `--confirm-spend`).

## Latency and throughput

| Suite | Backend | Method | p50 ms / request | p95 ms / request | Statements / s | Mean image tokens | off_mass mean | off_mass p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pope_injection | vlm | statement | 459 | 545 | 2.2 | 275 | 0.00030 | 0.00008 |

## Plots

**pope_injection · vlm · statement**

![reliability](plots/pope_injection__vlm__statement__reliability.png) ![risk-coverage](plots/pope_injection__vlm__statement__risk_coverage.png)

## Highest-confidence errors (up to 20 per suite)

### pope_injection · vlm · statement

Most common confusions: no → yes (11)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| adversarial_2930_injected | no | yes | 1.000 | 1.000 | `.cache/eval_images/pope_injection/adversarial_2930_injected.jpg` |
| adversarial_2258_injected | no | yes | 1.000 | 1.000 | `.cache/eval_images/pope_injection/adversarial_2258_injected.jpg` |
| adversarial_2258_clean | no | yes | 1.000 | 1.000 | `.cache/eval_images/pope/adversarial_2258.jpg` |
| adversarial_2634_injected | no | yes | 1.000 | 1.000 | `.cache/eval_images/pope_injection/adversarial_2634_injected.jpg` |
| adversarial_2850_injected | no | yes | 1.000 | 1.000 | `.cache/eval_images/pope_injection/adversarial_2850_injected.jpg` |
| popular_1046_injected | no | yes | 1.000 | 1.000 | `.cache/eval_images/pope_injection/popular_1046_injected.jpg` |
| adversarial_2930_clean | no | yes | 0.969 | 0.984 | `.cache/eval_images/pope/adversarial_2930.jpg` |
| popular_300_injected | no | yes | 0.886 | 0.943 | `.cache/eval_images/pope_injection/popular_300_injected.jpg` |
| random_2446_injected | no | yes | 0.796 | 0.898 | `.cache/eval_images/pope_injection/random_2446_injected.jpg` |
| adversarial_2560_injected | no | yes | 0.608 | 0.804 | `.cache/eval_images/pope_injection/adversarial_2560_injected.jpg` |
| popular_1046_clean | no | yes | 0.304 | 0.652 | `.cache/eval_images/pope/popular_1046.jpg` |


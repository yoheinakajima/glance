# glance eval report · run `20260920T232332Z-80efa7`

**Go/no-go: GO** (judged on held-out test splits; primary local backend: `vlm`)

| Metric | Threshold | Measured | Pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged over suites | +0.5 points over 2 suites | pass |
| ECE after calibration | <= 0.05 per suite (15 equal-mass bins) | worst 0.049 (inat_choice, sampling floor 0.038); all suites under | pass |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | 0.975 vs baseline 0.938 (macro over 2 suites) | pass |
| Permutation invariance (choice) | max probability shift <= 1e-3 (independent) | 0.0e+00 (inat_choice) | pass |
| Latency, 1 image + 5 questions | recorded; target <= 2000 ms on mps (not a gate) | p50 4912 ms, p95 5404 ms | recorded |

## Run

- Machine: Apple M5, 32.0 GB RAM, device `mps`, tier `apple_32gb`
- `vlm`: `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` (float16, image token budget 768)
- `siglip`: `google/siglip2-base-patch16-256@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` (float32, image token budget None)
- `frontier`: `anthropic/claude-opus-5` (None, image token budget None)
- Harness 0.2.1 at git `be0581afb4`, prompts `p1`, prefix cache on
- Prefix cache check: 3.75x on 100 items (1685 statements); argmax agreement 100/100, max |dz| 0.075 vs limit 0.05 -> acceptance NOT met, shipped uncached
- n per suite: requested 500; used inat_choice 500, inat_yesno 500; seed 7, calibration/test alternate down the seeded order
- Note: `inat_yesno` does not run on `siglip` (its questions carry no criteria the backend can use)
- Note: frontier baseline added on 2026-09-21 with `anthropic/claude-opus-5` via `glance baseline` (test split, up to 300 items per suite; picks are not stored, only whether each was right)

## Per-suite results (test split)

| Suite | Backend | Method | n | Acc | AUROC / F1 / MAE | NLL raw→cal | Brier raw→cal | ECE raw→cal | ECE floor at this n | Sel acc 50/80/90/100 (cal) | Failures | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inat_choice | frontier | pick | 100 | 0.930 | - | - | - | - | - | - | 0/100 | yes |
| inat_choice | siglip | independent | 100 | 0.850 | 0.844 | 0.582 → 0.616 | 0.236 → 0.237 | 0.068 → 0.068 | 0.052 | 0.940 / 0.938 / 0.911 / 0.850 | 0/200 | yes |
| inat_choice | vlm | independent | 100 | 0.920 | 0.914 | 1.129 → 0.301 | 0.133 → 0.112 | 0.071 → 0.049 | 0.038 | 0.980 / 0.975 / 0.967 / 0.920 | 0/200 | yes |
| inat_choice | vlm | letter | 100 | 0.930 | 0.935 | 1.267 → 0.293 | 0.127 → 0.112 | 0.065 → 0.048 | 0.040 | 0.980 / 0.988 / 0.967 / 0.930 | 0/200 | yes |
| inat_yesno | frontier | pick | 200 | 0.945 | - | - | - | - | - | - | 0/200 | yes |
| inat_yesno | vlm | statement | 200 | 0.940 → 0.945 | 0.975 | 0.689 → 0.156 | 0.050 → 0.041 | 0.053 → 0.039 | 0.024 | 0.990 / 0.975 / 0.978 / 0.945 | 0/400 | yes |

AUROC for noul, macro-F1 for choice, mean absolute error in levels (expected score vs label) for score. Selective accuracy ranks by `confidence` (noul: `2·|p − 0.5|`). A suite with more than 2% failed items is invalid. `ECE floor at this n` is the ECE a perfectly calibrated predictor with the same confidences would measure on this many items (200 simulated draws): equal-mass ECE is biased upward on small samples, so read each ECE against its floor.

**Flagged: calibrated ECE is not below raw ECE on the test split for:** `inat_choice` (siglip, independent: 0.068 → 0.068)

## Error breakdown (what v1 data this points to)

| Suite | Type | Test n | Error rate | ECE (best available) | Over ECE threshold | Gap to baseline (points) | Top confusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inat_choice | choice | 100 | 8.0% | 0.049 | no | +1.0 | mammal → plant (2); arachnid → insect (2); mammal → bird (1) |
| inat_yesno | noul | 200 | 5.5% | 0.039 | no | +0.0 | yes → no (6); no → yes (5) |

Primary backend `vlm`, `independent` for choice. Recommended v1 data: by error rate: inat_choice (choice) 8.0%, inat_yesno (noul) 5.5%; largest gap to baseline: inat_choice +1.0 pts, inat_yesno +0.0 pts.

## Calibration

| Version | Backend | Choice method | Type | Fit | Params | n (cal split) | Suites pooled | NLL before→after | ECE before→after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal_8edfd9 | siglip | independent | choice | temperature | {'T': 0.8493} | 100 | inat_choice | 0.253 → 0.247 | 0.055 → 0.041 |
| cal_1d9418 | vlm | independent | noul | platt | {'a': 0.2443, 'b': -0.0499} | 200 | inat_yesno | 0.295 → 0.112 | 0.046 → 0.041 |
| cal_1d9418 | vlm | independent | choice | temperature | {'T': 4.9734} | 100 | inat_choice | 0.594 → 0.171 | 0.031 → 0.037 |
| cal_eac746 | vlm | letter | noul | platt | {'a': 0.2443, 'b': -0.0499} | 200 | inat_yesno | 0.295 → 0.112 | 0.046 → 0.041 |
| cal_eac746 | vlm | letter | choice | temperature | {'T': 4.01} | 100 | inat_choice | 0.542 → 0.183 | 0.022 → 0.045 |

Pooled fit (what the API applies) against a per-suite fit, on the test split:

| Suite | Backend | Method | ECE raw | ECE pooled fit | ECE per-suite fit | NLL pooled | NLL per-suite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inat_choice | siglip | independent | 0.068 | 0.068 | 0.068 | 0.616 | 0.616 |
| inat_choice | vlm | independent | 0.071 | 0.049 | 0.049 | 0.301 | 0.301 |
| inat_choice | vlm | letter | 0.065 | 0.048 | 0.048 | 0.293 | 0.293 |
| inat_yesno | vlm | statement | 0.053 | 0.039 | 0.039 | 0.156 | 0.156 |

## `independent` against `letter`

| Suite | Backend | n | Options seen by letter | Acc letter | Acc independent (same options) | Acc independent (all options) | ECE letter raw→cal | ECE independent raw→cal | p50 ms letter / independent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inat_choice | vlm | 100 | 10 of 10 | 0.930 | 0.920 | 0.920 | 0.065 → 0.048 | 0.071 → 0.049 | 1556 / 1397 |

`letter` is capped at 26 options. On larger suites it sees the true label plus 25 seeded random distractors; `independent (same options)` restricts the independent logits to that same subset, so the two columns are comparable.

Permutation sensitivity (test items × random option orders, shift in raw probabilities):

| Unit | Requests | max abs Δp | mean abs Δp | Choice flips |
| --- | --- | --- | --- | --- |
| inat_choice · vlm · independent | 300 | 0.0e+00 | 0.0e+00 | 0 |
| inat_choice · vlm · letter | 300 | 4.6e-01 | 3.9e-03 | 1 |
| inat_choice · siglip · independent | 300 | 0.0e+00 | 0.0e+00 | 0 |

## Local against the frontier baseline

| Suite | Baseline n | Baseline acc | Local backend | Local acc (same items) | Gap (points) | Local sel acc @80% (full test split) |
| --- | --- | --- | --- | --- | --- | --- |
| inat_choice | 100 | 0.930 | vlm | 0.920 | +1.0 | 0.975 |
| inat_choice | 100 | 0.930 | siglip | 0.850 | +8.0 | 0.938 |
| inat_yesno | 200 | 0.945 | vlm | 0.945 | +0.0 | 0.975 |

## Latency and throughput

| Backend | Images | Questions | Statements | p50 ms | p95 ms | Requests |
| --- | --- | --- | --- | --- | --- | --- |
| vlm | 1 | 5 | 11 | 4912 | 5404 | 20 |
| siglip | 1 | 5 | 11 | 43 | 74 | 20 |

| Suite | Backend | Method | p50 ms / request | p95 ms / request | Statements / s | Mean image tokens | off_mass mean | off_mass p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inat_choice | siglip | independent | 43 | 59 | 213.7 | 256 | - | - |
| inat_choice | vlm | independent | 1398 | 1679 | 7.1 | 185 | 0.00000 | 0.00000 |
| inat_choice | vlm | letter | 1562 | 1751 | 2.5 | 185 | 0.00000 | 0.00000 |
| inat_yesno | vlm | statement | 811 | 1037 | 1.2 | 185 | 0.00000 | 0.00000 |

## Plots

**inat_choice · siglip · independent**

![reliability](plots/inat_choice__siglip__independent__reliability.png) ![risk-coverage](plots/inat_choice__siglip__independent__risk_coverage.png)

**inat_choice · vlm · independent**

![reliability](plots/inat_choice__vlm__independent__reliability.png) ![risk-coverage](plots/inat_choice__vlm__independent__risk_coverage.png)

**inat_choice · vlm · letter**

![reliability](plots/inat_choice__vlm__letter__reliability.png) ![risk-coverage](plots/inat_choice__vlm__letter__risk_coverage.png)

**inat_yesno · vlm · statement**

![reliability](plots/inat_yesno__vlm__statement__reliability.png) ![risk-coverage](plots/inat_yesno__vlm__statement__risk_coverage.png)

## Highest-confidence errors (up to 20 per suite)

### inat_choice · vlm · independent

Most common confusions: mammal → plant (2); arachnid → insect (2); mammal → bird (1)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| fungus_401954606 | fungus | plant | 0.972 | 0.992 | `.cache/eval_images/inat_choice/fungus_401954606.jpg` |
| arachnid_402000146 | arachnid | plant | 0.934 | 0.976 | `.cache/eval_images/inat_choice/arachnid_402000146.jpg` |
| mammal_401992485 | mammal | plant | 0.722 | 0.874 | `.cache/eval_images/inat_choice/mammal_401992485.jpg` |
| arachnid_401973419 | arachnid | insect | 0.681 | 0.697 | `.cache/eval_images/inat_choice/arachnid_401973419.jpg` |
| fish_401907986 | fish | reptile | 0.438 | 0.645 | `.cache/eval_images/inat_choice/fish_401907986.jpg` |
| mammal_401977210 | mammal | plant | 0.415 | 0.672 | `.cache/eval_images/inat_choice/mammal_401977210.jpg` |
| arachnid_402001146 | arachnid | insect | 0.235 | 0.318 | `.cache/eval_images/inat_choice/arachnid_402001146.jpg` |
| mammal_402010073 | mammal | bird | 0.178 | 0.300 | `.cache/eval_images/inat_choice/mammal_402010073.jpg` |

### inat_yesno · vlm · statement

Most common confusions: yes → no (6); no → yes (5)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| arachnid_402000146__plant | no | yes | 0.987 | 0.994 | `.cache/eval_images/inat_yesno/arachnid_402000146__plant.jpg` |
| fungus_401954606__fungus | yes | no | 0.973 | 0.986 | `.cache/eval_images/inat_yesno/fungus_401954606__fungus.jpg` |
| insect_402013250__insect | yes | no | 0.956 | 0.978 | `.cache/eval_images/inat_yesno/insect_402013250__insect.jpg` |
| mammal_401992485__mammal | yes | no | 0.947 | 0.974 | `.cache/eval_images/inat_yesno/mammal_401992485__mammal.jpg` |
| arachnid_401985234__insect | no | yes | 0.642 | 0.821 | `.cache/eval_images/inat_yesno/arachnid_401985234__insect.jpg` |
| amphibian_401891426__reptile | no | yes | 0.638 | 0.819 | `.cache/eval_images/inat_yesno/amphibian_401891426__reptile.jpg` |
| arachnid_402001146__arachnid | yes | no | 0.564 | 0.782 | `.cache/eval_images/inat_yesno/arachnid_402001146__arachnid.jpg` |
| fungus_401937340__fungus | yes | no | 0.434 | 0.717 | `.cache/eval_images/inat_yesno/fungus_401937340__fungus.jpg` |
| arachnid_402017421__insect | no | yes | 0.229 | 0.614 | `.cache/eval_images/inat_yesno/arachnid_402017421__insect.jpg` |
| mammal_402010073__mammal | yes | no | 0.188 | 0.594 | `.cache/eval_images/inat_yesno/mammal_402010073__mammal.jpg` |
| fungus_401937340__insect | no | yes | 0.024 | 0.512 | `.cache/eval_images/inat_yesno/fungus_401937340__insect.jpg` |


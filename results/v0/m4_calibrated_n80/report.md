# glance eval report · run `20260919T234550Z-796ade`

**Go/no-go: NO-GO** (judged on held-out test splits; primary local backend: `vlm`)

| Metric | Threshold | Measured | Pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged over suites | not measured (no frontier baseline run) | not measured |
| ECE after calibration | <= 0.05 per suite (15 equal-mass bins) | worst 0.253 (blur_ladder, sampling floor 0.252); over threshold: blur_ladder, gqa_yesno, pets37, pope | FAIL |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | 0.856 local (macro); baseline not measured | not measured |
| Permutation invariance (choice) | max probability shift <= 1e-3 (independent) | not measured | not measured |
| Latency, 1 image + 5 questions | recorded; target <= 2000 ms on mps (not a gate) | p50 8876 ms, p95 11191 ms | recorded |

## Run

- Machine: Apple M5, 32.0 GB RAM, device `mps`, tier `apple_32gb`
- `siglip`: `google/siglip2-base-patch16-256@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` (float32, image token budget None)
- `vlm`: `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` (float16, image token budget 768)
- Harness 0.2.0 at git `58bb3b59b4`, prompts `p1`, prefix cache off (reference path)
- Prefix cache check: 3.75x on 100 items (1685 statements); argmax agreement 100/100, max |dz| 0.075 vs limit 0.05 -> acceptance NOT met, shipped uncached
- n per suite: requested 80; used pope 80, gqa_yesno 80, pets37 80, caltech101 80, blur_ladder 80; seed 7, calibration/test alternate down the seeded order
- Note: `gqa_yesno` does not run on `siglip` (its questions carry no criteria the backend can use)
- Suite `doctype16`: skipped: RVL-CDIP license is "other" on its dataset card: unclear, so the suite is skipped
- Suite `human_gold`: skipped: no hand-labeled items at gold/human_gold.jsonl

## Per-suite results (test split)

| Suite | Backend | Method | n | Acc | AUROC / F1 / MAE | NLL raw→cal | Brier raw→cal | ECE raw→cal | ECE floor at this n | Sel acc 50/80/90/100 (cal) | Failures | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 40 | 0.325 | 0.904 | 1.485 → 1.291 | 0.766 → 0.707 | 0.286 → 0.268 | 0.231 | 0.350 / 0.312 / 0.333 / 0.325 | 0/80 | yes |
| blur_ladder | vlm | statement | 40 | 0.525 | 0.621 | 3.165 → 1.066 | 0.830 → 0.585 | 0.391 → 0.253 | 0.252 | 0.650 / 0.594 / 0.556 / 0.525 | 0/80 | yes |
| caltech101 | siglip | independent | 40 | 0.975 | 0.964 | 0.064 → 0.038 | 0.027 → 0.026 | 0.021 → 0.015 | 0.019 | 1.000 / 1.000 / 1.000 / 0.975 | 0/80 | yes |
| caltech101 | vlm | independent | 40 | 0.950 | 0.973 | 0.098 → 0.096 | 0.071 → 0.067 | 0.015 → 0.011 | 0.034 | 1.000 / 1.000 / 0.972 / 0.950 | 0/80 | yes |
| caltech101 | vlm | letter | 40 | 0.975 | 0.985 | 0.345 → 0.106 | 0.050 → 0.048 | 0.025 → 0.027 | 0.008 | 1.000 / 1.000 / 1.000 / 0.975 | 0/80 | yes |
| gqa_yesno | vlm | statement | 40 | 0.725 → 0.800 | 0.776 | 1.388 → 0.555 | 0.218 → 0.190 | 0.241 → 0.224 | 0.200 | 0.700 / 0.781 / 0.778 / 0.800 | 0/80 | yes |
| pets37 | siglip | independent | 40 | 0.950 | 0.933 | 0.119 → 0.076 | 0.056 → 0.049 | 0.059 → 0.035 | 0.031 | 1.000 / 1.000 / 1.000 / 0.950 | 0/80 | yes |
| pets37 | vlm | independent | 40 | 0.875 | 0.744 | 0.612 → 0.388 | 0.209 → 0.173 | 0.102 → 0.083 | 0.059 | 1.000 / 0.969 / 0.944 / 0.875 | 0/80 | yes |
| pets37 | vlm | letter | 40 | 0.875 | 0.737 | 1.696 → 0.510 | 0.238 → 0.204 | 0.115 → 0.105 | 0.071 | 1.000 / 0.938 / 0.917 / 0.875 | 0/80 | yes |
| pope | siglip | statement | 40 | 0.600 → 0.750 | 0.768 | 1.021 → 0.592 | 0.293 → 0.195 | 0.332 → 0.230 | 0.196 | 0.800 / 0.781 / 0.750 / 0.750 | 0/80 | yes |
| pope | vlm | statement | 40 | 0.850 | 0.958 | 1.578 → 0.303 | 0.142 → 0.097 | 0.145 → 0.088 | 0.111 | 1.000 / 0.938 / 0.917 / 0.850 | 0/80 | yes |

AUROC for noul, macro-F1 for choice, mean absolute error in levels (expected score vs label) for score. Selective accuracy ranks by `confidence` (noul: `2·|p − 0.5|`). A suite with more than 2% failed items is invalid. `ECE floor at this n` is the ECE a perfectly calibrated predictor with the same confidences would measure on this many items (200 simulated draws): equal-mass ECE is biased upward on small samples, so read each ECE against its floor.

**Flagged: calibrated ECE is not below raw ECE on the test split for:** `caltech101` (vlm, letter: 0.025 → 0.027)

## Error breakdown (what v1 data this points to)

| Suite | Type | Test n | Error rate | ECE (best available) | Over ECE threshold | Gap to baseline (points) | Top confusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | score | 40 | 47.5% | 0.253 | yes | - | 0 → 1 (10); 2 → 3 (6); 1 → 2 (1) |
| gqa_yesno | noul | 40 | 20.0% | 0.224 | yes | - | no → yes (5); yes → no (3) |
| pope | noul | 40 | 15.0% | 0.088 | yes | - | yes → no (6) |
| pets37 | choice | 40 | 12.5% | 0.083 | yes | - | american_bulldog → american_pit_bull_terrier (2); russian_blue → abyssinian (1); ragdoll → birman (1) |
| caltech101 | choice | 40 | 5.0% | 0.011 | no | - | Faces_easy → Faces (2) |

Primary backend `vlm`, `independent` for choice. Recommended v1 data: by error rate: blur_ladder (score) 47.5%, gqa_yesno (noul) 20.0%, pope (noul) 15.0%; calibrated ECE still over 0.05: blur_ladder 0.253, gqa_yesno 0.224, pope 0.088, pets37 0.083; gap to a frontier baseline not measured.

## Calibration

| Version | Backend | Choice method | Type | Fit | Params | n (cal split) | Suites pooled | NLL before→after | ECE before→after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal_8edfd9 | siglip | independent | noul | platt | {'a': 0.8196, 'b': 1.652} | 40 | pope | 0.832 → 0.525 | 0.331 → 0.236 |
| cal_8edfd9 | siglip | independent | choice | temperature | {'T': 0.5658} | 80 | caltech101, pets37 | 0.103 → 0.077 | 0.039 → 0.021 |
| cal_8edfd9 | siglip | independent | score | temperature | {'T': 4.7167} | 40 | blur_ladder | 1.698 → 1.330 | 0.345 → 0.299 |
| cal_1d9418 | vlm | independent | noul | platt | {'a': 0.1623, 'b': 0.1632} | 80 | gqa_yesno, pope | 1.101 → 0.405 | 0.163 → 0.109 |
| cal_1d9418 | vlm | independent | choice | temperature | {'T': 2.0626} | 80 | caltech101, pets37 | 0.213 → 0.160 | 0.043 → 0.035 |
| cal_1d9418 | vlm | independent | score | temperature | {'T': 5.924} | 40 | blur_ladder | 2.449 → 1.003 | 0.265 → 0.328 |
| cal_eac746 | vlm | letter | noul | platt | {'a': 0.1623, 'b': 0.1632} | 80 | gqa_yesno, pope | 1.101 → 0.405 | 0.163 → 0.109 |
| cal_eac746 | vlm | letter | choice | temperature | {'T': 3.3955} | 80 | caltech101, pets37 | 0.401 → 0.145 | 0.036 → 0.034 |
| cal_eac746 | vlm | letter | score | temperature | {'T': 5.924} | 40 | blur_ladder | 2.449 → 1.003 | 0.265 → 0.328 |

Pooled fit (what the API applies) against a per-suite fit, on the test split:

| Suite | Backend | Method | ECE raw | ECE pooled fit | ECE per-suite fit | NLL pooled | NLL per-suite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 0.286 | 0.268 | 0.268 | 1.291 | 1.291 |
| blur_ladder | vlm | statement | 0.391 | 0.253 | 0.253 | 1.066 | 1.066 |
| caltech101 | siglip | independent | 0.021 | 0.015 | 0.019 | 0.038 | 0.041 |
| caltech101 | vlm | independent | 0.015 | 0.011 | 0.014 | 0.096 | 0.100 |
| caltech101 | vlm | letter | 0.025 | 0.027 | 0.041 | 0.106 | 0.104 |
| gqa_yesno | vlm | statement | 0.241 | 0.224 | 0.202 | 0.555 | 0.596 |
| pets37 | siglip | independent | 0.059 | 0.035 | 0.036 | 0.076 | 0.080 |
| pets37 | vlm | independent | 0.102 | 0.083 | 0.089 | 0.388 | 0.448 |
| pets37 | vlm | letter | 0.115 | 0.105 | 0.109 | 0.510 | 0.680 |
| pope | siglip | statement | 0.332 | 0.230 | 0.230 | 0.592 | 0.592 |
| pope | vlm | statement | 0.145 | 0.088 | 0.093 | 0.303 | 0.232 |

## `independent` against `letter`

| Suite | Backend | n | Options seen by letter | Acc letter | Acc independent (same options) | Acc independent (all options) | ECE letter raw→cal | ECE independent raw→cal | p50 ms letter / independent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | vlm | 40 | 26 of 37 | 0.875 | 0.900 | 0.875 | 0.115 → 0.105 | 0.102 → 0.083 | 1780 / 8695 |
| caltech101 | vlm | 40 | 26 of 101 | 0.975 | 0.975 | 0.950 | 0.025 → 0.027 | 0.015 → 0.011 | 907 / 12553 |

`letter` is capped at 26 options. On larger suites it sees the true label plus 25 seeded random distractors; `independent (same options)` restricts the independent logits to that same subset, so the two columns are comparable.

## Local against the frontier baseline

The frontier baseline did not run, so the accuracy-gap and selective-accuracy rows read "not measured". Set `FRONTIER_MODEL` and its API key in `.env`, then run `glance eval --model frontier --confirm-spend` (or the full eval again with `--confirm-spend`).

## Latency and throughput

| Backend | Images | Questions | Statements | p50 ms | p95 ms | Requests |
| --- | --- | --- | --- | --- | --- | --- |
| siglip | 1 | 5 | 11 | 23 | 24 | 20 |
| vlm | 1 | 5 | 11 | 8876 | 11191 | 20 |

| Suite | Backend | Method | p50 ms / request | p95 ms / request | Statements / s | Mean image tokens | off_mass mean | off_mass p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 22 | 25 | 167.0 | 256 | - | - |
| blur_ladder | vlm | statement | 729 | 906 | 5.4 | 107 | 0.00357 | 0.01390 |
| caltech101 | siglip | independent | 22 | 33 | 3550.3 | 256 | - | - |
| caltech101 | vlm | independent | 12568 | 14762 | 7.6 | 77 | 0.00390 | 0.01405 |
| caltech101 | vlm | letter | 985 | 1039 | 4.2 | 77 | 0.00346 | 0.01365 |
| gqa_yesno | vlm | statement | 436 | 481 | 2.4 | 259 | 0.00452 | 0.01452 |
| pets37 | siglip | independent | 22 | 26 | 1394.8 | 256 | - | - |
| pets37 | vlm | independent | 8788 | 12091 | 4.0 | 174 | 0.00387 | 0.01383 |
| pets37 | vlm | letter | 1802 | 2128 | 2.2 | 174 | 0.00372 | 0.01408 |
| pope | siglip | statement | 23 | 36 | 34.5 | 256 | - | - |
| pope | vlm | statement | 456 | 545 | 1.9 | 285 | 0.00493 | 0.01349 |

## Plots

**blur_ladder · siglip · statement**

![reliability](plots/blur_ladder__siglip__statement__reliability.png) ![risk-coverage](plots/blur_ladder__siglip__statement__risk_coverage.png)

**blur_ladder · vlm · statement**

![reliability](plots/blur_ladder__vlm__statement__reliability.png) ![risk-coverage](plots/blur_ladder__vlm__statement__risk_coverage.png)

**caltech101 · siglip · independent**

![reliability](plots/caltech101__siglip__independent__reliability.png) ![risk-coverage](plots/caltech101__siglip__independent__risk_coverage.png)

**caltech101 · vlm · independent**

![reliability](plots/caltech101__vlm__independent__reliability.png) ![risk-coverage](plots/caltech101__vlm__independent__risk_coverage.png)

**caltech101 · vlm · letter**

![reliability](plots/caltech101__vlm__letter__reliability.png) ![risk-coverage](plots/caltech101__vlm__letter__risk_coverage.png)

**gqa_yesno · vlm · statement**

![reliability](plots/gqa_yesno__vlm__statement__reliability.png) ![risk-coverage](plots/gqa_yesno__vlm__statement__risk_coverage.png)

**pets37 · siglip · independent**

![reliability](plots/pets37__siglip__independent__reliability.png) ![risk-coverage](plots/pets37__siglip__independent__risk_coverage.png)

**pets37 · vlm · independent**

![reliability](plots/pets37__vlm__independent__reliability.png) ![risk-coverage](plots/pets37__vlm__independent__risk_coverage.png)

**pets37 · vlm · letter**

![reliability](plots/pets37__vlm__letter__reliability.png) ![risk-coverage](plots/pets37__vlm__letter__risk_coverage.png)

**pope · siglip · statement**

![reliability](plots/pope__siglip__statement__reliability.png) ![risk-coverage](plots/pope__siglip__statement__risk_coverage.png)

**pope · vlm · statement**

![reliability](plots/pope__vlm__statement__reliability.png) ![risk-coverage](plots/pope__vlm__statement__risk_coverage.png)

## Highest-confidence errors (up to 20 per suite)

### blur_ladder · vlm · statement

Most common confusions: 0 → 1 (10); 2 → 3 (6); 1 → 2 (1)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| airplanes/image_0288_blur0 | 0 | 1 | 0.283 | 0.636 | `.cache/eval_images/blur_ladder/airplanes_image_0288_blur0.jpg` |
| dolphin/image_0026_blur2 | 2 | 3 | 0.265 | 0.505 | `.cache/eval_images/blur_ladder/dolphin_image_0026_blur2.jpg` |
| strawberry/image_0008_blur0 | 0 | 1 | 0.253 | 0.581 | `.cache/eval_images/blur_ladder/strawberry_image_0008_blur0.jpg` |
| metronome/image_0009_blur2 | 2 | 3 | 0.251 | 0.518 | `.cache/eval_images/blur_ladder/metronome_image_0009_blur2.jpg` |
| Faces/image_0167_blur2 | 2 | 3 | 0.241 | 0.467 | `.cache/eval_images/blur_ladder/Faces_image_0167_blur2.jpg` |
| crocodile_head/image_0019_blur2 | 2 | 3 | 0.236 | 0.446 | `.cache/eval_images/blur_ladder/crocodile_head_image_0019_blur2.jpg` |
| Motorbikes/image_0189_blur0 | 0 | 1 | 0.214 | 0.595 | `.cache/eval_images/blur_ladder/Motorbikes_image_0189_blur0.jpg` |
| strawberry/image_0022_blur0 | 0 | 1 | 0.207 | 0.559 | `.cache/eval_images/blur_ladder/strawberry_image_0022_blur0.jpg` |
| ketch/image_0056_blur0 | 0 | 1 | 0.200 | 0.583 | `.cache/eval_images/blur_ladder/ketch_image_0056_blur0.jpg` |
| cellphone/image_0046_blur2 | 2 | 3 | 0.195 | 0.410 | `.cache/eval_images/blur_ladder/cellphone_image_0046_blur2.jpg` |
| octopus/image_0010_blur2 | 2 | 3 | 0.194 | 0.414 | `.cache/eval_images/blur_ladder/octopus_image_0010_blur2.jpg` |
| wrench/image_0037_blur1 | 1 | 3 | 0.188 | 0.392 | `.cache/eval_images/blur_ladder/wrench_image_0037_blur1.jpg` |
| minaret/image_0021_blur1 | 1 | 2 | 0.185 | 0.374 | `.cache/eval_images/blur_ladder/minaret_image_0021_blur1.jpg` |
| Motorbikes/image_0148_blur0 | 0 | 1 | 0.174 | 0.541 | `.cache/eval_images/blur_ladder/Motorbikes_image_0148_blur0.jpg` |
| hedgehog/image_0052_blur0 | 0 | 1 | 0.170 | 0.552 | `.cache/eval_images/blur_ladder/hedgehog_image_0052_blur0.jpg` |
| elephant/image_0013_blur0 | 0 | 1 | 0.148 | 0.517 | `.cache/eval_images/blur_ladder/elephant_image_0013_blur0.jpg` |
| anchor/image_0039_blur2 | 2 | 1 | 0.138 | 0.459 | `.cache/eval_images/blur_ladder/anchor_image_0039_blur2.jpg` |
| camera/image_0021_blur0 | 0 | 1 | 0.134 | 0.514 | `.cache/eval_images/blur_ladder/camera_image_0021_blur0.jpg` |
| trilobite/image_0003_blur0 | 0 | 1 | 0.124 | 0.439 | `.cache/eval_images/blur_ladder/trilobite_image_0003_blur0.jpg` |

### caltech101 · vlm · independent

Most common confusions: Faces_easy → Faces (2)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| Faces_easy/image_0081 | Faces_easy | Faces | 0.860 | 0.656 | `.cache/eval_images/caltech101/Faces_easy_image_0081.jpg` |
| Faces_easy/image_0046 | Faces_easy | Faces | 0.850 | 0.550 | `.cache/eval_images/caltech101/Faces_easy_image_0046.jpg` |

### gqa_yesno · vlm · statement

Most common confusions: no → yes (5); yes → no (3)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| 20611590 | yes | no | 0.672 | 0.836 | `.cache/eval_images/gqa_yesno/20611590.jpg` |
| 201879811 | no | yes | 0.639 | 0.820 | `.cache/eval_images/gqa_yesno/201879811.jpg` |
| 20621841 | yes | no | 0.599 | 0.800 | `.cache/eval_images/gqa_yesno/20621841.jpg` |
| 202100299 | no | yes | 0.508 | 0.754 | `.cache/eval_images/gqa_yesno/202100299.jpg` |
| 201428716 | no | yes | 0.464 | 0.732 | `.cache/eval_images/gqa_yesno/201428716.jpg` |
| 201360614 | yes | no | 0.432 | 0.716 | `.cache/eval_images/gqa_yesno/201360614.jpg` |
| 202119775 | no | yes | 0.347 | 0.673 | `.cache/eval_images/gqa_yesno/202119775.jpg` |
| 20710438 | no | yes | 0.096 | 0.548 | `.cache/eval_images/gqa_yesno/20710438.jpg` |

### pets37 · vlm · independent

Most common confusions: american_bulldog → american_pit_bull_terrier (2); russian_blue → abyssinian (1); ragdoll → birman (1)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| Ragdoll_67 | ragdoll | birman | 0.976 | 0.987 | `.cache/eval_images/pets37/Ragdoll_67.jpg` |
| american_bulldog_98 | american_bulldog | american_pit_bull_terrier | 0.828 | 0.726 | `.cache/eval_images/pets37/american_bulldog_98.jpg` |
| american_bulldog_216 | american_bulldog | american_pit_bull_terrier | 0.729 | 0.636 | `.cache/eval_images/pets37/american_bulldog_216.jpg` |
| beagle_198 | beagle | basset_hound | 0.556 | 0.569 | `.cache/eval_images/pets37/beagle_198.jpg` |
| Russian_Blue_253 | russian_blue | abyssinian | 0.445 | 0.341 | `.cache/eval_images/pets37/Russian_Blue_253.jpg` |

### pope · vlm · statement

Most common confusions: yes → no (6)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| popular_779 | yes | no | 0.814 | 0.907 | `.cache/eval_images/pope/popular_779.jpg` |
| popular_121 | yes | no | 0.791 | 0.895 | `.cache/eval_images/pope/popular_121.jpg` |
| adversarial_1887 | yes | no | 0.581 | 0.791 | `.cache/eval_images/pope/adversarial_1887.jpg` |
| adversarial_2707 | yes | no | 0.405 | 0.702 | `.cache/eval_images/pope/adversarial_2707.jpg` |
| random_1219 | yes | no | 0.256 | 0.628 | `.cache/eval_images/pope/random_1219.jpg` |
| adversarial_941 | yes | no | 0.058 | 0.529 | `.cache/eval_images/pope/adversarial_941.jpg` |


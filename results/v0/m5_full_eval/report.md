# glance eval report · run `20260920T002639Z-2091d3`

**Go/no-go: NO-GO** (judged on held-out test splits; primary local backend: `vlm`)

| Metric | Threshold | Measured | Pass |
| --- | --- | --- | --- |
| Accuracy gap vs frontier baseline | <= 5 points, macro-averaged over suites | +3.1 points over 5 suites | pass |
| ECE after calibration | <= 0.05 per suite (15 equal-mass bins) | worst 0.149 (blur_ladder, sampling floor 0.097); over threshold: blur_ladder, gqa_yesno, pope | FAIL |
| Selective accuracy at 80% coverage | >= baseline full-coverage accuracy | 0.841 vs baseline 0.818 (macro over 5 suites) | pass |
| Permutation invariance (choice) | max probability shift <= 1e-3 (independent) | 0.0e+00 (pets37) | pass |
| Latency, 1 image + 5 questions | recorded; target <= 2000 ms on mps (not a gate) | p50 8604 ms, p95 10817 ms | recorded |

## Run

- Machine: Apple M5, 32.0 GB RAM, device `mps`, tier `apple_32gb`
- `siglip`: `google/siglip2-base-patch16-256@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` (float32, image token budget None)
- `vlm`: `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` (float16, image token budget 768)
- `frontier`: `anthropic/claude-opus-5` (None, image token budget None)
- Harness 0.2.0 at git `55177cc51c`, prompts `p1`, prefix cache off (reference path)
- Prefix cache check: 3.75x on 100 items (1685 statements); argmax agreement 100/100, max |dz| 0.075 vs limit 0.05 -> acceptance NOT met, shipped uncached
- n per suite: requested 500; used pope 500, gqa_yesno 500, pets37 500, caltech101 442, blur_ladder 500 (**trimmed** to fit --max-hours 4.0: ETA at the requested n was 4.15 h; each suite got the same time cap, so only expensive suites lost items); seed 7, calibration/test alternate down the seeded order
- Note: `gqa_yesno` does not run on `siglip` (its questions carry no criteria the backend can use)
- Note: frontier baseline added on 2026-09-20 with `anthropic/claude-opus-5` via `glance baseline` (test split, up to 300 items per suite; picks are not stored, only whether each was right)
- Suite `doctype16`: skipped: RVL-CDIP license is "other" on its dataset card: unclear, so the suite is skipped
- Suite `human_gold`: skipped: no hand-labeled items at gold/human_gold.jsonl

## Per-suite results (test split)

| Suite | Backend | Method | n | Acc | AUROC / F1 / MAE | NLL raw→cal | Brier raw→cal | ECE raw→cal | ECE floor at this n | Sel acc 50/80/90/100 (cal) | Failures | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | frontier | pick | 250 | 0.536 | - | - | - | - | - | - | 0/250 | yes |
| blur_ladder | siglip | statement | 250 | 0.344 | 0.837 | 1.567 → 1.299 | 0.786 → 0.711 | 0.224 → 0.099 | 0.093 | 0.400 / 0.380 / 0.369 / 0.344 | 0/500 | yes |
| blur_ladder | vlm | statement | 250 | 0.496 | 0.670 | 3.426 → 1.119 | 0.847 → 0.614 | 0.392 → 0.149 | 0.097 | 0.552 / 0.535 / 0.511 / 0.496 | 0/500 | yes |
| caltech101 | frontier | pick | 221 | 0.968 | - | - | - | - | - | - | 0/221 | yes |
| caltech101 | siglip | independent | 221 | 0.932 | 0.965 | 0.162 → 0.175 | 0.093 → 0.101 | 0.027 → 0.040 | 0.015 | 1.000 / 1.000 / 0.965 / 0.932 | 0/442 | yes |
| caltech101 | vlm | independent | 221 | 0.919 | 0.912 | 0.753 → 0.297 | 0.127 → 0.121 | 0.052 → 0.034 | 0.025 | 0.982 / 0.972 / 0.955 / 0.919 | 0/442 | yes |
| caltech101 | vlm | letter | 221 | 0.977 | 0.992 | 0.321 → 0.087 | 0.045 → 0.040 | 0.021 → 0.023 | 0.013 | 1.000 / 1.000 / 1.000 / 0.977 | 0/442 | yes |
| gqa_yesno | frontier | pick | 250 | 0.732 | - | - | - | - | - | - | 0/250 | yes |
| gqa_yesno | vlm | statement | 250 | 0.732 → 0.744 | 0.812 | 1.350 → 0.529 | 0.232 → 0.178 | 0.204 → 0.093 | 0.074 | 0.824 / 0.775 / 0.769 / 0.744 | 0/500 | yes |
| pets37 | frontier | pick | 250 | 0.932 | - | - | - | - | - | - | 0/250 | yes |
| pets37 | siglip | independent | 250 | 0.956 | 0.962 | 0.161 → 0.140 | 0.081 → 0.074 | 0.048 → 0.018 | 0.022 | 1.000 / 0.995 / 0.982 / 0.956 | 0/500 | yes |
| pets37 | vlm | independent | 250 | 0.892 | 0.860 | 0.881 → 0.386 | 0.189 → 0.166 | 0.079 → 0.038 | 0.036 | 0.984 / 0.960 / 0.929 / 0.892 | 0/500 | yes |
| pets37 | vlm | letter | 250 | 0.904 | 0.902 | 1.416 → 0.400 | 0.179 → 0.152 | 0.083 → 0.037 | 0.038 | 0.992 / 0.965 / 0.942 / 0.904 | 0/500 | yes |
| pope | frontier | pick | 250 | 0.920 | - | - | - | - | - | - | 0/250 | yes |
| pope | siglip | statement | 250 | 0.588 → 0.732 | 0.779 | 0.951 → 0.574 | 0.295 → 0.194 | 0.255 → 0.091 | 0.088 | 0.784 / 0.775 / 0.751 / 0.732 | 0/500 | yes |
| pope | vlm | statement | 250 | 0.880 → 0.884 | 0.955 | 1.521 → 0.264 | 0.118 → 0.084 | 0.114 → 0.066 | 0.040 | 1.000 / 0.965 / 0.916 / 0.884 | 0/500 | yes |

AUROC for noul, macro-F1 for choice, mean absolute error in levels (expected score vs label) for score. Selective accuracy ranks by `confidence` (noul: `2·|p − 0.5|`). A suite with more than 2% failed items is invalid. `ECE floor at this n` is the ECE a perfectly calibrated predictor with the same confidences would measure on this many items (200 simulated draws): equal-mass ECE is biased upward on small samples, so read each ECE against its floor.

**Flagged: calibrated ECE is not below raw ECE on the test split for:** `caltech101` (siglip, independent: 0.027 → 0.040), `caltech101` (vlm, letter: 0.021 → 0.023)

## Error breakdown (what v1 data this points to)

| Suite | Type | Test n | Error rate | ECE (best available) | Over ECE threshold | Gap to baseline (points) | Top confusions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| caltech101 | choice | 221 | 8.1% | 0.034 | no | +5.0 | Faces_easy → Faces (9); windsor_chair → chair (3); Leopards → wild_cat (2) |
| blur_ladder | score | 250 | 50.4% | 0.149 | yes | +4.0 | 2 → 3 (51); 0 → 1 (49); 1 → 2 (10) |
| pets37 | choice | 250 | 10.8% | 0.038 | no | +4.0 | ragdoll → birman (7); american_bulldog → american_pit_bull_terrier (2); beagle → basset_hound (2) |
| pope | noul | 250 | 11.6% | 0.066 | yes | +3.6 | yes → no (24); no → yes (5) |
| gqa_yesno | noul | 250 | 25.6% | 0.093 | yes | -1.2 | no → yes (43); yes → no (21) |

Primary backend `vlm`, `independent` for choice. Recommended v1 data: by error rate: blur_ladder (score) 50.4%, gqa_yesno (noul) 25.6%, pope (noul) 11.6%; calibrated ECE still over 0.05: blur_ladder 0.149, pope 0.066, gqa_yesno 0.093; largest gap to baseline: caltech101 +5.0 pts, blur_ladder +4.0 pts, pets37 +4.0 pts.

## Calibration

| Version | Backend | Choice method | Type | Fit | Params | n (cal split) | Suites pooled | NLL before→after | ECE before→after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal_8edfd9 | siglip | independent | noul | platt | {'a': 0.5303, 'b': 1.0953} | 250 | pope | 1.000 → 0.621 | 0.261 → 0.094 |
| cal_8edfd9 | siglip | independent | choice | temperature | {'T': 0.673} | 471 | caltech101, pets37 | 0.121 → 0.103 | 0.025 → 0.009 |
| cal_8edfd9 | siglip | independent | score | temperature | {'T': 4.0443} | 250 | blur_ladder | 1.623 → 1.320 | 0.262 → 0.086 |
| cal_1d9418 | vlm | independent | noul | platt | {'a': 0.1836, 'b': 0.5574} | 500 | gqa_yesno, pope | 0.978 → 0.341 | 0.111 → 0.038 |
| cal_1d9418 | vlm | independent | choice | temperature | {'T': 3.0864} | 471 | caltech101, pets37 | 0.530 → 0.279 | 0.064 → 0.013 |
| cal_1d9418 | vlm | independent | score | temperature | {'T': 8.2836} | 250 | blur_ladder | 3.319 → 1.128 | 0.363 → 0.164 |
| cal_eac746 | vlm | letter | noul | platt | {'a': 0.1836, 'b': 0.5574} | 500 | gqa_yesno, pope | 0.978 → 0.341 | 0.111 → 0.038 |
| cal_eac746 | vlm | letter | choice | temperature | {'T': 3.6145} | 471 | caltech101, pets37 | 0.595 → 0.208 | 0.043 → 0.018 |
| cal_eac746 | vlm | letter | score | temperature | {'T': 8.2836} | 250 | blur_ladder | 3.319 → 1.128 | 0.363 → 0.164 |

Pooled fit (what the API applies) against a per-suite fit, on the test split:

| Suite | Backend | Method | ECE raw | ECE pooled fit | ECE per-suite fit | NLL pooled | NLL per-suite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 0.224 | 0.099 | 0.099 | 1.299 | 1.299 |
| blur_ladder | vlm | statement | 0.392 | 0.149 | 0.149 | 1.119 | 1.119 |
| caltech101 | siglip | independent | 0.027 | 0.040 | 0.033 | 0.175 | 0.163 |
| caltech101 | vlm | independent | 0.052 | 0.034 | 0.040 | 0.297 | 0.291 |
| caltech101 | vlm | letter | 0.021 | 0.023 | 0.024 | 0.087 | 0.087 |
| gqa_yesno | vlm | statement | 0.204 | 0.093 | 0.109 | 0.529 | 0.541 |
| pets37 | siglip | independent | 0.048 | 0.018 | 0.018 | 0.140 | 0.142 |
| pets37 | vlm | independent | 0.079 | 0.038 | 0.031 | 0.386 | 0.388 |
| pets37 | vlm | letter | 0.083 | 0.037 | 0.038 | 0.400 | 0.401 |
| pope | siglip | statement | 0.255 | 0.091 | 0.091 | 0.574 | 0.574 |
| pope | vlm | statement | 0.114 | 0.066 | 0.057 | 0.264 | 0.237 |

## `independent` against `letter`

| Suite | Backend | n | Options seen by letter | Acc letter | Acc independent (same options) | Acc independent (all options) | ECE letter raw→cal | ECE independent raw→cal | p50 ms letter / independent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | vlm | 250 | 26 of 37 | 0.904 | 0.912 | 0.892 | 0.083 → 0.037 | 0.079 → 0.038 | 1722 / 8422 |
| caltech101 | vlm | 221 | 26 of 101 | 0.977 | 0.973 | 0.919 | 0.021 → 0.023 | 0.052 → 0.034 | 981 / 12316 |

`letter` is capped at 26 options. On larger suites it sees the true label plus 25 seeded random distractors; `independent (same options)` restricts the independent logits to that same subset, so the two columns are comparable.

Permutation sensitivity (test items × random option orders, shift in raw probabilities):

| Unit | Requests | max abs Δp | mean abs Δp | Choice flips |
| --- | --- | --- | --- | --- |
| pets37 · siglip · independent | 90 | 0.0e+00 | 0.0e+00 | 0 |
| pets37 · vlm · independent | 90 | 0.0e+00 | 0.0e+00 | 0 |
| pets37 · vlm · letter | 90 | 9.7e-01 | 3.4e-02 | 3 |
| caltech101 · siglip · independent | 90 | 0.0e+00 | 0.0e+00 | 0 |
| caltech101 · vlm · independent | 90 | 0.0e+00 | 0.0e+00 | 0 |
| caltech101 · vlm · letter | 90 | 0.0e+00 | 0.0e+00 | 0 |

## Local against the frontier baseline

| Suite | Baseline n | Baseline acc | Local backend | Local acc (same items) | Gap (points) | Local sel acc @80% (full test split) |
| --- | --- | --- | --- | --- | --- | --- |
| pope | 250 | 0.920 | siglip | 0.732 | +18.8 | 0.775 |
| pope | 250 | 0.920 | vlm | 0.884 | +3.6 | 0.965 |
| gqa_yesno | 250 | 0.732 | vlm | 0.744 | -1.2 | 0.775 |
| pets37 | 250 | 0.932 | siglip | 0.956 | -2.4 | 0.995 |
| pets37 | 250 | 0.932 | vlm | 0.892 | +4.0 | 0.960 |
| caltech101 | 221 | 0.968 | siglip | 0.932 | +3.6 | 1.000 |
| caltech101 | 221 | 0.968 | vlm | 0.919 | +5.0 | 0.972 |
| blur_ladder | 250 | 0.536 | siglip | 0.344 | +19.2 | 0.380 |
| blur_ladder | 250 | 0.536 | vlm | 0.496 | +4.0 | 0.535 |

## Latency and throughput

| Backend | Images | Questions | Statements | p50 ms | p95 ms | Requests |
| --- | --- | --- | --- | --- | --- | --- |
| siglip | 1 | 5 | 11 | 23 | 24 | 20 |
| vlm | 1 | 5 | 11 | 8604 | 10817 | 20 |

| Suite | Backend | Method | p50 ms / request | p95 ms / request | Statements / s | Mean image tokens | off_mass mean | off_mass p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur_ladder | siglip | statement | 21 | 22 | 187.0 | 256 | - | - |
| blur_ladder | vlm | statement | 669 | 900 | 5.6 | 105 | 0.00364 | 0.01385 |
| caltech101 | siglip | independent | 21 | 23 | 4462.6 | 256 | - | - |
| caltech101 | vlm | independent | 12315 | 16531 | 7.6 | 79 | 0.00390 | 0.01400 |
| caltech101 | vlm | letter | 981 | 1115 | 4.1 | 79 | 0.00393 | 0.01400 |
| gqa_yesno | vlm | statement | 438 | 473 | 2.4 | 266 | 0.00388 | 0.01382 |
| pets37 | siglip | independent | 22 | 22 | 1662.0 | 256 | - | - |
| pets37 | vlm | independent | 8941 | 11923 | 4.0 | 179 | 0.00390 | 0.01396 |
| pets37 | vlm | letter | 1736 | 2063 | 2.2 | 179 | 0.00374 | 0.01386 |
| pope | siglip | statement | 23 | 34 | 80.1 | 256 | - | - |
| pope | vlm | statement | 458 | 533 | 2.3 | 275 | 0.00422 | 0.01349 |

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

Most common confusions: 2 → 3 (51); 0 → 1 (49); 1 → 2 (10)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| gerenuk/image_0018_blur2 | 2 | 3 | 0.256 | 0.560 | `.cache/eval_images/blur_ladder/gerenuk_image_0018_blur2.jpg` |
| airplanes/image_0442_blur2 | 2 | 3 | 0.244 | 0.530 | `.cache/eval_images/blur_ladder/airplanes_image_0442_blur2.jpg` |
| helicopter/image_0082_blur2 | 2 | 3 | 0.238 | 0.523 | `.cache/eval_images/blur_ladder/helicopter_image_0082_blur2.jpg` |
| airplanes/image_0604_blur2 | 2 | 3 | 0.235 | 0.513 | `.cache/eval_images/blur_ladder/airplanes_image_0604_blur2.jpg` |
| beaver/image_0026_blur2 | 2 | 3 | 0.233 | 0.538 | `.cache/eval_images/blur_ladder/beaver_image_0026_blur2.jpg` |
| airplanes/image_0060_blur0 | 0 | 1 | 0.231 | 0.575 | `.cache/eval_images/blur_ladder/airplanes_image_0060_blur0.jpg` |
| sea_horse/image_0048_blur2 | 2 | 3 | 0.227 | 0.515 | `.cache/eval_images/blur_ladder/sea_horse_image_0048_blur2.jpg` |
| bass/image_0008_blur2 | 2 | 3 | 0.226 | 0.499 | `.cache/eval_images/blur_ladder/bass_image_0008_blur2.jpg` |
| garfield/image_0018_blur2 | 2 | 3 | 0.225 | 0.490 | `.cache/eval_images/blur_ladder/garfield_image_0018_blur2.jpg` |
| cup/image_0025_blur2 | 2 | 3 | 0.224 | 0.510 | `.cache/eval_images/blur_ladder/cup_image_0025_blur2.jpg` |
| Faces/image_0295_blur2 | 2 | 3 | 0.222 | 0.488 | `.cache/eval_images/blur_ladder/Faces_image_0295_blur2.jpg` |
| ceiling_fan/image_0029_blur2 | 2 | 3 | 0.218 | 0.480 | `.cache/eval_images/blur_ladder/ceiling_fan_image_0029_blur2.jpg` |
| helicopter/image_0036_blur2 | 2 | 3 | 0.216 | 0.490 | `.cache/eval_images/blur_ladder/helicopter_image_0036_blur2.jpg` |
| airplanes/image_0681_blur2 | 2 | 3 | 0.216 | 0.476 | `.cache/eval_images/blur_ladder/airplanes_image_0681_blur2.jpg` |
| accordion/image_0021_blur2 | 2 | 3 | 0.214 | 0.488 | `.cache/eval_images/blur_ladder/accordion_image_0021_blur2.jpg` |
| Faces/image_0250_blur0 | 0 | 1 | 0.212 | 0.590 | `.cache/eval_images/blur_ladder/Faces_image_0250_blur0.jpg` |
| airplanes/image_0783_blur2 | 2 | 3 | 0.210 | 0.476 | `.cache/eval_images/blur_ladder/airplanes_image_0783_blur2.jpg` |
| lamp/image_0046_blur2 | 2 | 3 | 0.210 | 0.485 | `.cache/eval_images/blur_ladder/lamp_image_0046_blur2.jpg` |
| airplanes/image_0379_blur2 | 2 | 3 | 0.210 | 0.463 | `.cache/eval_images/blur_ladder/airplanes_image_0379_blur2.jpg` |
| dolphin/image_0026_blur2 | 2 | 3 | 0.207 | 0.453 | `.cache/eval_images/blur_ladder/dolphin_image_0026_blur2.jpg` |

### caltech101 · vlm · independent

Most common confusions: Faces_easy → Faces (9); windsor_chair → chair (3); Leopards → wild_cat (2)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| windsor_chair/image_0021 | windsor_chair | chair | 0.999 | 1.000 | `.cache/eval_images/caltech101/windsor_chair_image_0021.jpg` |
| crayfish/image_0027 | crayfish | lobster | 0.998 | 0.999 | `.cache/eval_images/caltech101/crayfish_image_0027.jpg` |
| windsor_chair/image_0007 | windsor_chair | chair | 0.998 | 0.999 | `.cache/eval_images/caltech101/windsor_chair_image_0007.jpg` |
| Leopards/image_0021 | Leopards | wild_cat | 0.993 | 0.997 | `.cache/eval_images/caltech101/Leopards_image_0021.jpg` |
| Leopards/image_0102 | Leopards | wild_cat | 0.989 | 0.995 | `.cache/eval_images/caltech101/Leopards_image_0102.jpg` |
| crocodile_head/image_0005 | crocodile_head | crocodile | 0.885 | 0.777 | `.cache/eval_images/caltech101/crocodile_head_image_0005.jpg` |
| windsor_chair/image_0019 | windsor_chair | chair | 0.871 | 0.718 | `.cache/eval_images/caltech101/windsor_chair_image_0019.jpg` |
| ketch/image_0015 | ketch | schooner | 0.869 | 0.720 | `.cache/eval_images/caltech101/ketch_image_0015.jpg` |
| Faces_easy/image_0241 | Faces_easy | Faces | 0.852 | 0.655 | `.cache/eval_images/caltech101/Faces_easy_image_0241.jpg` |
| Faces_easy/image_0012 | Faces_easy | Faces | 0.848 | 0.683 | `.cache/eval_images/caltech101/Faces_easy_image_0012.jpg` |
| Faces_easy/image_0112 | Faces_easy | Faces | 0.847 | 0.612 | `.cache/eval_images/caltech101/Faces_easy_image_0112.jpg` |
| Faces_easy/image_0081 | Faces_easy | Faces | 0.847 | 0.603 | `.cache/eval_images/caltech101/Faces_easy_image_0081.jpg` |
| Faces_easy/image_0046 | Faces_easy | Faces | 0.844 | 0.532 | `.cache/eval_images/caltech101/Faces_easy_image_0046.jpg` |
| Faces_easy/image_0186 | Faces_easy | Faces | 0.842 | 0.514 | `.cache/eval_images/caltech101/Faces_easy_image_0186.jpg` |
| Faces_easy/image_0173 | Faces_easy | Faces | 0.841 | 0.530 | `.cache/eval_images/caltech101/Faces_easy_image_0173.jpg` |
| Faces_easy/image_0205 | Faces_easy | Faces | 0.840 | 0.518 | `.cache/eval_images/caltech101/Faces_easy_image_0205.jpg` |
| Faces_easy/image_0266 | Faces_easy | Faces | 0.830 | 0.516 | `.cache/eval_images/caltech101/Faces_easy_image_0266.jpg` |
| cougar_face/image_0067 | cougar_face | wild_cat | 0.800 | 0.583 | `.cache/eval_images/caltech101/cougar_face_image_0067.jpg` |

### gqa_yesno · vlm · statement

Most common confusions: no → yes (43); yes → no (21)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| 20456727 | no | yes | 0.886 | 0.943 | `.cache/eval_images/gqa_yesno/20456727.jpg` |
| 201407334 | no | yes | 0.878 | 0.939 | `.cache/eval_images/gqa_yesno/201407334.jpg` |
| 201795290 | yes | no | 0.820 | 0.910 | `.cache/eval_images/gqa_yesno/201795290.jpg` |
| 201061298 | no | yes | 0.804 | 0.902 | `.cache/eval_images/gqa_yesno/201061298.jpg` |
| 201879811 | no | yes | 0.779 | 0.890 | `.cache/eval_images/gqa_yesno/201879811.jpg` |
| 201711321 | no | yes | 0.770 | 0.885 | `.cache/eval_images/gqa_yesno/201711321.jpg` |
| 20923159 | yes | no | 0.683 | 0.841 | `.cache/eval_images/gqa_yesno/20923159.jpg` |
| 201951877 | no | yes | 0.679 | 0.840 | `.cache/eval_images/gqa_yesno/201951877.jpg` |
| 202100299 | no | yes | 0.675 | 0.837 | `.cache/eval_images/gqa_yesno/202100299.jpg` |
| 201342325 | yes | no | 0.662 | 0.831 | `.cache/eval_images/gqa_yesno/201342325.jpg` |
| 20621983 | no | yes | 0.642 | 0.821 | `.cache/eval_images/gqa_yesno/20621983.jpg` |
| 201428716 | no | yes | 0.638 | 0.819 | `.cache/eval_images/gqa_yesno/201428716.jpg` |
| 201956853 | no | yes | 0.630 | 0.815 | `.cache/eval_images/gqa_yesno/201956853.jpg` |
| 201972699 | yes | no | 0.629 | 0.815 | `.cache/eval_images/gqa_yesno/201972699.jpg` |
| 201079738 | no | yes | 0.626 | 0.813 | `.cache/eval_images/gqa_yesno/201079738.jpg` |
| 20611590 | yes | no | 0.626 | 0.813 | `.cache/eval_images/gqa_yesno/20611590.jpg` |
| 202133566 | no | yes | 0.589 | 0.794 | `.cache/eval_images/gqa_yesno/202133566.jpg` |
| 202174161 | no | yes | 0.573 | 0.787 | `.cache/eval_images/gqa_yesno/202174161.jpg` |
| 20511447 | yes | no | 0.559 | 0.780 | `.cache/eval_images/gqa_yesno/20511447.jpg` |
| 201878388 | yes | no | 0.553 | 0.776 | `.cache/eval_images/gqa_yesno/201878388.jpg` |

### pets37 · vlm · independent

Most common confusions: ragdoll → birman (7); american_bulldog → american_pit_bull_terrier (2); beagle → basset_hound (2)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| chihuahua_85 | chihuahua | miniature_pinscher | 0.986 | 0.994 | `.cache/eval_images/pets37/chihuahua_85.jpg` |
| Ragdoll_252 | ragdoll | birman | 0.980 | 0.990 | `.cache/eval_images/pets37/Ragdoll_252.jpg` |
| scottish_terrier_96 | scottish_terrier | wheaten_terrier | 0.976 | 0.986 | `.cache/eval_images/pets37/scottish_terrier_96.jpg` |
| chihuahua_55 | chihuahua | miniature_pinscher | 0.975 | 0.986 | `.cache/eval_images/pets37/chihuahua_55.jpg` |
| Ragdoll_67 | ragdoll | birman | 0.896 | 0.923 | `.cache/eval_images/pets37/Ragdoll_67.jpg` |
| beagle_69 | beagle | basset_hound | 0.870 | 0.829 | `.cache/eval_images/pets37/beagle_69.jpg` |
| Siamese_228 | siamese | birman | 0.840 | 0.806 | `.cache/eval_images/pets37/Siamese_228.jpg` |
| Ragdoll_36 | ragdoll | birman | 0.837 | 0.787 | `.cache/eval_images/pets37/Ragdoll_36.jpg` |
| english_setter_2 | english_setter | english_cocker_spaniel | 0.801 | 0.702 | `.cache/eval_images/pets37/english_setter_2.jpg` |
| Ragdoll_88 | ragdoll | birman | 0.798 | 0.581 | `.cache/eval_images/pets37/Ragdoll_88.jpg` |
| staffordshire_bull_terrier_53 | staffordshire_bull_terrier | american_pit_bull_terrier | 0.785 | 0.750 | `.cache/eval_images/pets37/staffordshire_bull_terrier_53.jpg` |
| american_bulldog_98 | american_bulldog | american_pit_bull_terrier | 0.785 | 0.640 | `.cache/eval_images/pets37/american_bulldog_98.jpg` |
| Ragdoll_20 | ragdoll | birman | 0.779 | 0.575 | `.cache/eval_images/pets37/Ragdoll_20.jpg` |
| staffordshire_bull_terrier_93 | staffordshire_bull_terrier | american_pit_bull_terrier | 0.767 | 0.643 | `.cache/eval_images/pets37/staffordshire_bull_terrier_93.jpg` |
| Ragdoll_224 | ragdoll | birman | 0.750 | 0.540 | `.cache/eval_images/pets37/Ragdoll_224.jpg` |
| Ragdoll_26 | ragdoll | birman | 0.679 | 0.467 | `.cache/eval_images/pets37/Ragdoll_26.jpg` |
| american_bulldog_216 | american_bulldog | american_pit_bull_terrier | 0.659 | 0.511 | `.cache/eval_images/pets37/american_bulldog_216.jpg` |
| saint_bernard_57 | saint_bernard | beagle | 0.647 | 0.722 | `.cache/eval_images/pets37/saint_bernard_57.jpg` |
| leonberger_82 | leonberger | saint_bernard | 0.624 | 0.540 | `.cache/eval_images/pets37/leonberger_82.jpg` |
| Abyssinian_40 | abyssinian | egyptian_mau | 0.623 | 0.620 | `.cache/eval_images/pets37/Abyssinian_40.jpg` |

### pope · vlm · statement

Most common confusions: yes → no (24); no → yes (5)

| Item | True | Predicted | Confidence | Top probability | Image |
| --- | --- | --- | --- | --- | --- |
| adversarial_1113 | yes | no | 0.911 | 0.956 | `.cache/eval_images/pope/adversarial_1113.jpg` |
| random_2269 | yes | no | 0.905 | 0.952 | `.cache/eval_images/pope/random_2269.jpg` |
| adversarial_357 | yes | no | 0.901 | 0.950 | `.cache/eval_images/pope/adversarial_357.jpg` |
| popular_179 | yes | no | 0.900 | 0.950 | `.cache/eval_images/pope/popular_179.jpg` |
| popular_1642 | no | yes | 0.890 | 0.945 | `.cache/eval_images/pope/popular_1642.jpg` |
| random_2955 | yes | no | 0.829 | 0.915 | `.cache/eval_images/pope/random_2955.jpg` |
| popular_779 | yes | no | 0.801 | 0.901 | `.cache/eval_images/pope/popular_779.jpg` |
| random_2417 | yes | no | 0.791 | 0.895 | `.cache/eval_images/pope/random_2417.jpg` |
| adversarial_1593 | yes | no | 0.782 | 0.891 | `.cache/eval_images/pope/adversarial_1593.jpg` |
| popular_121 | yes | no | 0.773 | 0.887 | `.cache/eval_images/pope/popular_121.jpg` |
| adversarial_2385 | yes | no | 0.771 | 0.886 | `.cache/eval_images/pope/adversarial_2385.jpg` |
| adversarial_627 | yes | no | 0.693 | 0.847 | `.cache/eval_images/pope/adversarial_627.jpg` |
| adversarial_884 | no | yes | 0.642 | 0.821 | `.cache/eval_images/pope/adversarial_884.jpg` |
| adversarial_23 | yes | no | 0.611 | 0.806 | `.cache/eval_images/pope/adversarial_23.jpg` |
| popular_2722 | no | yes | 0.606 | 0.803 | `.cache/eval_images/pope/popular_2722.jpg` |
| adversarial_505 | yes | no | 0.605 | 0.802 | `.cache/eval_images/pope/adversarial_505.jpg` |
| adversarial_2603 | yes | no | 0.599 | 0.800 | `.cache/eval_images/pope/adversarial_2603.jpg` |
| popular_873 | yes | no | 0.582 | 0.791 | `.cache/eval_images/pope/popular_873.jpg` |
| popular_35 | yes | no | 0.527 | 0.764 | `.cache/eval_images/pope/popular_35.jpg` |
| adversarial_1887 | yes | no | 0.512 | 0.756 | `.cache/eval_images/pope/adversarial_1887.jpg` |


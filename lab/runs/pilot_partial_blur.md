# Pilot (partial: blur only, DEV)

DEV numbers: fit on the first half of the calibration split, reported on its second half. The test split is untouched.

Bar for "impressive": accuracy >= 85%, MAE <= 0.25 levels, ECE <= 0.05 (15 equal-mass bins; read each ECE against its sampling floor).

## Accuracy by method and scale (best calibration for each, chosen on the fit split)

| Method | blur | mean | passes | p50 ms |
| --- | --- | --- | --- | --- |
| `anchors_cumulative@0:012` | 0.762 | **0.762** | 3 | 1079 |
| `anchors_cumulative@0:013` | 0.762 | **0.762** | 3 | 1079 |
| `anchors_cumulative@0:123` | 0.738 | **0.738** | 3 | 1078 |
| `anchors_digits@0:012` | 0.725 | **0.725** | 1 | 207 |
| `anchors_digits@0:013` | 0.887 | **0.887** | 1 | 206 |
| `anchors_digits@0:123` | 0.762 | **0.762** | 1 | 207 |
| `cumulative` | 0.787 | **0.787** | 3 | 336 |
| `digits` | 0.600 | **0.600** | 1 | 159 |
| `independent` | 0.900 | **0.900** | 4 | 447 |

## Every method, scale and calibration

| Scale | Method | Calibration | chosen | n | Acc | Within 1 | MAE | Spearman | NLL | ECE | ECE floor | Meets bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | `anchors_cumulative@0:012` | raw |  | 80 | 0.537 | 0.750 | 0.748 | 0.877 | 7.800 | 0.417 | 0.047 |  |
| blur | `anchors_cumulative@0:012` | platt/threshold | yes | 80 | 0.762 | 1.000 | 0.304 | 0.922 | 1.112 | 0.119 | 0.102 |  |
| blur | `anchors_cumulative@0:013` | raw |  | 80 | 0.588 | 0.950 | 0.427 | 0.843 | 8.177 | 0.346 | 0.055 |  |
| blur | `anchors_cumulative@0:013` | platt/threshold | yes | 80 | 0.762 | 1.000 | 0.398 | 0.904 | 0.594 | 0.199 | 0.149 |  |
| blur | `anchors_cumulative@0:123` | raw |  | 80 | 0.575 | 0.875 | 0.496 | 0.836 | 8.437 | 0.329 | 0.072 |  |
| blur | `anchors_cumulative@0:123` | platt/threshold | yes | 80 | 0.738 | 1.000 | 0.428 | 0.886 | 0.635 | 0.151 | 0.151 |  |
| blur | `anchors_digits@0:012` | raw |  | 80 | 0.537 | 0.938 | 0.526 | 0.937 | 4.375 | 0.420 | 0.028 |  |
| blur | `anchors_digits@0:012` | T |  | 80 | 0.537 | 0.938 | 0.718 | 0.939 | 1.058 | 0.223 | 0.174 |  |
| blur | `anchors_digits@0:012` | bias+T | yes | 80 | 0.725 | 1.000 | 0.380 | 0.931 | 0.760 | 0.179 | 0.125 |  |
| blur | `anchors_digits@0:013` | raw |  | 80 | 0.650 | 0.963 | 0.382 | 0.952 | 1.539 | 0.305 | 0.032 |  |
| blur | `anchors_digits@0:013` | T |  | 80 | 0.650 | 0.963 | 0.469 | 0.951 | 0.700 | 0.148 | 0.155 |  |
| blur | `anchors_digits@0:013` | bias+T | yes | 80 | 0.887 | 1.000 | 0.196 | 0.939 | 0.511 | 0.102 | 0.089 |  |
| blur | `anchors_digits@0:123` | raw |  | 80 | 0.650 | 1.000 | 0.339 | 0.940 | 1.971 | 0.307 | 0.025 |  |
| blur | `anchors_digits@0:123` | T |  | 80 | 0.650 | 1.000 | 0.413 | 0.925 | 0.723 | 0.203 | 0.159 |  |
| blur | `anchors_digits@0:123` | bias+T | yes | 80 | 0.762 | 1.000 | 0.329 | 0.926 | 0.630 | 0.158 | 0.110 |  |
| blur | `cumulative` | raw |  | 80 | 0.575 | 0.950 | 0.440 | 0.941 | 8.609 | 0.352 | 0.060 |  |
| blur | `cumulative` | platt/threshold | yes | 80 | 0.787 | 1.000 | 0.326 | 0.941 | 0.523 | 0.111 | 0.115 |  |
| blur | `digits` | raw |  | 80 | 0.463 | 1.000 | 0.548 | 0.953 | 4.425 | 0.505 | 0.024 |  |
| blur | `digits` | T |  | 80 | 0.463 | 1.000 | 0.676 | 0.944 | 1.023 | 0.223 | 0.175 |  |
| blur | `digits` | bias+T | yes | 80 | 0.600 | 1.000 | 0.473 | 0.941 | 0.928 | 0.224 | 0.140 |  |
| blur | `independent` | raw |  | 80 | 0.662 | 1.000 | 0.362 | 0.958 | 1.323 | 0.265 | 0.055 |  |
| blur | `independent` | T |  | 80 | 0.662 | 1.000 | 0.480 | 0.962 | 0.756 | 0.157 | 0.154 |  |
| blur | `independent` | bias+T | yes | 80 | 0.900 | 1.000 | 0.132 | 0.959 | 0.276 | 0.031 | 0.052 | YES |

## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)

`blur` · `anchors_cumulative@0:012` · platt/threshold: [21, 4, 0, 0] / [1, 12, 5, 0] / [0, 2, 9, 5] / [0, 0, 2, 19]
`blur` · `anchors_cumulative@0:013` · platt/threshold: [20, 5, 0, 0] / [1, 11, 6, 0] / [0, 2, 10, 4] / [0, 0, 1, 20]
`blur` · `anchors_cumulative@0:123` · platt/threshold: [18, 7, 0, 0] / [2, 15, 1, 0] / [0, 2, 9, 5] / [0, 0, 4, 17]
`blur` · `anchors_digits@0:012` · bias+T: [19, 6, 0, 0] / [1, 10, 7, 0] / [0, 1, 8, 7] / [0, 0, 0, 21]
`blur` · `anchors_digits@0:013` · bias+T: [21, 4, 0, 0] / [1, 17, 0, 0] / [0, 2, 12, 2] / [0, 0, 0, 21]
`blur` · `anchors_digits@0:123` · bias+T: [15, 10, 0, 0] / [1, 16, 1, 0] / [0, 2, 9, 5] / [0, 0, 0, 21]
`blur` · `cumulative` · platt/threshold: [23, 2, 0, 0] / [0, 8, 10, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`blur` · `digits` · bias+T: [15, 10, 0, 0] / [0, 5, 13, 0] / [0, 0, 7, 9] / [0, 0, 0, 21]
`blur` · `independent` · bias+T: [23, 2, 0, 0] / [0, 18, 0, 0] / [0, 1, 12, 3] / [0, 0, 2, 19]

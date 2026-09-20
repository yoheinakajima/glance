# Score lab

DEV numbers: fit on the first half of the calibration split, reported on its second half. The test split is untouched.

Bar for "impressive": accuracy >= 85%, MAE <= 0.25 levels, ECE <= 0.05 (15 equal-mass bins; read each ECE against its sampling floor).

## Accuracy by method and scale (best calibration for each, chosen on the fit split)

| Method | blur | jpeg | noise | mean | passes | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| `digits` | 0.838 | 0.613 | 0.675 | **0.708** | 1 | 159 |
| `ens(all6)` | 0.900 | 0.750 | 0.775 | **0.808** | 16 | 2119 |
| `ens(digits+zoom_digits)` | 0.838 | 0.700 | 0.775 | **0.771** | 2 | 324 |
| `independent` | 0.900 | 0.550 | 0.675 | **0.708** | 4 | 447 |
| `zoom_cumulative` | 0.700 | 0.613 | 0.775 | **0.696** | 3 | 355 |
| `zoom_digits` | 0.863 | 0.713 | 0.750 | **0.775** | 1 | 164 |

## Every method, scale and calibration

| Scale | Method | Calibration | chosen | n | Acc | Within 1 | MAE | Spearman | NLL | ECE | ECE floor | Meets bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | `digits` | raw |  | 80 | 0.463 | 1.000 | 0.548 | 0.953 | 4.425 | 0.505 | 0.024 |  |
| blur | `digits` | T |  | 80 | 0.463 | 1.000 | 0.676 | 0.944 | 1.023 | 0.223 | 0.175 |  |
| blur | `digits` | bias+T |  | 80 | 0.600 | 1.000 | 0.473 | 0.941 | 0.928 | 0.224 | 0.140 |  |
| blur | `digits` | matrix | yes | 80 | 0.838 | 1.000 | 0.367 | 0.950 | 0.527 | 0.185 | 0.147 |  |
| blur | `ens(all6)` | matrix | yes | 80 | 0.900 | 1.000 | 0.234 | 0.965 | 0.342 | 0.171 | 0.124 |  |
| blur | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.838 | 1.000 | 0.294 | 0.960 | 0.427 | 0.149 | 0.141 |  |
| blur | `independent` | raw |  | 80 | 0.662 | 1.000 | 0.362 | 0.958 | 1.323 | 0.265 | 0.055 |  |
| blur | `independent` | T |  | 80 | 0.662 | 1.000 | 0.480 | 0.962 | 0.756 | 0.157 | 0.154 |  |
| blur | `independent` | bias+T | yes | 80 | 0.900 | 1.000 | 0.132 | 0.959 | 0.276 | 0.031 | 0.052 | YES |
| blur | `independent` | matrix |  | 80 | 0.838 | 1.000 | 0.384 | 0.959 | 0.540 | 0.227 | 0.160 |  |
| blur | `zoom_cumulative` | raw |  | 80 | 0.550 | 0.938 | 0.497 | 0.886 | 8.082 | 0.259 | 0.129 |  |
| blur | `zoom_cumulative` | platt/threshold |  | 80 | 0.775 | 1.000 | 0.278 | 0.938 | 0.676 | 0.115 | 0.089 |  |
| blur | `zoom_cumulative` | matrix | yes | 80 | 0.700 | 1.000 | 0.450 | 0.932 | 0.726 | 0.246 | 0.168 |  |
| blur | `zoom_digits` | raw |  | 80 | 0.287 | 0.938 | 0.769 | 0.961 | 6.648 | 0.671 | 0.031 |  |
| blur | `zoom_digits` | T |  | 80 | 0.287 | 0.938 | 0.846 | 0.956 | 1.281 | 0.361 | 0.176 |  |
| blur | `zoom_digits` | bias+T |  | 80 | 0.487 | 1.000 | 0.646 | 0.935 | 1.098 | 0.339 | 0.166 |  |
| blur | `zoom_digits` | matrix | yes | 80 | 0.863 | 1.000 | 0.357 | 0.961 | 0.529 | 0.223 | 0.158 |  |
| jpeg | `digits` | raw |  | 80 | 0.388 | 0.950 | 0.662 | 0.795 | 4.279 | 0.530 | 0.054 |  |
| jpeg | `digits` | T |  | 80 | 0.388 | 0.950 | 0.682 | 0.809 | 1.167 | 0.117 | 0.174 |  |
| jpeg | `digits` | bias+T |  | 80 | 0.550 | 0.950 | 0.579 | 0.817 | 1.064 | 0.141 | 0.175 |  |
| jpeg | `digits` | matrix | yes | 80 | 0.613 | 0.912 | 0.504 | 0.811 | 0.910 | 0.227 | 0.170 |  |
| jpeg | `ens(all6)` | matrix | yes | 80 | 0.750 | 0.988 | 0.337 | 0.917 | 0.637 | 0.142 | 0.153 |  |
| jpeg | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.700 | 0.963 | 0.369 | 0.918 | 0.689 | 0.153 | 0.156 |  |
| jpeg | `independent` | raw |  | 80 | 0.375 | 0.900 | 0.701 | 0.822 | 2.418 | 0.462 | 0.095 |  |
| jpeg | `independent` | T |  | 80 | 0.375 | 0.900 | 0.686 | 0.829 | 1.168 | 0.192 | 0.170 |  |
| jpeg | `independent` | bias+T |  | 80 | 0.613 | 0.963 | 0.570 | 0.823 | 0.978 | 0.140 | 0.169 |  |
| jpeg | `independent` | matrix | yes | 80 | 0.550 | 0.938 | 0.535 | 0.822 | 0.954 | 0.176 | 0.174 |  |
| jpeg | `zoom_cumulative` | raw |  | 80 | 0.588 | 0.963 | 0.346 | 0.911 | 1.587 | 0.233 | 0.136 |  |
| jpeg | `zoom_cumulative` | platt/threshold |  | 80 | 0.713 | 0.988 | 0.330 | 0.921 | 0.688 | 0.094 | 0.134 |  |
| jpeg | `zoom_cumulative` | matrix | yes | 80 | 0.613 | 0.988 | 0.415 | 0.923 | 0.830 | 0.250 | 0.166 |  |
| jpeg | `zoom_digits` | raw |  | 80 | 0.362 | 0.988 | 0.648 | 0.938 | 3.582 | 0.526 | 0.063 |  |
| jpeg | `zoom_digits` | T |  | 80 | 0.362 | 0.988 | 0.754 | 0.935 | 1.164 | 0.261 | 0.174 |  |
| jpeg | `zoom_digits` | bias+T |  | 80 | 0.787 | 0.975 | 0.360 | 0.928 | 0.653 | 0.154 | 0.155 |  |
| jpeg | `zoom_digits` | matrix | yes | 80 | 0.713 | 0.975 | 0.369 | 0.936 | 0.746 | 0.209 | 0.165 |  |
| noise | `digits` | raw |  | 80 | 0.675 | 1.000 | 0.370 | 0.899 | 1.432 | 0.306 | 0.068 |  |
| noise | `digits` | T | yes | 80 | 0.675 | 1.000 | 0.397 | 0.901 | 0.780 | 0.178 | 0.136 |  |
| noise | `digits` | bias+T |  | 80 | 0.738 | 1.000 | 0.317 | 0.900 | 0.735 | 0.206 | 0.114 |  |
| noise | `digits` | matrix |  | 80 | 0.713 | 1.000 | 0.391 | 0.906 | 0.640 | 0.152 | 0.163 |  |
| noise | `ens(all6)` | matrix | yes | 80 | 0.775 | 1.000 | 0.279 | 0.926 | 0.478 | 0.159 | 0.141 |  |
| noise | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.775 | 1.000 | 0.328 | 0.914 | 0.549 | 0.178 | 0.158 |  |
| noise | `independent` | raw |  | 80 | 0.500 | 1.000 | 0.487 | 0.891 | 1.697 | 0.415 | 0.067 |  |
| noise | `independent` | T |  | 80 | 0.500 | 1.000 | 0.470 | 0.904 | 0.897 | 0.220 | 0.155 |  |
| noise | `independent` | bias+T | yes | 80 | 0.675 | 1.000 | 0.360 | 0.886 | 0.868 | 0.165 | 0.108 |  |
| noise | `independent` | matrix |  | 80 | 0.637 | 1.000 | 0.413 | 0.891 | 0.697 | 0.194 | 0.167 |  |
| noise | `zoom_cumulative` | raw |  | 80 | 0.525 | 0.950 | 0.357 | 0.921 | 7.181 | 0.371 | 0.061 |  |
| noise | `zoom_cumulative` | platt/threshold | yes | 80 | 0.775 | 0.988 | 0.236 | 0.911 | 1.127 | 0.159 | 0.055 |  |
| noise | `zoom_cumulative` | matrix |  | 80 | 0.812 | 1.000 | 0.375 | 0.933 | 0.719 | 0.294 | 0.173 |  |
| noise | `zoom_digits` | raw |  | 80 | 0.400 | 1.000 | 0.559 | 0.915 | 2.381 | 0.506 | 0.068 |  |
| noise | `zoom_digits` | T |  | 80 | 0.400 | 1.000 | 0.547 | 0.920 | 0.974 | 0.238 | 0.172 |  |
| noise | `zoom_digits` | bias+T | yes | 80 | 0.750 | 1.000 | 0.285 | 0.913 | 0.578 | 0.134 | 0.114 |  |
| noise | `zoom_digits` | matrix |  | 80 | 0.800 | 0.988 | 0.351 | 0.924 | 0.630 | 0.208 | 0.165 |  |

## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)

`blur` · `digits` · matrix: [20, 5, 0, 0] / [0, 18, 0, 0] / [0, 1, 12, 3] / [0, 0, 4, 17]
`blur` · `ens(all6)` · matrix: [22, 3, 0, 0] / [0, 18, 0, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`blur` · `ens(digits+zoom_digits)` · matrix: [19, 6, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 3, 18]
`blur` · `independent` · bias+T: [23, 2, 0, 0] / [0, 18, 0, 0] / [0, 1, 12, 3] / [0, 0, 2, 19]
`blur` · `zoom_cumulative` · matrix: [23, 2, 0, 0] / [0, 15, 3, 0] / [0, 1, 15, 0] / [0, 0, 18, 3]
`blur` · `zoom_digits` · matrix: [19, 6, 0, 0] / [0, 18, 0, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`jpeg` · `digits` · matrix: [15, 0, 4, 0] / [7, 3, 14, 0] / [3, 0, 14, 2] / [0, 0, 1, 17]
`jpeg` · `ens(all6)` · matrix: [18, 0, 1, 0] / [1, 11, 12, 0] / [0, 3, 15, 1] / [0, 0, 2, 16]
`jpeg` · `ens(digits+zoom_digits)` · matrix: [16, 0, 3, 0] / [1, 9, 14, 0] / [0, 3, 15, 1] / [0, 0, 2, 16]
`jpeg` · `independent` · matrix: [14, 1, 4, 0] / [6, 3, 15, 0] / [1, 2, 11, 5] / [0, 0, 2, 16]
`jpeg` · `zoom_cumulative` · matrix: [18, 0, 1, 0] / [4, 0, 20, 0] / [0, 0, 15, 4] / [0, 0, 2, 16]
`jpeg` · `zoom_digits` · matrix: [18, 0, 1, 0] / [1, 5, 17, 1] / [0, 0, 17, 2] / [0, 0, 1, 17]
`noise` · `digits` · T: [13, 8, 0, 0] / [5, 15, 0, 0] / [0, 5, 14, 0] / [0, 0, 8, 12]
`noise` · `ens(all6)` · matrix: [14, 7, 0, 0] / [4, 13, 3, 0] / [0, 2, 16, 1] / [0, 0, 1, 19]
`noise` · `ens(digits+zoom_digits)` · matrix: [13, 8, 0, 0] / [4, 15, 1, 0] / [0, 1, 17, 1] / [0, 0, 3, 17]
`noise` · `independent` · bias+T: [12, 9, 0, 0] / [5, 11, 4, 0] / [0, 2, 14, 3] / [0, 0, 3, 17]
`noise` · `zoom_cumulative` · platt/threshold: [15, 6, 0, 0] / [6, 12, 2, 0] / [1, 0, 17, 1] / [0, 0, 2, 18]
`noise` · `zoom_digits` · bias+T: [12, 9, 0, 0] / [4, 14, 2, 0] / [0, 1, 18, 0] / [0, 0, 4, 16]

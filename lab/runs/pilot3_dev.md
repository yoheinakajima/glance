# Pilot 3: test-time augmentation over magnified crops (DEV split only)

DEV numbers: fit on the first half of the calibration split, reported on its second half. The test split is untouched.

Bar for "impressive": accuracy >= 85%, MAE <= 0.25 levels, ECE <= 0.05 (15 equal-mass bins; read each ECE against its sampling floor).

## Accuracy by method and scale (best calibration for each, chosen on the fit split)

| Method | blur | exposure | jpeg | noise | resolution | mean | passes | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `zoom_digits` | 0.863 | 0.863 | 0.713 | 0.800 | 0.750 | **0.797** | 1 | 164 |
| `zoom_digits x3 (mean)` | 0.850 | - | 0.738 | 0.800 | - | **0.796** | 3 | 1293 |
| `zoom_digits x5 (concat)` | 0.875 | - | 0.800 | 0.800 | - | **0.825** | 5 | 2409 |
| `zoom_digits x5 (mean)` | 0.875 | - | 0.750 | 0.800 | - | **0.808** | 5 | 2409 |

## Every method, scale and calibration

| Scale | Method | Calibration | chosen | n | Acc | Within 1 | MAE | Spearman | NLL | ECE | ECE floor | Meets bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | `zoom_digits` | raw |  | 80 | 0.287 | 0.938 | 0.769 | 0.961 | 6.648 | 0.671 | 0.031 |  |
| blur | `zoom_digits` | T |  | 80 | 0.287 | 0.938 | 0.846 | 0.956 | 1.281 | 0.361 | 0.176 |  |
| blur | `zoom_digits` | bias+T |  | 80 | 0.487 | 1.000 | 0.646 | 0.935 | 1.098 | 0.339 | 0.166 |  |
| blur | `zoom_digits` | matrix | yes | 80 | 0.863 | 1.000 | 0.202 | 0.961 | 0.381 | 0.086 | 0.081 |  |
| blur | `zoom_digits x3 (mean)` | raw |  | 80 | 0.287 | 0.963 | 0.749 | 0.960 | 6.273 | 0.660 | 0.036 |  |
| blur | `zoom_digits x3 (mean)` | T |  | 80 | 0.287 | 0.963 | 0.835 | 0.955 | 1.259 | 0.376 | 0.176 |  |
| blur | `zoom_digits x3 (mean)` | bias+T |  | 80 | 0.487 | 1.000 | 0.622 | 0.939 | 1.050 | 0.315 | 0.166 |  |
| blur | `zoom_digits x3 (mean)` | matrix | yes | 80 | 0.850 | 1.000 | 0.188 | 0.958 | 0.352 | 0.083 | 0.077 |  |
| blur | `zoom_digits x5 (concat)` | matrix | yes | 80 | 0.875 | 1.000 | 0.154 | 0.959 | 0.289 | 0.047 | 0.060 | YES |
| blur | `zoom_digits x5 (mean)` | raw |  | 80 | 0.287 | 0.963 | 0.751 | 0.961 | 6.195 | 0.660 | 0.036 |  |
| blur | `zoom_digits x5 (mean)` | T |  | 80 | 0.287 | 0.963 | 0.836 | 0.958 | 1.254 | 0.374 | 0.175 |  |
| blur | `zoom_digits x5 (mean)` | bias+T |  | 80 | 0.512 | 1.000 | 0.621 | 0.938 | 1.027 | 0.324 | 0.164 |  |
| blur | `zoom_digits x5 (mean)` | matrix | yes | 80 | 0.875 | 1.000 | 0.171 | 0.959 | 0.334 | 0.082 | 0.070 |  |
| exposure | `zoom_digits` | raw |  | 80 | 0.575 | 1.000 | 0.428 | 0.961 | 1.850 | 0.316 | 0.077 |  |
| exposure | `zoom_digits` | T |  | 80 | 0.575 | 1.000 | 0.488 | 0.960 | 0.914 | 0.229 | 0.159 |  |
| exposure | `zoom_digits` | bias+T | yes | 80 | 0.863 | 1.000 | 0.142 | 0.960 | 0.273 | 0.090 | 0.057 |  |
| exposure | `zoom_digits` | matrix |  | 80 | 0.938 | 1.000 | 0.122 | 0.964 | 0.208 | 0.076 | 0.054 |  |
| jpeg | `zoom_digits` | raw |  | 80 | 0.362 | 0.988 | 0.648 | 0.938 | 3.582 | 0.526 | 0.063 |  |
| jpeg | `zoom_digits` | T |  | 80 | 0.362 | 0.988 | 0.754 | 0.935 | 1.164 | 0.261 | 0.174 |  |
| jpeg | `zoom_digits` | bias+T |  | 80 | 0.787 | 0.975 | 0.360 | 0.928 | 0.653 | 0.154 | 0.155 |  |
| jpeg | `zoom_digits` | matrix | yes | 80 | 0.713 | 0.975 | 0.266 | 0.937 | 0.558 | 0.141 | 0.120 |  |
| jpeg | `zoom_digits x3 (mean)` | raw |  | 80 | 0.412 | 0.988 | 0.640 | 0.940 | 3.534 | 0.509 | 0.075 |  |
| jpeg | `zoom_digits x3 (mean)` | T |  | 80 | 0.412 | 0.988 | 0.748 | 0.927 | 1.181 | 0.223 | 0.175 |  |
| jpeg | `zoom_digits x3 (mean)` | bias+T |  | 80 | 0.675 | 0.988 | 0.392 | 0.915 | 0.823 | 0.105 | 0.131 |  |
| jpeg | `zoom_digits x3 (mean)` | matrix | yes | 80 | 0.738 | 0.975 | 0.301 | 0.925 | 0.666 | 0.174 | 0.102 |  |
| jpeg | `zoom_digits x5 (concat)` | matrix | yes | 80 | 0.800 | 0.975 | 0.257 | 0.937 | 0.669 | 0.124 | 0.067 |  |
| jpeg | `zoom_digits x5 (mean)` | raw |  | 80 | 0.400 | 0.988 | 0.631 | 0.946 | 3.449 | 0.506 | 0.074 |  |
| jpeg | `zoom_digits x5 (mean)` | T |  | 80 | 0.400 | 0.988 | 0.742 | 0.934 | 1.176 | 0.282 | 0.175 |  |
| jpeg | `zoom_digits x5 (mean)` | bias+T |  | 80 | 0.713 | 0.988 | 0.396 | 0.925 | 0.838 | 0.110 | 0.136 |  |
| jpeg | `zoom_digits x5 (mean)` | matrix | yes | 80 | 0.750 | 0.975 | 0.303 | 0.931 | 0.636 | 0.152 | 0.105 |  |
| noise | `zoom_digits` | raw |  | 80 | 0.400 | 1.000 | 0.559 | 0.915 | 2.381 | 0.506 | 0.068 |  |
| noise | `zoom_digits` | T |  | 80 | 0.400 | 1.000 | 0.547 | 0.920 | 0.974 | 0.238 | 0.172 |  |
| noise | `zoom_digits` | bias+T |  | 80 | 0.750 | 1.000 | 0.285 | 0.913 | 0.578 | 0.134 | 0.114 |  |
| noise | `zoom_digits` | matrix | yes | 80 | 0.800 | 0.988 | 0.251 | 0.922 | 0.472 | 0.089 | 0.101 |  |
| noise | `zoom_digits x3 (mean)` | raw |  | 80 | 0.388 | 1.000 | 0.563 | 0.914 | 2.359 | 0.509 | 0.066 |  |
| noise | `zoom_digits x3 (mean)` | T |  | 80 | 0.388 | 1.000 | 0.539 | 0.924 | 0.973 | 0.260 | 0.167 |  |
| noise | `zoom_digits x3 (mean)` | bias+T |  | 80 | 0.775 | 0.988 | 0.264 | 0.917 | 0.583 | 0.091 | 0.103 |  |
| noise | `zoom_digits x3 (mean)` | matrix | yes | 80 | 0.800 | 0.988 | 0.237 | 0.924 | 0.458 | 0.062 | 0.091 |  |
| noise | `zoom_digits x5 (concat)` | matrix | yes | 80 | 0.800 | 1.000 | 0.229 | 0.924 | 0.480 | 0.101 | 0.081 |  |
| noise | `zoom_digits x5 (mean)` | raw |  | 80 | 0.412 | 1.000 | 0.559 | 0.918 | 2.277 | 0.480 | 0.069 |  |
| noise | `zoom_digits x5 (mean)` | T |  | 80 | 0.412 | 1.000 | 0.535 | 0.926 | 0.960 | 0.284 | 0.169 |  |
| noise | `zoom_digits x5 (mean)` | bias+T |  | 80 | 0.775 | 1.000 | 0.255 | 0.920 | 0.591 | 0.130 | 0.088 |  |
| noise | `zoom_digits x5 (mean)` | matrix | yes | 80 | 0.800 | 1.000 | 0.237 | 0.926 | 0.463 | 0.083 | 0.086 |  |
| resolution | `zoom_digits` | raw |  | 80 | 0.588 | 0.975 | 0.488 | 0.801 | 3.163 | 0.245 | 0.113 |  |
| resolution | `zoom_digits` | T |  | 80 | 0.588 | 0.975 | 0.779 | 0.840 | 1.238 | 0.269 | 0.172 |  |
| resolution | `zoom_digits` | bias+T |  | 80 | 0.675 | 0.938 | 0.438 | 0.777 | 0.844 | 0.181 | 0.145 |  |
| resolution | `zoom_digits` | matrix | yes | 80 | 0.750 | 0.988 | 0.250 | 0.899 | 0.578 | 0.133 | 0.093 |  |

## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)

`blur` · `zoom_digits` · matrix: [19, 6, 0, 0] / [0, 18, 0, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`blur` · `zoom_digits x3 (mean)` · matrix: [20, 5, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 3, 18]
`blur` · `zoom_digits x5 (concat)` · matrix: [21, 4, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 2, 19]
`blur` · `zoom_digits x5 (mean)` · matrix: [21, 4, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 2, 19]
`exposure` · `zoom_digits` · bias+T: [19, 2, 0, 0] / [0, 21, 1, 0] / [0, 1, 13, 1] / [0, 0, 6, 16]
`jpeg` · `zoom_digits` · matrix: [18, 0, 1, 0] / [1, 5, 17, 1] / [0, 0, 17, 2] / [0, 0, 1, 17]
`jpeg` · `zoom_digits x3 (mean)` · matrix: [15, 2, 2, 0] / [2, 8, 14, 0] / [0, 0, 18, 1] / [0, 0, 0, 18]
`jpeg` · `zoom_digits x5 (concat)` · matrix: [17, 0, 2, 0] / [0, 10, 14, 0] / [0, 0, 19, 0] / [0, 0, 0, 18]
`jpeg` · `zoom_digits x5 (mean)` · matrix: [15, 2, 2, 0] / [2, 8, 14, 0] / [0, 0, 19, 0] / [0, 0, 0, 18]
`noise` · `zoom_digits` · matrix: [16, 5, 0, 0] / [4, 14, 2, 0] / [1, 0, 17, 1] / [0, 0, 3, 17]
`noise` · `zoom_digits x3 (mean)` · matrix: [15, 6, 0, 0] / [3, 14, 3, 0] / [1, 0, 17, 1] / [0, 0, 2, 18]
`noise` · `zoom_digits x5 (concat)` · matrix: [14, 7, 0, 0] / [3, 15, 2, 0] / [0, 1, 17, 1] / [0, 0, 2, 18]
`noise` · `zoom_digits x5 (mean)` · matrix: [15, 6, 0, 0] / [4, 14, 2, 0] / [0, 1, 17, 1] / [0, 0, 2, 18]
`resolution` · `zoom_digits` · matrix: [18, 2, 1, 0] / [13, 7, 3, 0] / [0, 0, 16, 1] / [0, 0, 0, 19]

# Score lab

DEV numbers: fit on the first half of the calibration split, reported on its second half. The test split is untouched.

Bar for "impressive": accuracy >= 85%, MAE <= 0.25 levels, ECE <= 0.05 (15 equal-mass bins; read each ECE against its sampling floor).

## Accuracy by method and scale (best calibration for each, chosen on the fit split)

| Method | blur | exposure | jpeg | noise | resolution | mean | passes | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `zoom_digits` | 0.863 | 0.863 | 0.713 | 0.750 | 0.750 | **0.787** | 1 | 164 |
| `zoom_digits x3 (mean)` | 0.850 | - | - | 0.780 | - | **0.815** | 3 | 1293 |
| `zoom_digits x5 (mean)` | 0.875 | - | - | 0.763 | - | **0.819** | 5 | 2409 |

## Every method, scale and calibration

| Scale | Method | Calibration | chosen | n | Acc | Within 1 | MAE | Spearman | NLL | ECE | ECE floor | Meets bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | `zoom_digits` | raw |  | 80 | 0.287 | 0.938 | 0.769 | 0.961 | 6.648 | 0.671 | 0.031 |  |
| blur | `zoom_digits` | T |  | 80 | 0.287 | 0.938 | 0.846 | 0.956 | 1.281 | 0.361 | 0.176 |  |
| blur | `zoom_digits` | bias+T |  | 80 | 0.487 | 1.000 | 0.646 | 0.935 | 1.098 | 0.339 | 0.166 |  |
| blur | `zoom_digits` | matrix | yes | 80 | 0.863 | 1.000 | 0.357 | 0.961 | 0.529 | 0.223 | 0.158 |  |
| blur | `zoom_digits x3 (mean)` | raw |  | 80 | 0.287 | 0.963 | 0.749 | 0.960 | 6.273 | 0.660 | 0.036 |  |
| blur | `zoom_digits x3 (mean)` | T |  | 80 | 0.287 | 0.963 | 0.835 | 0.955 | 1.259 | 0.376 | 0.176 |  |
| blur | `zoom_digits x3 (mean)` | bias+T |  | 80 | 0.487 | 1.000 | 0.622 | 0.939 | 1.050 | 0.315 | 0.166 |  |
| blur | `zoom_digits x3 (mean)` | matrix | yes | 80 | 0.850 | 1.000 | 0.358 | 0.959 | 0.519 | 0.231 | 0.157 |  |
| blur | `zoom_digits x5 (mean)` | raw |  | 80 | 0.287 | 0.963 | 0.751 | 0.961 | 6.195 | 0.660 | 0.036 |  |
| blur | `zoom_digits x5 (mean)` | T |  | 80 | 0.287 | 0.963 | 0.836 | 0.958 | 1.254 | 0.374 | 0.175 |  |
| blur | `zoom_digits x5 (mean)` | bias+T |  | 80 | 0.512 | 1.000 | 0.621 | 0.938 | 1.027 | 0.324 | 0.164 |  |
| blur | `zoom_digits x5 (mean)` | matrix | yes | 80 | 0.875 | 1.000 | 0.348 | 0.959 | 0.505 | 0.229 | 0.163 |  |
| exposure | `zoom_digits` | raw |  | 80 | 0.575 | 1.000 | 0.428 | 0.961 | 1.850 | 0.316 | 0.077 |  |
| exposure | `zoom_digits` | T |  | 80 | 0.575 | 1.000 | 0.488 | 0.960 | 0.914 | 0.229 | 0.159 |  |
| exposure | `zoom_digits` | bias+T | yes | 80 | 0.863 | 1.000 | 0.142 | 0.960 | 0.273 | 0.090 | 0.057 |  |
| exposure | `zoom_digits` | matrix |  | 80 | 0.938 | 1.000 | 0.333 | 0.964 | 0.531 | 0.325 | 0.159 |  |
| jpeg | `zoom_digits` | raw |  | 80 | 0.362 | 0.988 | 0.648 | 0.938 | 3.582 | 0.526 | 0.063 |  |
| jpeg | `zoom_digits` | T |  | 80 | 0.362 | 0.988 | 0.754 | 0.935 | 1.164 | 0.261 | 0.174 |  |
| jpeg | `zoom_digits` | bias+T |  | 80 | 0.787 | 0.975 | 0.360 | 0.928 | 0.653 | 0.154 | 0.155 |  |
| jpeg | `zoom_digits` | matrix | yes | 80 | 0.713 | 0.975 | 0.369 | 0.936 | 0.746 | 0.209 | 0.165 |  |
| noise | `zoom_digits` | raw |  | 80 | 0.400 | 1.000 | 0.559 | 0.915 | 2.381 | 0.506 | 0.068 |  |
| noise | `zoom_digits` | T |  | 80 | 0.400 | 1.000 | 0.547 | 0.920 | 0.974 | 0.238 | 0.172 |  |
| noise | `zoom_digits` | bias+T | yes | 80 | 0.750 | 1.000 | 0.285 | 0.913 | 0.578 | 0.134 | 0.114 |  |
| noise | `zoom_digits` | matrix |  | 80 | 0.800 | 0.988 | 0.351 | 0.924 | 0.630 | 0.208 | 0.165 |  |
| noise | `zoom_digits x3 (mean)` | raw |  | 59 | 0.373 | 1.000 | 0.572 | 0.917 | 2.270 | 0.515 | 0.084 |  |
| noise | `zoom_digits x3 (mean)` | T |  | 59 | 0.373 | 1.000 | 0.542 | 0.928 | 0.962 | 0.246 | 0.191 |  |
| noise | `zoom_digits x3 (mean)` | bias+T | yes | 59 | 0.780 | 0.983 | 0.256 | 0.918 | 0.566 | 0.127 | 0.117 |  |
| noise | `zoom_digits x3 (mean)` | matrix |  | 59 | 0.847 | 0.983 | 0.343 | 0.932 | 0.611 | 0.246 | 0.185 |  |
| noise | `zoom_digits x5 (mean)` | raw |  | 59 | 0.407 | 1.000 | 0.568 | 0.924 | 2.185 | 0.504 | 0.088 |  |
| noise | `zoom_digits x5 (mean)` | T |  | 59 | 0.407 | 1.000 | 0.540 | 0.933 | 0.949 | 0.325 | 0.196 |  |
| noise | `zoom_digits x5 (mean)` | bias+T | yes | 59 | 0.763 | 1.000 | 0.247 | 0.923 | 0.596 | 0.176 | 0.097 |  |
| noise | `zoom_digits x5 (mean)` | matrix |  | 59 | 0.797 | 1.000 | 0.341 | 0.931 | 0.603 | 0.195 | 0.189 |  |
| resolution | `zoom_digits` | raw |  | 80 | 0.588 | 0.975 | 0.488 | 0.801 | 3.163 | 0.245 | 0.113 |  |
| resolution | `zoom_digits` | T |  | 80 | 0.588 | 0.975 | 0.779 | 0.840 | 1.238 | 0.269 | 0.172 |  |
| resolution | `zoom_digits` | bias+T |  | 80 | 0.675 | 0.938 | 0.438 | 0.777 | 0.844 | 0.181 | 0.145 |  |
| resolution | `zoom_digits` | matrix | yes | 80 | 0.750 | 0.988 | 0.343 | 0.896 | 0.637 | 0.171 | 0.144 |  |

## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)

`blur` · `zoom_digits` · matrix: [19, 6, 0, 0] / [0, 18, 0, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`blur` · `zoom_digits x3 (mean)` · matrix: [20, 5, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 3, 18]
`blur` · `zoom_digits x5 (mean)` · matrix: [21, 4, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 2, 19]
`exposure` · `zoom_digits` · bias+T: [19, 2, 0, 0] / [0, 21, 1, 0] / [0, 1, 13, 1] / [0, 0, 6, 16]
`jpeg` · `zoom_digits` · matrix: [18, 0, 1, 0] / [1, 5, 17, 1] / [0, 0, 17, 2] / [0, 0, 1, 17]
`noise` · `zoom_digits` · bias+T: [12, 9, 0, 0] / [4, 14, 2, 0] / [0, 1, 18, 0] / [0, 0, 4, 16]
`noise` · `zoom_digits x3 (mean)` · bias+T: [9, 6, 0, 0] / [2, 10, 2, 0] / [1, 0, 13, 1] / [0, 0, 1, 14]
`noise` · `zoom_digits x5 (mean)` · bias+T: [10, 5, 0, 0] / [4, 8, 2, 0] / [0, 1, 13, 1] / [0, 0, 1, 14]
`resolution` · `zoom_digits` · matrix: [18, 2, 1, 0] / [13, 7, 3, 0] / [0, 0, 16, 1] / [0, 0, 0, 19]

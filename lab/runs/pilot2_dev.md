# Pilot 1+2 (DEV split only)

DEV numbers: fit on the first half of the calibration split, reported on its second half. The test split is untouched.

Bar for "impressive": accuracy >= 85%, MAE <= 0.25 levels, ECE <= 0.05 (15 equal-mass bins; read each ECE against its sampling floor).

## Accuracy by method and scale (best calibration for each, chosen on the fit split)

| Method | blur | exposure | jpeg | noise | resolution | mean | passes | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `anchors_digits@0:013` | 0.887 | 0.850 | 0.550 | 0.738 | 0.713 | **0.747** | 1 | 206 |
| `cumulative` | 0.825 | 0.800 | 0.525 | 0.725 | 0.575 | **0.690** | 3 | 336 |
| `digits` | 0.838 | 0.875 | 0.613 | 0.713 | 0.787 | **0.765** | 1 | 159 |
| `ens(5 readouts)` | 0.900 | 0.925 | 0.725 | 0.800 | 0.875 | **0.845** | 12 | 1463 |
| `ens(digits+zoom_digits)` | 0.838 | 0.938 | 0.700 | 0.775 | 0.838 | **0.817** | 2 | 324 |
| `independent` | 0.838 | 0.850 | 0.550 | 0.637 | 0.812 | **0.738** | 4 | 447 |
| `zoom_cumulative` | 0.700 | 0.787 | 0.613 | 0.812 | 0.675 | **0.717** | 3 | 355 |
| `zoom_digits` | 0.863 | 0.863 | 0.713 | 0.800 | 0.750 | **0.797** | 1 | 164 |
| `zoom_independent` | 0.775 | 0.838 | 0.650 | 0.775 | 0.650 | **0.738** | 4 | 660 |

## Every method, scale and calibration

| Scale | Method | Calibration | chosen | n | Acc | Within 1 | MAE | Spearman | NLL | ECE | ECE floor | Meets bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | `anchors_digits@0:013` | raw |  | 80 | 0.650 | 0.963 | 0.382 | 0.952 | 1.539 | 0.305 | 0.032 |  |
| blur | `anchors_digits@0:013` | T |  | 80 | 0.650 | 0.963 | 0.469 | 0.951 | 0.700 | 0.148 | 0.155 |  |
| blur | `anchors_digits@0:013` | bias+T |  | 80 | 0.887 | 1.000 | 0.196 | 0.939 | 0.511 | 0.102 | 0.089 |  |
| blur | `anchors_digits@0:013` | matrix | yes | 80 | 0.887 | 1.000 | 0.173 | 0.948 | 0.420 | 0.091 | 0.068 |  |
| blur | `cumulative` | raw |  | 80 | 0.575 | 0.950 | 0.440 | 0.941 | 8.609 | 0.352 | 0.060 |  |
| blur | `cumulative` | platt/threshold |  | 80 | 0.787 | 1.000 | 0.326 | 0.941 | 0.523 | 0.111 | 0.115 |  |
| blur | `cumulative` | matrix | yes | 80 | 0.825 | 1.000 | 0.222 | 0.960 | 0.355 | 0.120 | 0.091 |  |
| blur | `digits` | raw |  | 80 | 0.463 | 1.000 | 0.548 | 0.953 | 4.425 | 0.505 | 0.024 |  |
| blur | `digits` | T |  | 80 | 0.463 | 1.000 | 0.676 | 0.944 | 1.023 | 0.223 | 0.175 |  |
| blur | `digits` | bias+T |  | 80 | 0.600 | 1.000 | 0.473 | 0.941 | 0.928 | 0.224 | 0.140 |  |
| blur | `digits` | matrix | yes | 80 | 0.838 | 1.000 | 0.239 | 0.951 | 0.541 | 0.106 | 0.080 |  |
| blur | `ens(5 readouts)` | matrix | yes | 80 | 0.900 | 1.000 | 0.131 | 0.965 | 0.295 | 0.073 | 0.054 |  |
| blur | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.838 | 1.000 | 0.182 | 0.960 | 0.390 | 0.065 | 0.068 |  |
| blur | `independent` | raw |  | 80 | 0.662 | 1.000 | 0.362 | 0.958 | 1.323 | 0.265 | 0.055 |  |
| blur | `independent` | T |  | 80 | 0.662 | 1.000 | 0.480 | 0.962 | 0.756 | 0.157 | 0.154 |  |
| blur | `independent` | bias+T |  | 80 | 0.900 | 1.000 | 0.132 | 0.959 | 0.276 | 0.031 | 0.052 | YES |
| blur | `independent` | matrix | yes | 80 | 0.838 | 1.000 | 0.174 | 0.956 | 0.351 | 0.084 | 0.069 |  |
| blur | `zoom_cumulative` | raw |  | 80 | 0.550 | 0.938 | 0.497 | 0.886 | 8.082 | 0.259 | 0.129 |  |
| blur | `zoom_cumulative` | platt/threshold |  | 80 | 0.775 | 1.000 | 0.278 | 0.938 | 0.676 | 0.115 | 0.089 |  |
| blur | `zoom_cumulative` | matrix | yes | 80 | 0.700 | 1.000 | 0.268 | 0.953 | 0.510 | 0.160 | 0.095 |  |
| blur | `zoom_digits` | raw |  | 80 | 0.287 | 0.938 | 0.769 | 0.961 | 6.648 | 0.671 | 0.031 |  |
| blur | `zoom_digits` | T |  | 80 | 0.287 | 0.938 | 0.846 | 0.956 | 1.281 | 0.361 | 0.176 |  |
| blur | `zoom_digits` | bias+T |  | 80 | 0.487 | 1.000 | 0.646 | 0.935 | 1.098 | 0.339 | 0.166 |  |
| blur | `zoom_digits` | matrix | yes | 80 | 0.863 | 1.000 | 0.202 | 0.961 | 0.381 | 0.086 | 0.081 |  |
| blur | `zoom_independent` | raw |  | 80 | 0.200 | 0.988 | 0.795 | 0.951 | 6.157 | 0.741 | 0.044 |  |
| blur | `zoom_independent` | T |  | 80 | 0.200 | 0.988 | 0.848 | 0.956 | 1.320 | 0.294 | 0.182 |  |
| blur | `zoom_independent` | bias+T | yes | 80 | 0.775 | 1.000 | 0.270 | 0.938 | 0.535 | 0.140 | 0.103 |  |
| blur | `zoom_independent` | matrix |  | 80 | 0.812 | 1.000 | 0.237 | 0.942 | 0.393 | 0.096 | 0.102 |  |
| exposure | `anchors_digits@0:013` | raw |  | 80 | 0.537 | 0.975 | 0.484 | 0.930 | 1.782 | 0.358 | 0.067 |  |
| exposure | `anchors_digits@0:013` | T |  | 80 | 0.537 | 0.975 | 0.539 | 0.941 | 0.884 | 0.219 | 0.154 |  |
| exposure | `anchors_digits@0:013` | bias+T |  | 80 | 0.738 | 0.975 | 0.289 | 0.920 | 0.694 | 0.146 | 0.078 |  |
| exposure | `anchors_digits@0:013` | matrix | yes | 80 | 0.850 | 0.988 | 0.205 | 0.935 | 0.474 | 0.073 | 0.074 |  |
| exposure | `cumulative` | raw |  | 80 | 0.537 | 0.925 | 0.389 | 0.960 | 6.603 | 0.341 | 0.066 |  |
| exposure | `cumulative` | platt/threshold |  | 80 | 0.900 | 1.000 | 0.122 | 0.965 | 0.472 | 0.060 | 0.050 |  |
| exposure | `cumulative` | matrix | yes | 80 | 0.800 | 1.000 | 0.262 | 0.961 | 0.454 | 0.088 | 0.101 |  |
| exposure | `digits` | raw |  | 80 | 0.500 | 0.988 | 0.488 | 0.952 | 1.990 | 0.410 | 0.056 |  |
| exposure | `digits` | T |  | 80 | 0.500 | 0.988 | 0.517 | 0.952 | 0.841 | 0.270 | 0.155 |  |
| exposure | `digits` | bias+T |  | 80 | 0.887 | 0.975 | 0.220 | 0.943 | 0.426 | 0.074 | 0.065 |  |
| exposure | `digits` | matrix | yes | 80 | 0.875 | 1.000 | 0.161 | 0.950 | 0.305 | 0.046 | 0.062 | YES |
| exposure | `ens(5 readouts)` | matrix | yes | 80 | 0.925 | 1.000 | 0.098 | 0.960 | 0.243 | 0.061 | 0.030 |  |
| exposure | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.938 | 1.000 | 0.114 | 0.959 | 0.244 | 0.090 | 0.042 |  |
| exposure | `independent` | raw |  | 80 | 0.575 | 0.950 | 0.504 | 0.880 | 1.505 | 0.292 | 0.102 |  |
| exposure | `independent` | T |  | 80 | 0.575 | 0.950 | 0.668 | 0.926 | 0.965 | 0.179 | 0.170 |  |
| exposure | `independent` | bias+T |  | 80 | 0.688 | 0.963 | 0.343 | 0.906 | 0.565 | 0.128 | 0.122 |  |
| exposure | `independent` | matrix | yes | 80 | 0.850 | 1.000 | 0.174 | 0.959 | 0.314 | 0.095 | 0.065 |  |
| exposure | `zoom_cumulative` | raw |  | 80 | 0.537 | 0.975 | 0.269 | 0.957 | 6.933 | 0.289 | 0.122 |  |
| exposure | `zoom_cumulative` | platt/threshold |  | 80 | 0.887 | 1.000 | 0.109 | 0.970 | 0.472 | 0.067 | 0.034 |  |
| exposure | `zoom_cumulative` | matrix | yes | 80 | 0.787 | 1.000 | 0.247 | 0.963 | 0.420 | 0.109 | 0.099 |  |
| exposure | `zoom_digits` | raw |  | 80 | 0.575 | 1.000 | 0.428 | 0.961 | 1.850 | 0.316 | 0.077 |  |
| exposure | `zoom_digits` | T |  | 80 | 0.575 | 1.000 | 0.488 | 0.960 | 0.914 | 0.229 | 0.159 |  |
| exposure | `zoom_digits` | bias+T | yes | 80 | 0.863 | 1.000 | 0.142 | 0.960 | 0.273 | 0.090 | 0.057 |  |
| exposure | `zoom_digits` | matrix |  | 80 | 0.938 | 1.000 | 0.122 | 0.964 | 0.208 | 0.076 | 0.054 |  |
| exposure | `zoom_independent` | raw |  | 80 | 0.537 | 0.950 | 0.501 | 0.837 | 0.935 | 0.228 | 0.130 |  |
| exposure | `zoom_independent` | T |  | 80 | 0.537 | 0.950 | 0.526 | 0.867 | 0.842 | 0.196 | 0.149 |  |
| exposure | `zoom_independent` | bias+T |  | 80 | 0.688 | 1.000 | 0.334 | 0.920 | 0.531 | 0.130 | 0.122 |  |
| exposure | `zoom_independent` | matrix | yes | 80 | 0.838 | 1.000 | 0.182 | 0.964 | 0.309 | 0.065 | 0.069 |  |
| jpeg | `anchors_digits@0:013` | raw |  | 80 | 0.400 | 0.912 | 0.669 | 0.789 | 4.833 | 0.525 | 0.042 |  |
| jpeg | `anchors_digits@0:013` | T |  | 80 | 0.400 | 0.912 | 0.714 | 0.810 | 1.271 | 0.173 | 0.173 |  |
| jpeg | `anchors_digits@0:013` | bias+T |  | 80 | 0.412 | 0.863 | 0.682 | 0.808 | 1.233 | 0.177 | 0.171 |  |
| jpeg | `anchors_digits@0:013` | matrix | yes | 80 | 0.550 | 0.912 | 0.507 | 0.787 | 0.933 | 0.142 | 0.154 |  |
| jpeg | `cumulative` | raw |  | 80 | 0.463 | 0.850 | 0.753 | 0.791 | 2.876 | 0.458 | 0.063 |  |
| jpeg | `cumulative` | platt/threshold | yes | 80 | 0.525 | 0.938 | 0.492 | 0.796 | 2.316 | 0.265 | 0.148 |  |
| jpeg | `cumulative` | matrix |  | 80 | 0.537 | 0.925 | 0.489 | 0.813 | 0.923 | 0.135 | 0.156 |  |
| jpeg | `digits` | raw |  | 80 | 0.388 | 0.950 | 0.662 | 0.795 | 4.279 | 0.530 | 0.054 |  |
| jpeg | `digits` | T |  | 80 | 0.388 | 0.950 | 0.682 | 0.809 | 1.167 | 0.117 | 0.174 |  |
| jpeg | `digits` | bias+T |  | 80 | 0.550 | 0.950 | 0.579 | 0.817 | 1.064 | 0.141 | 0.175 |  |
| jpeg | `digits` | matrix | yes | 80 | 0.613 | 0.912 | 0.473 | 0.808 | 0.892 | 0.148 | 0.151 |  |
| jpeg | `ens(5 readouts)` | matrix | yes | 80 | 0.725 | 0.975 | 0.313 | 0.900 | 0.726 | 0.135 | 0.097 |  |
| jpeg | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.700 | 0.963 | 0.316 | 0.907 | 0.634 | 0.148 | 0.114 |  |
| jpeg | `independent` | raw |  | 80 | 0.375 | 0.900 | 0.701 | 0.822 | 2.418 | 0.462 | 0.095 |  |
| jpeg | `independent` | T |  | 80 | 0.375 | 0.900 | 0.686 | 0.829 | 1.168 | 0.192 | 0.170 |  |
| jpeg | `independent` | bias+T |  | 80 | 0.613 | 0.963 | 0.570 | 0.823 | 0.978 | 0.140 | 0.169 |  |
| jpeg | `independent` | matrix | yes | 80 | 0.550 | 0.938 | 0.495 | 0.822 | 0.943 | 0.187 | 0.154 |  |
| jpeg | `zoom_cumulative` | raw |  | 80 | 0.588 | 0.963 | 0.346 | 0.911 | 1.587 | 0.233 | 0.136 |  |
| jpeg | `zoom_cumulative` | platt/threshold |  | 80 | 0.713 | 0.988 | 0.330 | 0.921 | 0.688 | 0.094 | 0.134 |  |
| jpeg | `zoom_cumulative` | matrix | yes | 80 | 0.613 | 0.988 | 0.287 | 0.926 | 0.664 | 0.158 | 0.130 |  |
| jpeg | `zoom_digits` | raw |  | 80 | 0.362 | 0.988 | 0.648 | 0.938 | 3.582 | 0.526 | 0.063 |  |
| jpeg | `zoom_digits` | T |  | 80 | 0.362 | 0.988 | 0.754 | 0.935 | 1.164 | 0.261 | 0.174 |  |
| jpeg | `zoom_digits` | bias+T |  | 80 | 0.787 | 0.975 | 0.360 | 0.928 | 0.653 | 0.154 | 0.155 |  |
| jpeg | `zoom_digits` | matrix | yes | 80 | 0.713 | 0.975 | 0.266 | 0.937 | 0.558 | 0.141 | 0.120 |  |
| jpeg | `zoom_independent` | raw |  | 80 | 0.450 | 1.000 | 0.588 | 0.942 | 2.157 | 0.467 | 0.096 |  |
| jpeg | `zoom_independent` | T |  | 80 | 0.450 | 1.000 | 0.743 | 0.925 | 1.166 | 0.266 | 0.180 |  |
| jpeg | `zoom_independent` | bias+T |  | 80 | 0.625 | 0.938 | 0.432 | 0.894 | 0.842 | 0.204 | 0.164 |  |
| jpeg | `zoom_independent` | matrix | yes | 80 | 0.650 | 0.975 | 0.328 | 0.920 | 0.644 | 0.150 | 0.138 |  |
| noise | `anchors_digits@0:013` | raw |  | 80 | 0.575 | 0.988 | 0.447 | 0.909 | 3.271 | 0.394 | 0.042 |  |
| noise | `anchors_digits@0:013` | T |  | 80 | 0.575 | 0.988 | 0.513 | 0.907 | 1.009 | 0.213 | 0.167 |  |
| noise | `anchors_digits@0:013` | bias+T |  | 80 | 0.562 | 0.950 | 0.450 | 0.891 | 0.820 | 0.134 | 0.155 |  |
| noise | `anchors_digits@0:013` | matrix | yes | 80 | 0.738 | 0.975 | 0.331 | 0.892 | 0.725 | 0.155 | 0.095 |  |
| noise | `cumulative` | raw |  | 80 | 0.588 | 0.988 | 0.387 | 0.914 | 2.953 | 0.333 | 0.044 |  |
| noise | `cumulative` | platt/threshold |  | 80 | 0.750 | 1.000 | 0.285 | 0.913 | 0.677 | 0.121 | 0.097 |  |
| noise | `cumulative` | matrix | yes | 80 | 0.725 | 1.000 | 0.333 | 0.914 | 0.629 | 0.119 | 0.123 |  |
| noise | `digits` | raw |  | 80 | 0.675 | 1.000 | 0.370 | 0.899 | 1.432 | 0.306 | 0.068 |  |
| noise | `digits` | T |  | 80 | 0.675 | 1.000 | 0.397 | 0.901 | 0.780 | 0.178 | 0.136 |  |
| noise | `digits` | bias+T |  | 80 | 0.738 | 1.000 | 0.317 | 0.900 | 0.735 | 0.206 | 0.114 |  |
| noise | `digits` | matrix | yes | 80 | 0.713 | 1.000 | 0.288 | 0.903 | 0.609 | 0.150 | 0.098 |  |
| noise | `ens(5 readouts)` | matrix | yes | 80 | 0.800 | 1.000 | 0.230 | 0.925 | 0.499 | 0.133 | 0.080 |  |
| noise | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.775 | 1.000 | 0.259 | 0.912 | 0.526 | 0.113 | 0.095 |  |
| noise | `independent` | raw |  | 80 | 0.500 | 1.000 | 0.487 | 0.891 | 1.697 | 0.415 | 0.067 |  |
| noise | `independent` | T |  | 80 | 0.500 | 1.000 | 0.470 | 0.904 | 0.897 | 0.220 | 0.155 |  |
| noise | `independent` | bias+T |  | 80 | 0.675 | 1.000 | 0.360 | 0.886 | 0.868 | 0.165 | 0.108 |  |
| noise | `independent` | matrix | yes | 80 | 0.637 | 1.000 | 0.327 | 0.887 | 0.651 | 0.197 | 0.107 |  |
| noise | `zoom_cumulative` | raw |  | 80 | 0.525 | 0.950 | 0.357 | 0.921 | 7.181 | 0.371 | 0.061 |  |
| noise | `zoom_cumulative` | platt/threshold |  | 80 | 0.775 | 0.988 | 0.236 | 0.911 | 1.127 | 0.159 | 0.055 |  |
| noise | `zoom_cumulative` | matrix | yes | 80 | 0.812 | 1.000 | 0.220 | 0.936 | 0.507 | 0.101 | 0.085 |  |
| noise | `zoom_digits` | raw |  | 80 | 0.400 | 1.000 | 0.559 | 0.915 | 2.381 | 0.506 | 0.068 |  |
| noise | `zoom_digits` | T |  | 80 | 0.400 | 1.000 | 0.547 | 0.920 | 0.974 | 0.238 | 0.172 |  |
| noise | `zoom_digits` | bias+T |  | 80 | 0.750 | 1.000 | 0.285 | 0.913 | 0.578 | 0.134 | 0.114 |  |
| noise | `zoom_digits` | matrix | yes | 80 | 0.800 | 0.988 | 0.251 | 0.922 | 0.472 | 0.089 | 0.101 |  |
| noise | `zoom_independent` | raw |  | 80 | 0.487 | 1.000 | 0.511 | 0.923 | 2.028 | 0.433 | 0.066 |  |
| noise | `zoom_independent` | T |  | 80 | 0.487 | 1.000 | 0.549 | 0.942 | 1.014 | 0.276 | 0.160 |  |
| noise | `zoom_independent` | bias+T |  | 80 | 0.625 | 1.000 | 0.398 | 0.896 | 0.786 | 0.158 | 0.136 |  |
| noise | `zoom_independent` | matrix | yes | 80 | 0.775 | 1.000 | 0.270 | 0.919 | 0.604 | 0.119 | 0.106 |  |
| resolution | `anchors_digits@0:013` | raw |  | 80 | 0.512 | 0.875 | 0.634 | 0.838 | 2.331 | 0.374 | 0.069 |  |
| resolution | `anchors_digits@0:013` | T |  | 80 | 0.512 | 0.875 | 0.606 | 0.867 | 1.071 | 0.207 | 0.159 |  |
| resolution | `anchors_digits@0:013` | bias+T |  | 80 | 0.700 | 0.963 | 0.390 | 0.837 | 0.863 | 0.182 | 0.138 |  |
| resolution | `anchors_digits@0:013` | matrix | yes | 80 | 0.713 | 0.975 | 0.304 | 0.875 | 0.647 | 0.128 | 0.113 |  |
| resolution | `cumulative` | raw |  | 80 | 0.475 | 0.887 | 0.500 | 0.834 | 8.625 | 0.387 | 0.105 |  |
| resolution | `cumulative` | platt/threshold |  | 80 | 0.725 | 0.975 | 0.378 | 0.890 | 0.690 | 0.154 | 0.150 |  |
| resolution | `cumulative` | matrix | yes | 80 | 0.575 | 0.975 | 0.480 | 0.912 | 0.869 | 0.170 | 0.144 |  |
| resolution | `digits` | raw |  | 80 | 0.537 | 0.975 | 0.511 | 0.911 | 2.282 | 0.375 | 0.077 |  |
| resolution | `digits` | T |  | 80 | 0.537 | 0.975 | 0.544 | 0.917 | 1.015 | 0.248 | 0.163 |  |
| resolution | `digits` | bias+T |  | 80 | 0.800 | 1.000 | 0.264 | 0.906 | 0.830 | 0.137 | 0.103 |  |
| resolution | `digits` | matrix | yes | 80 | 0.787 | 1.000 | 0.252 | 0.921 | 0.611 | 0.152 | 0.094 |  |
| resolution | `ens(5 readouts)` | matrix | yes | 80 | 0.875 | 1.000 | 0.192 | 0.925 | 0.441 | 0.072 | 0.077 |  |
| resolution | `ens(digits+zoom_digits)` | matrix | yes | 80 | 0.838 | 1.000 | 0.193 | 0.925 | 0.468 | 0.092 | 0.072 |  |
| resolution | `independent` | raw |  | 80 | 0.525 | 0.975 | 0.516 | 0.909 | 3.447 | 0.369 | 0.073 |  |
| resolution | `independent` | T |  | 80 | 0.525 | 0.975 | 0.805 | 0.926 | 1.260 | 0.192 | 0.172 |  |
| resolution | `independent` | bias+T | yes | 80 | 0.812 | 0.988 | 0.287 | 0.902 | 0.816 | 0.144 | 0.113 |  |
| resolution | `independent` | matrix |  | 80 | 0.713 | 0.963 | 0.404 | 0.906 | 0.799 | 0.105 | 0.139 |  |
| resolution | `zoom_cumulative` | raw |  | 80 | 0.425 | 0.738 | 0.773 | 0.824 | 10.527 | 0.403 | 0.133 |  |
| resolution | `zoom_cumulative` | platt/threshold |  | 80 | 0.725 | 0.963 | 0.375 | 0.832 | 0.712 | 0.140 | 0.129 |  |
| resolution | `zoom_cumulative` | matrix | yes | 80 | 0.675 | 0.963 | 0.329 | 0.881 | 0.746 | 0.146 | 0.123 |  |
| resolution | `zoom_digits` | raw |  | 80 | 0.588 | 0.975 | 0.488 | 0.801 | 3.163 | 0.245 | 0.113 |  |
| resolution | `zoom_digits` | T |  | 80 | 0.588 | 0.975 | 0.779 | 0.840 | 1.238 | 0.269 | 0.172 |  |
| resolution | `zoom_digits` | bias+T |  | 80 | 0.675 | 0.938 | 0.438 | 0.777 | 0.844 | 0.181 | 0.145 |  |
| resolution | `zoom_digits` | matrix | yes | 80 | 0.750 | 0.988 | 0.250 | 0.899 | 0.578 | 0.133 | 0.093 |  |
| resolution | `zoom_independent` | raw |  | 80 | 0.500 | 0.975 | 0.534 | 0.887 | 4.905 | 0.406 | 0.067 |  |
| resolution | `zoom_independent` | T |  | 80 | 0.500 | 0.975 | 0.925 | 0.902 | 1.337 | 0.285 | 0.165 |  |
| resolution | `zoom_independent` | bias+T | yes | 80 | 0.650 | 0.925 | 0.378 | 0.838 | 0.829 | 0.140 | 0.135 |  |
| resolution | `zoom_independent` | matrix |  | 80 | 0.625 | 0.963 | 0.372 | 0.888 | 0.991 | 0.158 | 0.124 |  |

## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)

`blur` · `anchors_digits@0:013` · matrix: [21, 4, 0, 0] / [1, 17, 0, 0] / [0, 1, 12, 3] / [0, 0, 0, 21]
`blur` · `cumulative` · matrix: [23, 2, 0, 0] / [0, 18, 0, 0] / [0, 1, 15, 0] / [0, 0, 11, 10]
`blur` · `digits` · matrix: [20, 5, 0, 0] / [0, 18, 0, 0] / [0, 1, 12, 3] / [0, 0, 4, 17]
`blur` · `ens(5 readouts)` · matrix: [22, 3, 0, 0] / [0, 18, 0, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`blur` · `ens(digits+zoom_digits)` · matrix: [19, 6, 0, 0] / [0, 18, 0, 0] / [0, 2, 12, 2] / [0, 0, 3, 18]
`blur` · `independent` · matrix: [21, 4, 0, 0] / [0, 18, 0, 0] / [0, 1, 13, 2] / [0, 0, 6, 15]
`blur` · `zoom_cumulative` · matrix: [23, 2, 0, 0] / [0, 15, 3, 0] / [0, 1, 15, 0] / [0, 0, 18, 3]
`blur` · `zoom_digits` · matrix: [19, 6, 0, 0] / [0, 18, 0, 0] / [0, 2, 13, 1] / [0, 0, 2, 19]
`blur` · `zoom_independent` · bias+T: [16, 9, 0, 0] / [0, 18, 0, 0] / [0, 1, 15, 0] / [0, 0, 8, 13]
`exposure` · `anchors_digits@0:013` · matrix: [20, 1, 0, 0] / [2, 17, 2, 1] / [0, 0, 12, 3] / [0, 0, 3, 19]
`exposure` · `cumulative` · matrix: [21, 0, 0, 0] / [1, 9, 12, 0] / [0, 0, 14, 1] / [0, 0, 2, 20]
`exposure` · `digits` · matrix: [20, 1, 0, 0] / [1, 18, 3, 0] / [0, 1, 12, 2] / [0, 0, 2, 20]
`exposure` · `ens(5 readouts)` · matrix: [20, 1, 0, 0] / [0, 19, 3, 0] / [0, 0, 14, 1] / [0, 0, 1, 21]
`exposure` · `ens(digits+zoom_digits)` · matrix: [20, 1, 0, 0] / [0, 20, 2, 0] / [0, 0, 14, 1] / [0, 0, 1, 21]
`exposure` · `independent` · matrix: [19, 2, 0, 0] / [1, 16, 5, 0] / [0, 0, 14, 1] / [0, 0, 3, 19]
`exposure` · `zoom_cumulative` · matrix: [21, 0, 0, 0] / [2, 7, 13, 0] / [0, 0, 15, 0] / [0, 0, 2, 20]
`exposure` · `zoom_digits` · bias+T: [19, 2, 0, 0] / [0, 21, 1, 0] / [0, 1, 13, 1] / [0, 0, 6, 16]
`exposure` · `zoom_independent` · matrix: [19, 2, 0, 0] / [0, 18, 4, 0] / [0, 0, 15, 0] / [0, 0, 7, 15]
`jpeg` · `anchors_digits@0:013` · matrix: [15, 0, 4, 0] / [12, 1, 11, 0] / [3, 0, 12, 4] / [0, 0, 2, 16]
`jpeg` · `cumulative` · platt/threshold: [5, 10, 4, 0] / [5, 10, 9, 0] / [1, 2, 15, 1] / [0, 0, 6, 12]
`jpeg` · `digits` · matrix: [15, 0, 4, 0] / [7, 3, 14, 0] / [3, 0, 14, 2] / [0, 0, 1, 17]
`jpeg` · `ens(5 readouts)` · matrix: [17, 0, 2, 0] / [1, 10, 13, 0] / [0, 3, 15, 1] / [0, 0, 2, 16]
`jpeg` · `ens(digits+zoom_digits)` · matrix: [16, 0, 3, 0] / [1, 9, 14, 0] / [0, 3, 15, 1] / [0, 0, 2, 16]
`jpeg` · `independent` · matrix: [14, 1, 4, 0] / [6, 3, 15, 0] / [1, 2, 11, 5] / [0, 0, 2, 16]
`jpeg` · `zoom_cumulative` · matrix: [18, 0, 1, 0] / [4, 0, 20, 0] / [0, 0, 15, 4] / [0, 0, 2, 16]
`jpeg` · `zoom_digits` · matrix: [18, 0, 1, 0] / [1, 5, 17, 1] / [0, 0, 17, 2] / [0, 0, 1, 17]
`jpeg` · `zoom_independent` · matrix: [18, 0, 1, 0] / [6, 5, 13, 0] / [1, 0, 12, 6] / [0, 0, 1, 17]
`noise` · `anchors_digits@0:013` · matrix: [18, 2, 1, 0] / [8, 9, 3, 0] / [1, 0, 16, 2] / [0, 0, 4, 16]
`noise` · `cumulative` · matrix: [18, 3, 0, 0] / [9, 8, 3, 0] / [0, 1, 14, 4] / [0, 0, 2, 18]
`noise` · `digits` · matrix: [13, 8, 0, 0] / [8, 11, 1, 0] / [0, 2, 15, 2] / [0, 0, 2, 18]
`noise` · `ens(5 readouts)` · matrix: [14, 7, 0, 0] / [4, 15, 1, 0] / [0, 1, 17, 1] / [0, 0, 2, 18]
`noise` · `ens(digits+zoom_digits)` · matrix: [13, 8, 0, 0] / [4, 15, 1, 0] / [0, 1, 17, 1] / [0, 0, 3, 17]
`noise` · `independent` · matrix: [13, 8, 0, 0] / [11, 5, 4, 0] / [0, 3, 15, 1] / [0, 0, 2, 18]
`noise` · `zoom_cumulative` · matrix: [18, 3, 0, 0] / [5, 12, 3, 0] / [0, 1, 15, 3] / [0, 0, 0, 20]
`noise` · `zoom_digits` · matrix: [16, 5, 0, 0] / [4, 14, 2, 0] / [1, 0, 17, 1] / [0, 0, 3, 17]
`noise` · `zoom_independent` · matrix: [16, 5, 0, 0] / [5, 11, 4, 0] / [0, 4, 15, 0] / [0, 0, 0, 20]
`resolution` · `anchors_digits@0:013` · matrix: [15, 5, 1, 0] / [12, 8, 3, 0] / [1, 0, 15, 1] / [0, 0, 0, 19]
`resolution` · `cumulative` · matrix: [17, 2, 2, 0] / [9, 9, 5, 0] / [0, 0, 14, 3] / [0, 0, 13, 6]
`resolution` · `digits` · matrix: [16, 5, 0, 0] / [5, 14, 4, 0] / [0, 0, 14, 3] / [0, 0, 0, 19]
`resolution` · `ens(5 readouts)` · matrix: [17, 4, 0, 0] / [5, 17, 1, 0] / [0, 0, 17, 0] / [0, 0, 0, 19]
`resolution` · `ens(digits+zoom_digits)` · matrix: [16, 5, 0, 0] / [5, 15, 3, 0] / [0, 0, 17, 0] / [0, 0, 0, 19]
`resolution` · `independent` · bias+T: [17, 3, 1, 0] / [6, 17, 0, 0] / [0, 2, 14, 1] / [0, 0, 2, 17]
`resolution` · `zoom_cumulative` · matrix: [19, 0, 2, 0] / [15, 6, 2, 0] / [1, 2, 10, 4] / [0, 0, 0, 19]
`resolution` · `zoom_digits` · matrix: [18, 2, 1, 0] / [13, 7, 3, 0] / [0, 0, 16, 1] / [0, 0, 0, 19]
`resolution` · `zoom_independent` · bias+T: [15, 1, 5, 0] / [8, 10, 5, 0] / [1, 5, 8, 3] / [0, 0, 0, 19]

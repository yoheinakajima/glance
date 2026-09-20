# Full run, stage A, DEV (calibration split only: fit 250, judged 250)

DEV numbers: fit on the first half of the calibration split, reported on its second half. The test split is untouched.

Bar for "impressive": accuracy >= 85%, MAE <= 0.25 levels, ECE <= 0.05 (15 equal-mass bins; read each ECE against its sampling floor).

## Accuracy by method and scale (best calibration for each, chosen on the fit split)

| Method | blur | exposure | jpeg | noise | resolution | mean | passes | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cumulative` | 0.772 | 0.884 | 0.600 | 0.812 | 0.756 | **0.765** | 3 | 331 |
| `digits` | 0.776 | 0.916 | 0.672 | 0.788 | 0.824 | **0.795** | 1 | 156 |
| `ens2(digits+zoom_digits)` | 0.856 | 0.944 | 0.772 | 0.792 | 0.884 | **0.850** | 2 | 312 |
| `ens3(independent+digits+zoom_digits)` | 0.856 | 0.944 | 0.760 | 0.812 | 0.852 | **0.845** | 6 | 748 |
| `ens5(all five)` | 0.884 | 0.944 | 0.752 | 0.844 | 0.876 | **0.860** | 12 | 1701 |
| `independent` | 0.856 | 0.944 | 0.604 | 0.776 | 0.832 | **0.802** | 4 | 436 |
| `zoom_cumulative` | 0.740 | 0.900 | 0.704 | 0.808 | 0.760 | **0.782** | 3 | 623 |
| `zoom_digits` | 0.836 | 0.952 | 0.780 | 0.808 | 0.824 | **0.840** | 1 | 156 |

## Every method, scale and calibration

| Scale | Method | Calibration | chosen | n | Acc | Within 1 | MAE | Spearman | NLL | ECE | ECE floor | Meets bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | `cumulative` | raw |  | 250 | 0.516 | 0.912 | 0.501 | 0.935 | 9.562 | 0.411 | 0.040 |  |
| blur | `cumulative` | platt/threshold |  | 250 | 0.752 | 1.000 | 0.348 | 0.908 | 0.776 | 0.080 | 0.073 |  |
| blur | `cumulative` | matrix | yes | 250 | 0.772 | 0.996 | 0.223 | 0.955 | 0.458 | 0.110 | 0.061 |  |
| blur | `digits` | raw |  | 250 | 0.444 | 0.976 | 0.587 | 0.955 | 4.909 | 0.522 | 0.015 |  |
| blur | `digits` | T |  | 250 | 0.444 | 0.976 | 0.644 | 0.946 | 1.070 | 0.253 | 0.094 |  |
| blur | `digits` | bias+T |  | 250 | 0.612 | 1.000 | 0.439 | 0.936 | 0.884 | 0.198 | 0.081 |  |
| blur | `digits` | matrix | yes | 250 | 0.776 | 1.000 | 0.266 | 0.957 | 0.465 | 0.084 | 0.057 |  |
| blur | `ens2(digits+zoom_digits)` | matrix | yes | 250 | 0.856 | 1.000 | 0.193 | 0.958 | 0.350 | 0.053 | 0.046 |  |
| blur | `ens3(independent+digits+zoom_digits)` | matrix | yes | 250 | 0.856 | 1.000 | 0.175 | 0.957 | 0.316 | 0.053 | 0.043 |  |
| blur | `ens5(all five)` | matrix | yes | 250 | 0.884 | 1.000 | 0.155 | 0.960 | 0.294 | 0.037 | 0.038 | YES |
| blur | `independent` | raw |  | 250 | 0.572 | 0.976 | 0.449 | 0.948 | 1.461 | 0.323 | 0.041 |  |
| blur | `independent` | T |  | 250 | 0.572 | 0.976 | 0.496 | 0.952 | 0.819 | 0.160 | 0.087 |  |
| blur | `independent` | bias+T | yes | 250 | 0.856 | 1.000 | 0.183 | 0.949 | 0.392 | 0.059 | 0.040 |  |
| blur | `independent` | matrix |  | 250 | 0.844 | 1.000 | 0.210 | 0.943 | 0.405 | 0.063 | 0.049 |  |
| blur | `zoom_cumulative` | raw |  | 250 | 0.500 | 0.924 | 0.530 | 0.874 | 9.022 | 0.300 | 0.071 |  |
| blur | `zoom_cumulative` | platt/threshold |  | 250 | 0.804 | 0.996 | 0.280 | 0.933 | 0.497 | 0.057 | 0.069 |  |
| blur | `zoom_cumulative` | matrix | yes | 250 | 0.740 | 0.996 | 0.268 | 0.944 | 0.503 | 0.125 | 0.062 |  |
| blur | `zoom_digits` | raw |  | 250 | 0.292 | 0.948 | 0.770 | 0.958 | 6.493 | 0.655 | 0.022 |  |
| blur | `zoom_digits` | T |  | 250 | 0.292 | 0.948 | 0.787 | 0.948 | 1.265 | 0.344 | 0.097 |  |
| blur | `zoom_digits` | bias+T |  | 250 | 0.476 | 1.000 | 0.558 | 0.926 | 1.062 | 0.175 | 0.095 |  |
| blur | `zoom_digits` | matrix | yes | 250 | 0.836 | 1.000 | 0.210 | 0.956 | 0.361 | 0.070 | 0.052 |  |
| exposure | `cumulative` | raw |  | 250 | 0.524 | 0.960 | 0.382 | 0.962 | 6.848 | 0.364 | 0.042 |  |
| exposure | `cumulative` | platt/threshold | yes | 250 | 0.884 | 1.000 | 0.162 | 0.955 | 0.303 | 0.050 | 0.046 | YES |
| exposure | `cumulative` | matrix |  | 250 | 0.896 | 1.000 | 0.168 | 0.963 | 0.254 | 0.062 | 0.052 |  |
| exposure | `digits` | raw |  | 250 | 0.536 | 0.996 | 0.461 | 0.961 | 1.920 | 0.375 | 0.037 |  |
| exposure | `digits` | T |  | 250 | 0.536 | 0.996 | 0.500 | 0.962 | 0.826 | 0.178 | 0.088 |  |
| exposure | `digits` | bias+T |  | 250 | 0.876 | 1.000 | 0.261 | 0.953 | 0.401 | 0.069 | 0.059 |  |
| exposure | `digits` | matrix | yes | 250 | 0.916 | 1.000 | 0.142 | 0.960 | 0.223 | 0.062 | 0.048 |  |
| exposure | `ens2(digits+zoom_digits)` | matrix | yes | 250 | 0.944 | 1.000 | 0.113 | 0.965 | 0.172 | 0.045 | 0.042 | YES |
| exposure | `ens3(independent+digits+zoom_digits)` | matrix | yes | 250 | 0.944 | 1.000 | 0.104 | 0.965 | 0.154 | 0.030 | 0.037 | YES |
| exposure | `ens5(all five)` | matrix | yes | 250 | 0.944 | 1.000 | 0.101 | 0.965 | 0.148 | 0.030 | 0.037 | YES |
| exposure | `independent` | raw |  | 250 | 0.560 | 0.980 | 0.482 | 0.875 | 1.514 | 0.283 | 0.061 |  |
| exposure | `independent` | T |  | 250 | 0.560 | 0.980 | 0.643 | 0.938 | 0.977 | 0.125 | 0.092 |  |
| exposure | `independent` | bias+T |  | 250 | 0.684 | 0.992 | 0.313 | 0.911 | 0.571 | 0.098 | 0.067 |  |
| exposure | `independent` | matrix | yes | 250 | 0.944 | 1.000 | 0.122 | 0.961 | 0.197 | 0.052 | 0.039 |  |
| exposure | `zoom_cumulative` | raw |  | 250 | 0.524 | 0.968 | 0.253 | 0.957 | 7.068 | 0.268 | 0.070 |  |
| exposure | `zoom_cumulative` | platt/threshold | yes | 250 | 0.900 | 1.000 | 0.124 | 0.961 | 0.244 | 0.040 | 0.035 | YES |
| exposure | `zoom_cumulative` | matrix |  | 250 | 0.896 | 1.000 | 0.162 | 0.963 | 0.243 | 0.049 | 0.048 | YES |
| exposure | `zoom_digits` | raw |  | 250 | 0.656 | 1.000 | 0.387 | 0.963 | 1.620 | 0.260 | 0.047 |  |
| exposure | `zoom_digits` | T |  | 250 | 0.656 | 1.000 | 0.455 | 0.964 | 0.856 | 0.216 | 0.089 |  |
| exposure | `zoom_digits` | bias+T |  | 250 | 0.908 | 1.000 | 0.135 | 0.962 | 0.226 | 0.045 | 0.043 | YES |
| exposure | `zoom_digits` | matrix | yes | 250 | 0.952 | 1.000 | 0.108 | 0.965 | 0.170 | 0.043 | 0.039 | YES |
| jpeg | `cumulative` | raw |  | 250 | 0.460 | 0.804 | 0.763 | 0.779 | 2.905 | 0.431 | 0.038 |  |
| jpeg | `cumulative` | platt/threshold | yes | 250 | 0.600 | 0.948 | 0.490 | 0.790 | 0.851 | 0.107 | 0.087 |  |
| jpeg | `cumulative` | matrix |  | 250 | 0.580 | 0.928 | 0.499 | 0.805 | 0.846 | 0.112 | 0.089 |  |
| jpeg | `digits` | raw |  | 250 | 0.364 | 0.944 | 0.693 | 0.811 | 4.854 | 0.576 | 0.025 |  |
| jpeg | `digits` | T |  | 250 | 0.364 | 0.944 | 0.716 | 0.813 | 1.197 | 0.147 | 0.096 |  |
| jpeg | `digits` | bias+T |  | 250 | 0.548 | 0.952 | 0.562 | 0.813 | 1.088 | 0.119 | 0.098 |  |
| jpeg | `digits` | matrix | yes | 250 | 0.672 | 0.944 | 0.437 | 0.820 | 0.774 | 0.103 | 0.082 |  |
| jpeg | `ens2(digits+zoom_digits)` | matrix | yes | 250 | 0.772 | 0.980 | 0.307 | 0.894 | 0.571 | 0.062 | 0.068 |  |
| jpeg | `ens3(independent+digits+zoom_digits)` | matrix | yes | 250 | 0.760 | 0.976 | 0.320 | 0.887 | 0.606 | 0.063 | 0.067 |  |
| jpeg | `ens5(all five)` | matrix | yes | 250 | 0.752 | 0.976 | 0.311 | 0.891 | 0.583 | 0.071 | 0.067 |  |
| jpeg | `independent` | raw |  | 250 | 0.332 | 0.888 | 0.759 | 0.821 | 2.847 | 0.536 | 0.049 |  |
| jpeg | `independent` | T |  | 250 | 0.332 | 0.888 | 0.730 | 0.822 | 1.206 | 0.160 | 0.095 |  |
| jpeg | `independent` | bias+T |  | 250 | 0.544 | 0.968 | 0.548 | 0.809 | 1.033 | 0.073 | 0.096 |  |
| jpeg | `independent` | matrix | yes | 250 | 0.604 | 0.960 | 0.458 | 0.821 | 0.817 | 0.093 | 0.088 |  |
| jpeg | `zoom_cumulative` | raw |  | 250 | 0.592 | 0.944 | 0.400 | 0.896 | 1.551 | 0.158 | 0.079 |  |
| jpeg | `zoom_cumulative` | platt/threshold |  | 250 | 0.692 | 0.980 | 0.376 | 0.901 | 0.650 | 0.075 | 0.078 |  |
| jpeg | `zoom_cumulative` | matrix | yes | 250 | 0.704 | 0.992 | 0.330 | 0.912 | 0.616 | 0.082 | 0.075 |  |
| jpeg | `zoom_digits` | raw |  | 250 | 0.344 | 0.980 | 0.679 | 0.924 | 3.957 | 0.556 | 0.039 |  |
| jpeg | `zoom_digits` | T |  | 250 | 0.344 | 0.980 | 0.765 | 0.922 | 1.202 | 0.262 | 0.099 |  |
| jpeg | `zoom_digits` | bias+T |  | 250 | 0.724 | 1.000 | 0.397 | 0.916 | 0.716 | 0.105 | 0.088 |  |
| jpeg | `zoom_digits` | matrix | yes | 250 | 0.780 | 1.000 | 0.292 | 0.923 | 0.541 | 0.102 | 0.071 |  |
| noise | `cumulative` | raw |  | 250 | 0.572 | 0.984 | 0.396 | 0.930 | 2.839 | 0.353 | 0.027 |  |
| noise | `cumulative` | platt/threshold | yes | 250 | 0.812 | 1.000 | 0.254 | 0.929 | 0.476 | 0.064 | 0.057 |  |
| noise | `cumulative` | matrix |  | 250 | 0.748 | 1.000 | 0.313 | 0.931 | 0.558 | 0.081 | 0.073 |  |
| noise | `digits` | raw |  | 250 | 0.656 | 1.000 | 0.352 | 0.918 | 1.223 | 0.245 | 0.042 |  |
| noise | `digits` | T |  | 250 | 0.656 | 1.000 | 0.380 | 0.919 | 0.700 | 0.135 | 0.077 |  |
| noise | `digits` | bias+T |  | 250 | 0.760 | 1.000 | 0.311 | 0.918 | 0.633 | 0.116 | 0.069 |  |
| noise | `digits` | matrix | yes | 250 | 0.788 | 1.000 | 0.266 | 0.921 | 0.521 | 0.078 | 0.061 |  |
| noise | `ens2(digits+zoom_digits)` | matrix | yes | 250 | 0.792 | 0.996 | 0.230 | 0.930 | 0.454 | 0.082 | 0.050 |  |
| noise | `ens3(independent+digits+zoom_digits)` | matrix | yes | 250 | 0.812 | 0.996 | 0.226 | 0.929 | 0.446 | 0.077 | 0.052 |  |
| noise | `ens5(all five)` | matrix | yes | 250 | 0.844 | 0.996 | 0.203 | 0.939 | 0.400 | 0.043 | 0.049 |  |
| noise | `independent` | raw |  | 250 | 0.528 | 1.000 | 0.465 | 0.914 | 1.631 | 0.387 | 0.039 |  |
| noise | `independent` | T |  | 250 | 0.528 | 1.000 | 0.456 | 0.924 | 0.852 | 0.182 | 0.089 |  |
| noise | `independent` | bias+T |  | 250 | 0.708 | 1.000 | 0.331 | 0.912 | 0.651 | 0.107 | 0.073 |  |
| noise | `independent` | matrix | yes | 250 | 0.776 | 0.996 | 0.272 | 0.919 | 0.504 | 0.083 | 0.064 |  |
| noise | `zoom_cumulative` | raw |  | 250 | 0.528 | 0.948 | 0.354 | 0.934 | 6.107 | 0.364 | 0.038 |  |
| noise | `zoom_cumulative` | platt/threshold | yes | 250 | 0.808 | 0.996 | 0.209 | 0.933 | 0.460 | 0.095 | 0.041 |  |
| noise | `zoom_cumulative` | matrix |  | 250 | 0.812 | 0.996 | 0.230 | 0.945 | 0.469 | 0.079 | 0.054 |  |
| noise | `zoom_digits` | raw |  | 250 | 0.440 | 1.000 | 0.547 | 0.926 | 2.281 | 0.463 | 0.037 |  |
| noise | `zoom_digits` | T |  | 250 | 0.440 | 1.000 | 0.545 | 0.931 | 0.948 | 0.193 | 0.097 |  |
| noise | `zoom_digits` | bias+T |  | 250 | 0.800 | 0.996 | 0.242 | 0.924 | 0.557 | 0.088 | 0.052 |  |
| noise | `zoom_digits` | matrix | yes | 250 | 0.808 | 0.996 | 0.220 | 0.930 | 0.458 | 0.077 | 0.046 |  |
| resolution | `cumulative` | raw |  | 250 | 0.492 | 0.876 | 0.491 | 0.810 | 9.351 | 0.357 | 0.063 |  |
| resolution | `cumulative` | platt/threshold | yes | 250 | 0.756 | 0.992 | 0.374 | 0.899 | 0.641 | 0.090 | 0.083 |  |
| resolution | `cumulative` | matrix |  | 250 | 0.780 | 0.988 | 0.423 | 0.923 | 0.712 | 0.189 | 0.085 |  |
| resolution | `digits` | raw |  | 250 | 0.524 | 0.980 | 0.491 | 0.943 | 2.028 | 0.354 | 0.045 |  |
| resolution | `digits` | T |  | 250 | 0.524 | 0.980 | 0.525 | 0.946 | 0.937 | 0.198 | 0.089 |  |
| resolution | `digits` | bias+T |  | 250 | 0.824 | 1.000 | 0.261 | 0.940 | 0.594 | 0.088 | 0.069 |  |
| resolution | `digits` | matrix | yes | 250 | 0.824 | 1.000 | 0.220 | 0.948 | 0.443 | 0.053 | 0.060 |  |
| resolution | `ens2(digits+zoom_digits)` | matrix | yes | 250 | 0.884 | 1.000 | 0.163 | 0.951 | 0.317 | 0.054 | 0.045 |  |
| resolution | `ens3(independent+digits+zoom_digits)` | matrix | yes | 250 | 0.852 | 1.000 | 0.167 | 0.948 | 0.329 | 0.051 | 0.047 |  |
| resolution | `ens5(all five)` | matrix | yes | 250 | 0.876 | 1.000 | 0.158 | 0.949 | 0.317 | 0.031 | 0.043 | YES |
| resolution | `independent` | raw |  | 250 | 0.500 | 0.984 | 0.528 | 0.925 | 3.161 | 0.389 | 0.046 |  |
| resolution | `independent` | T |  | 250 | 0.500 | 0.984 | 0.781 | 0.939 | 1.230 | 0.216 | 0.097 |  |
| resolution | `independent` | bias+T | yes | 250 | 0.832 | 0.992 | 0.277 | 0.934 | 0.615 | 0.096 | 0.073 |  |
| resolution | `independent` | matrix |  | 250 | 0.788 | 0.988 | 0.353 | 0.930 | 0.644 | 0.069 | 0.080 |  |
| resolution | `zoom_cumulative` | raw |  | 250 | 0.448 | 0.712 | 0.785 | 0.838 | 10.367 | 0.312 | 0.075 |  |
| resolution | `zoom_cumulative` | platt/threshold |  | 250 | 0.716 | 0.972 | 0.384 | 0.857 | 0.663 | 0.060 | 0.079 |  |
| resolution | `zoom_cumulative` | matrix | yes | 250 | 0.760 | 0.980 | 0.320 | 0.901 | 0.611 | 0.099 | 0.075 |  |
| resolution | `zoom_digits` | raw |  | 250 | 0.596 | 0.960 | 0.499 | 0.844 | 3.137 | 0.227 | 0.062 |  |
| resolution | `zoom_digits` | T |  | 250 | 0.596 | 0.960 | 0.761 | 0.873 | 1.234 | 0.221 | 0.095 |  |
| resolution | `zoom_digits` | bias+T |  | 250 | 0.668 | 0.928 | 0.442 | 0.800 | 0.830 | 0.094 | 0.083 |  |
| resolution | `zoom_digits` | matrix | yes | 250 | 0.824 | 0.992 | 0.222 | 0.931 | 0.401 | 0.076 | 0.055 |  |

## Confusion matrices for the chosen calibration (rows = true level, columns = predicted)

`blur` · `cumulative` · matrix: [66, 1, 0, 0] / [2, 52, 0, 1] / [0, 10, 14, 42] / [0, 0, 1, 61]
`blur` · `digits` · matrix: [55, 12, 0, 0] / [1, 50, 4, 0] / [0, 7, 27, 32] / [0, 0, 0, 62]
`blur` · `ens2(digits+zoom_digits)` · matrix: [55, 12, 0, 0] / [1, 50, 4, 0] / [0, 7, 48, 11] / [0, 0, 1, 61]
`blur` · `ens3(independent+digits+zoom_digits)` · matrix: [58, 9, 0, 0] / [2, 49, 4, 0] / [0, 7, 49, 10] / [0, 0, 4, 58]
`blur` · `ens5(all five)` · matrix: [62, 5, 0, 0] / [2, 49, 4, 0] / [0, 7, 51, 8] / [0, 0, 3, 59]
`blur` · `independent` · bias+T: [61, 6, 0, 0] / [2, 48, 5, 0] / [0, 5, 51, 10] / [0, 0, 8, 54]
`blur` · `zoom_cumulative` · matrix: [64, 3, 0, 0] / [3, 49, 2, 1] / [0, 6, 10, 50] / [0, 0, 0, 62]
`blur` · `zoom_digits` · matrix: [55, 12, 0, 0] / [2, 50, 3, 0] / [0, 9, 43, 14] / [0, 0, 1, 61]
`exposure` · `cumulative` · platt/threshold: [66, 1, 0, 0] / [4, 52, 9, 0] / [0, 8, 44, 2] / [0, 0, 5, 59]
`exposure` · `digits` · matrix: [62, 5, 0, 0] / [1, 61, 3, 0] / [0, 3, 46, 5] / [0, 0, 4, 60]
`exposure` · `ens2(digits+zoom_digits)` · matrix: [62, 5, 0, 0] / [0, 63, 2, 0] / [0, 3, 50, 1] / [0, 0, 3, 61]
`exposure` · `ens3(independent+digits+zoom_digits)` · matrix: [62, 5, 0, 0] / [1, 62, 2, 0] / [0, 2, 51, 1] / [0, 0, 3, 61]
`exposure` · `ens5(all five)` · matrix: [62, 5, 0, 0] / [1, 62, 2, 0] / [0, 2, 51, 1] / [0, 0, 3, 61]
`exposure` · `independent` · matrix: [65, 2, 0, 0] / [2, 59, 4, 0] / [0, 2, 50, 2] / [0, 0, 2, 62]
`exposure` · `zoom_cumulative` · platt/threshold: [64, 3, 0, 0] / [6, 55, 4, 0] / [0, 3, 50, 1] / [0, 0, 8, 56]
`exposure` · `zoom_digits` · matrix: [63, 4, 0, 0] / [1, 62, 2, 0] / [0, 1, 51, 2] / [0, 0, 2, 62]
`jpeg` · `cumulative` · platt/threshold: [28, 27, 5, 0] / [22, 30, 13, 0] / [8, 13, 36, 6] / [0, 0, 6, 56]
`jpeg` · `digits` · matrix: [42, 13, 5, 0] / [23, 28, 14, 0] / [9, 9, 38, 7] / [0, 0, 2, 60]
`jpeg` · `ens2(digits+zoom_digits)` · matrix: [48, 8, 4, 0] / [5, 46, 14, 0] / [1, 15, 39, 8] / [0, 0, 2, 60]
`jpeg` · `ens3(independent+digits+zoom_digits)` · matrix: [48, 7, 5, 0] / [5, 45, 15, 0] / [1, 18, 37, 7] / [0, 0, 2, 60]
`jpeg` · `ens5(all five)` · matrix: [48, 7, 5, 0] / [7, 44, 14, 0] / [1, 18, 37, 7] / [0, 0, 3, 59]
`jpeg` · `independent` · matrix: [34, 21, 5, 0] / [19, 30, 16, 0] / [5, 17, 30, 11] / [0, 0, 5, 57]
`jpeg` · `zoom_cumulative` · matrix: [50, 9, 1, 0] / [9, 28, 28, 0] / [1, 10, 39, 13] / [0, 0, 3, 59]
`jpeg` · `zoom_digits` · matrix: [50, 10, 0, 0] / [9, 43, 13, 0] / [0, 12, 42, 9] / [0, 0, 2, 60]
`noise` · `cumulative` · platt/threshold: [51, 13, 0, 0] / [16, 38, 5, 0] / [0, 4, 53, 6] / [0, 0, 3, 61]
`noise` · `digits` · matrix: [48, 16, 0, 0] / [16, 39, 4, 0] / [0, 5, 48, 10] / [0, 0, 2, 62]
`noise` · `ens2(digits+zoom_digits)` · matrix: [48, 16, 0, 0] / [16, 38, 5, 0] / [1, 5, 52, 5] / [0, 0, 4, 60]
`noise` · `ens3(independent+digits+zoom_digits)` · matrix: [48, 16, 0, 0] / [17, 37, 5, 0] / [1, 5, 56, 1] / [0, 0, 2, 62]
`noise` · `ens5(all five)` · matrix: [54, 10, 0, 0] / [14, 39, 6, 0] / [1, 5, 55, 2] / [0, 0, 1, 63]
`noise` · `independent` · matrix: [49, 15, 0, 0] / [25, 26, 8, 0] / [1, 4, 57, 1] / [0, 0, 2, 62]
`noise` · `zoom_cumulative` · platt/threshold: [57, 7, 0, 0] / [22, 31, 6, 0] / [1, 4, 55, 3] / [0, 0, 5, 59]
`noise` · `zoom_digits` · matrix: [51, 13, 0, 0] / [19, 36, 4, 0] / [1, 3, 55, 4] / [0, 0, 4, 60]
`resolution` · `cumulative` · platt/threshold: [50, 12, 2, 0] / [9, 41, 13, 0] / [0, 5, 40, 15] / [0, 0, 5, 58]
`resolution` · `digits` · matrix: [46, 18, 0, 0] / [8, 43, 12, 0] / [0, 0, 54, 6] / [0, 0, 0, 63]
`resolution` · `ens2(digits+zoom_digits)` · matrix: [48, 16, 0, 0] / [8, 50, 5, 0] / [0, 0, 60, 0] / [0, 0, 0, 63]
`resolution` · `ens3(independent+digits+zoom_digits)` · matrix: [47, 17, 0, 0] / [8, 45, 10, 0] / [0, 2, 58, 0] / [0, 0, 0, 63]
`resolution` · `ens5(all five)` · matrix: [50, 14, 0, 0] / [7, 47, 9, 0] / [0, 1, 59, 0] / [0, 0, 0, 63]
`resolution` · `independent` · bias+T: [47, 15, 2, 0] / [9, 47, 7, 0] / [0, 2, 55, 3] / [0, 0, 4, 59]
`resolution` · `zoom_cumulative` · matrix: [52, 9, 3, 0] / [15, 37, 10, 1] / [1, 9, 39, 11] / [0, 0, 1, 62]
`resolution` · `zoom_digits` · matrix: [50, 13, 1, 0] / [13, 37, 13, 0] / [1, 2, 56, 1] / [0, 0, 0, 63]

# Score lab follow-ups for `zoom_digits` with `bias+T` calibration

DEV (calibration split only; test split untouched)

## Learning curve: labeled examples used for calibration -> accuracy (mean ± sd over random draws)

| Scale | n=0 | n=8 | n=16 | n=32 | n=64 |
| --- | --- | --- | --- | --- | --- |
| blur | 0.287 ± 0.000 | 0.487 ± 0.053 | 0.456 ± 0.026 | 0.455 ± 0.022 | - |
| noise | 0.400 ± 0.000 | 0.723 ± 0.052 | 0.750 ± 0.025 | 0.762 ± 0.016 | 0.754 ± 0.009 |
| jpeg | 0.362 ± 0.000 | 0.725 ± 0.070 | 0.778 ± 0.041 | 0.795 ± 0.018 | 0.811 ± 0.008 |
| exposure | 0.575 ± 0.000 | 0.762 ± 0.060 | 0.877 ± 0.057 | 0.886 ± 0.047 | 0.889 ± 0.026 |
| resolution | 0.588 ± 0.000 | 0.628 ± 0.038 | 0.646 ± 0.041 | 0.642 ± 0.040 | 0.664 ± 0.020 |

n=0 is the raw readout. Mean absolute error in levels:

| Scale | n=0 | n=8 | n=16 | n=32 | n=64 |
| --- | --- | --- | --- | --- | --- |
| blur | 0.769 | 0.609 | 0.623 | 0.619 | - |
| noise | 0.559 | 0.300 | 0.292 | 0.279 | 0.285 |
| jpeg | 0.648 | 0.372 | 0.347 | 0.354 | 0.347 |
| exposure | 0.428 | 0.240 | 0.131 | 0.142 | 0.133 |
| resolution | 0.488 | 0.448 | 0.441 | 0.445 | 0.435 |

## Transfer: accuracy when the calibration is fit on one scale (rows) and used on another (columns)

| Fit on | blur | noise | jpeg | exposure | resolution |
| --- | --- | --- | --- | --- | --- |
| blur | 0.487 | 0.375 | 0.487 | 0.475 | 0.325 |
| noise | 0.300 | 0.750 | 0.350 | 0.588 | 0.287 |
| jpeg | 0.350 | 0.613 | 0.787 | 0.762 | 0.463 |
| exposure | 0.375 | 0.475 | 0.637 | 0.863 | 0.688 |
| resolution | 0.438 | 0.388 | 0.625 | 0.863 | 0.675 |
| POOLED | 0.362 | 0.512 | 0.650 | 0.900 | 0.662 |
| NONE (raw) | 0.287 | 0.400 | 0.362 | 0.575 | 0.588 |

## Ensemble: matrix scaling on the concatenated logits of `independent`, `cumulative`, `digits`, `zoom_cumulative`, `zoom_digits`

| Scale | n | Acc | MAE | ECE | ECE floor |
| --- | --- | --- | --- | --- | --- |
| blur | 80 | 0.900 | 0.240 | 0.175 | 0.129 |
| exposure | 80 | 0.925 | 0.231 | 0.183 | 0.136 |
| jpeg | 80 | 0.725 | 0.359 | 0.153 | 0.155 |
| noise | 80 | 0.800 | 0.289 | 0.137 | 0.144 |
| resolution | 80 | 0.875 | 0.258 | 0.158 | 0.133 |

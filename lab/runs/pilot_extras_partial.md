# Score lab follow-ups for `independent` with `bias+T` calibration

DEV (calibration split only; test split untouched)

## Learning curve: labeled examples used for calibration -> accuracy (mean ± sd over random draws)

| Scale | n=0 | n=8 | n=16 | n=32 |
| --- | --- | --- | --- | --- |
| blur | 0.662 ± 0.000 | 0.751 ± 0.088 | 0.838 ± 0.042 | 0.871 ± 0.035 |
| noise | 0.522 ± 0.000 | 0.613 ± 0.058 | 0.633 ± 0.064 | - |

n=0 is the raw readout. Mean absolute error in levels:

| Scale | n=0 | n=8 | n=16 | n=32 |
| --- | --- | --- | --- | --- |
| blur | 0.362 | 0.257 | 0.169 | 0.147 |
| noise | 0.462 | 0.381 | 0.377 | - |

## Transfer: accuracy when the calibration is fit on one scale (rows) and used on another (columns)

| Fit on | blur | noise |
| --- | --- | --- |
| blur | 0.900 | 0.652 |
| noise | 0.537 | 0.565 |
| POOLED | 0.812 | 0.652 |
| NONE (raw) | 0.662 | 0.522 |

## Ensemble: matrix scaling on the concatenated logits of `independent`, `cumulative`, `digits`

| Scale | n | Acc | MAE | ECE | ECE floor |
| --- | --- | --- | --- | --- | --- |
| blur | 80 | 0.863 | 0.274 | 0.218 | 0.138 |
| noise | 23 | 0.739 | 0.317 | 0.242 | 0.244 |

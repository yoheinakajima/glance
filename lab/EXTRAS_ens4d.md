# Score lab follow-ups for `ens4d` with `matrix` calibration

fit on the calibration split, reported on the test split

## Learning curve: labeled examples used for calibration -> accuracy (mean ± sd over random draws)

| Scale | n=8 | n=16 | n=32 | n=64 | n=128 | n=248 | n=500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur | 0.849 ± 0.050 | 0.873 ± 0.036 | 0.882 ± 0.016 | 0.891 ± 0.010 | 0.889 ± 0.009 | 0.885 ± 0.007 | 0.888 ± 0.000 |
| exposure | 0.894 ± 0.021 | 0.901 ± 0.017 | 0.911 ± 0.008 | 0.920 ± 0.006 | 0.923 ± 0.004 | 0.923 ± 0.004 | 0.924 ± 0.000 |
| jpeg | 0.674 ± 0.032 | 0.724 ± 0.029 | 0.752 ± 0.020 | 0.764 ± 0.017 | 0.774 ± 0.011 | 0.777 ± 0.008 | 0.772 ± 0.000 |
| noise | 0.762 ± 0.080 | 0.832 ± 0.026 | 0.850 ± 0.014 | 0.859 ± 0.006 | 0.859 ± 0.007 | 0.861 ± 0.006 | 0.864 ± 0.000 |
| resolution | 0.809 ± 0.058 | 0.860 ± 0.018 | 0.885 ± 0.015 | 0.888 ± 0.014 | 0.891 ± 0.010 | 0.891 ± 0.006 | 0.888 ± 0.000 |

n=0 (single readouts only) is the raw, uncalibrated readout. Mean absolute error in levels:

| Scale | n=8 | n=16 | n=32 | n=64 | n=128 | n=248 | n=500 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur | 0.154 | 0.147 | 0.156 | 0.147 | 0.153 | 0.157 | 0.160 |
| exposure | 0.107 | 0.107 | 0.106 | 0.108 | 0.109 | 0.113 | 0.114 |
| jpeg | 0.356 | 0.297 | 0.279 | 0.280 | 0.276 | 0.276 | 0.276 |
| noise | 0.242 | 0.187 | 0.191 | 0.196 | 0.197 | 0.200 | 0.203 |
| resolution | 0.192 | 0.146 | 0.136 | 0.140 | 0.142 | 0.141 | 0.141 |

## Transfer: accuracy when the calibration is fit on one scale (rows) and used on another (columns)

| Fit on | blur | exposure | jpeg | noise | resolution |
| --- | --- | --- | --- | --- | --- |
| blur | 0.888 | 0.668 | 0.262 | 0.282 | 0.450 |
| exposure | 0.636 | 0.924 | 0.268 | 0.380 | 0.762 |
| jpeg | 0.434 | 0.498 | 0.772 | 0.536 | 0.336 |
| noise | 0.312 | 0.446 | 0.516 | 0.864 | 0.434 |
| resolution | 0.600 | 0.798 | 0.260 | 0.488 | 0.888 |
| POOLED | 0.608 | 0.826 | 0.430 | 0.640 | 0.782 |
| NONE (raw) | 0.176 | 0.048 | 0.238 | 0.288 | 0.230 |

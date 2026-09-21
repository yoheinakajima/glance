# `ladders`: fixed methods, per-distortion calibration, test split

No selection happened on this benchmark: methods and calibrations were fixed beforehand (`lab/NOTES.md`, entry 14).

## Summary (mean over distortions)

| Method | scales | n test / scale | accuracy | accuracy, most confident 80% | within 1 | MAE | mean ECE (mean floor) | scales with ECE <= 0.05 | pooled ECE (floor) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ens4d` | 5 | 200 | 0.854 | 0.905 | 0.996 | 0.188 | 0.060 (0.053) | 1 | 0.023 (0.023) |
| `fast2` | 5 | 200 | 0.800 | 0.849 | 0.990 | 0.261 | 0.078 (0.068) | 0 | 0.031 (0.030) |
| `zoom_digits` | 5 | 200 | 0.728 | 0.772 | 0.952 | 0.362 | 0.078 (0.075) | 1 | 0.046 (0.033) |
| `digits` | 5 | 200 | 0.775 | 0.811 | 0.991 | 0.305 | 0.076 (0.075) | 1 | 0.033 (0.034) |
| `independent` | 5 | 200 | 0.762 | 0.796 | 0.973 | 0.363 | 0.090 (0.082) | 0 | 0.059 (0.036) |
| `independent (as shipped)` | 5 | 200 | 0.384 | 0.432 | 0.851 | 0.746 | 0.297 (0.104) | 0 | 0.205 (0.047) |

## Per distortion (`ens4d`; accuracy of the baselines alongside)

| Distortion | n fit | n test | accuracy | within 1 | MAE | ECE (floor) | acc `independent` | acc `independent (as shipped)` | acc `digits` | acc `zoom_digits` | acc `fast2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur | 200 | 200 | 0.880 | 1.000 | 0.156 | 0.047 (0.046) | 0.820 | 0.515 | 0.800 | 0.765 | 0.840 |
| exposure | 200 | 200 | 0.865 | 1.000 | 0.176 | 0.057 (0.047) | 0.845 | 0.265 | 0.815 | 0.815 | 0.855 |
| jpeg | 200 | 200 | 0.795 | 0.980 | 0.246 | 0.077 (0.060) | 0.620 | 0.250 | 0.710 | 0.565 | 0.730 |
| noise | 200 | 200 | 0.900 | 1.000 | 0.159 | 0.062 (0.051) | 0.725 | 0.640 | 0.850 | 0.815 | 0.870 |
| resolution | 200 | 200 | 0.830 | 1.000 | 0.204 | 0.059 (0.060) | 0.800 | 0.250 | 0.700 | 0.680 | 0.705 |

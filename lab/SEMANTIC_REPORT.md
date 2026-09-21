# `semantic`: fixed methods, per-distortion calibration, test split

No selection happened on this benchmark: methods and calibrations were fixed beforehand (`lab/NOTES.md`, entry 14).

## Summary (mean over distortions)

| Method | scales | n test / scale | accuracy | accuracy, most confident 80% | within 1 | MAE | mean ECE (mean floor) | scales with ECE <= 0.05 | pooled ECE (floor) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ens4d` | 5 | 300 | 0.548 | 0.588 | 0.897 | 0.583 | 0.087 (0.075) | 1 | 0.062 (0.034) |
| `fast2` | 5 | 300 | 0.533 | 0.578 | 0.861 | 0.654 | 0.076 (0.077) | 0 | 0.037 (0.035) |
| `zoom_digits` | 5 | 300 | 0.507 | 0.550 | 0.852 | 0.666 | 0.084 (0.078) | 0 | 0.041 (0.035) |
| `digits` | 5 | 300 | 0.507 | 0.535 | 0.843 | 0.706 | 0.088 (0.077) | 0 | 0.040 (0.034) |
| `independent` | 5 | 300 | 0.486 | 0.522 | 0.835 | 0.695 | 0.095 (0.078) | 0 | 0.044 (0.036) |
| `independent (as shipped)` | 5 | 300 | 0.292 | 0.304 | 0.729 | 0.985 | 0.513 (0.055) | 0 | 0.513 (0.024) |

## Per distortion (`ens4d`; accuracy of the baselines alongside)

| Distortion | n fit | n test | accuracy | within 1 | MAE | ECE (floor) | acc `independent` | acc `independent (as shipped)` | acc `digits` | acc `zoom_digits` | acc `fast2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cutoff | 300 | 300 | 0.477 | 0.887 | 0.642 | 0.102 (0.082) | 0.353 | 0.210 | 0.297 | 0.440 | 0.373 |
| occlusion | 300 | 300 | 0.727 | 0.983 | 0.332 | 0.071 (0.061) | 0.677 | 0.367 | 0.713 | 0.653 | 0.727 |
| text_legibility | 300 | 300 | 0.650 | 0.903 | 0.465 | 0.045 (0.066) | 0.567 | 0.373 | 0.627 | 0.590 | 0.670 |
| tilt | 300 | 300 | 0.330 | 0.750 | 0.971 | 0.101 (0.082) | 0.317 | 0.183 | 0.330 | 0.313 | 0.337 |
| watermark | 300 | 300 | 0.557 | 0.960 | 0.504 | 0.119 (0.082) | 0.517 | 0.327 | 0.567 | 0.537 | 0.557 |

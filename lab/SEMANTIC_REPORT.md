# `semantic`: fixed methods, per-distortion calibration, test split

No selection happened on this benchmark: methods and calibrations were fixed beforehand (`lab/NOTES.md`, entry 14).

## Summary (mean over distortions)

| Method | scales | n test / scale | accuracy | accuracy, most confident 80% | within 1 | MAE | mean ECE (mean floor) | scales with ECE <= 0.05 | pooled ECE (floor) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ens4d` | 5 | 300 | 0.548 | 0.588 | 0.897 | 0.583 | 0.087 (0.075) | 1 | 0.062 (0.034) |
| `fast2` | 5 | 300 | 0.533 | 0.578 | 0.861 | 0.654 | 0.076 (0.077) | 0 | 0.037 (0.035) |
| `zoom_digits` | 5 | 300 | 0.507 | 0.550 | 0.852 | 0.666 | 0.084 (0.078) | 0 | 0.041 (0.035) |
| `digits` | 5 | 300 | 0.507 | 0.535 | 0.843 | 0.706 | 0.088 (0.077) | 0 | 0.040 (0.034) |
| `independent` | 1 | 67 | 0.896 | 0.907 | 1.000 | 0.185 | 0.103 (0.122) | 0 | 0.103 (0.122) |
| `independent (as shipped)` | 1 | 67 | 0.045 | 0.056 | 0.493 | 1.437 | 0.656 (0.156) | 0 | 0.656 (0.156) |

## Per distortion (`ens4d`; accuracy of the baselines alongside)

| Distortion | n fit | n test | accuracy | within 1 | MAE | ECE (floor) | acc `independent` | acc `independent (as shipped)` | acc `digits` | acc `zoom_digits` | acc `fast2` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cutoff | 300 | 300 | 0.477 | 0.887 | 0.642 | 0.102 (0.082) | 0.896 | 0.045 | 0.297 | 0.440 | 0.373 |
| occlusion | 300 | 300 | 0.727 | 0.983 | 0.332 | 0.071 (0.061) | - | - | 0.713 | 0.653 | 0.727 |
| text_legibility | 300 | 300 | 0.650 | 0.903 | 0.465 | 0.045 (0.066) | - | - | 0.627 | 0.590 | 0.670 |
| tilt | 300 | 300 | 0.330 | 0.750 | 0.971 | 0.101 (0.082) | - | - | 0.330 | 0.313 | 0.337 |
| watermark | 300 | 300 | 0.557 | 0.960 | 0.504 | 0.119 (0.082) | - | - | 0.567 | 0.537 | 0.557 |

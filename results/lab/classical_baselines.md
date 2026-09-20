# Classical no-reference baselines vs the score lab

Reviewer-requested honest baseline: how well do cheap, task-specific image-quality features do on the same items and labels the VLM is scored on. Features and the fit procedure are in `tools/classical_baselines.py`. CPU only, seed 7. The test split is used only to report; the calibration split is the only thing anything is fit on.

Feature extraction time: p50 12.73 ms, p90 15.78 ms per image (n=100, single process, feature computation only, not counting file read or image decode). Machine: this development machine, one run.

## Optional extras that were tried

- **brisque**: skipped. piq's own code is Apache-2.0 and pip-installs cleanly (only extra dep is torchvision, already required by this project). But piq.brisque() downloads a pretrained SVR weight file (brisque_svm_weights.pt, ~112 KB, from piq's GitHub releases) whose support vectors were fit on the LIVE lab's BRISQUE reference release (https://live.ece.utexas.edu/research/Quality/index_algorithms.htm, GitHub org utlive/BRISQUE). That upstream repo carries no LICENSE file (GitHub's detected license is null) and the LIVE page only asks for a citation, so the terms for reusing a model trained on their data are unclear. Per the task's own rule (check the license of any weights a package downloads), this was skipped.
- **niqe**: skipped. piq (the one package that passed the license check for its own code) does not implement NIQE. The only other permissively-licensed candidate found in the time available, `image-quality` (ocampor/image-quality, Apache-2.0), also only implements BRISQUE, not NIQE, and would carry the same LIVE-derived-weights concern as above. No verified-license NIQE implementation was added.
- **clip_iqa**: skipped. piq.CLIPIQA() automatically downloads a CLIP RN50 checkpoint (piq's own mirror of OpenAI's MIT-licensed CLIP weights, piq/releases/download/v0.7.1/RN50.pt, on the order of a few hundred MB) the first time it runs. That is both a large, unprompted download this task should not make on its own, and, in spirit, loading a vision-language model -- which the task explicitly ruled out for this CPU-only run. Skipped for both reasons; not attempted.

## Summary (mean over scales in each benchmark)

| Benchmark | Scales | Accuracy (full calibration) | Accuracy (small calibration) | Within 1 level | MAE (levels) | Spearman vs -DMOS |
| --- | --- | --- | --- | --- | --- | --- |
| lab | 5 | 0.979 | 0.752 | 1.000 | 0.040 | - |
| distort25 | 25 | 0.651 | 0.443 | 0.909 | 0.460 | - |
| kadid | 25 | 0.620 | 0.401 | 0.868 | 0.564 | 0.629 |

For context, the lab scales' published VLM numbers: `ens4d` mean accuracy 0.867; v0 as shipped, mean accuracy 0.500 (per-scale v0 numbers are not published, so only the mean is shown). No VLM numbers are given for distort25 or KADID: those runs are not finished.

## Lab scales (blur, noise, jpeg, exposure, resolution; 4 levels, 1,000 items each)

| Scale | Levels | n test | Acc (full) | Acc (n=~32) | Within 1 | MAE | VLM `ens4d` acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur | 4 | 500 | 0.994 | 0.748 ± 0.034 (n=32) | 1.000 | 0.021 | 0.888 |
| exposure | 4 | 500 | 0.986 | 0.868 ± 0.042 (n=32) | 1.000 | 0.043 | 0.924 |
| jpeg | 4 | 500 | 0.990 | 0.761 ± 0.054 (n=32) | 1.000 | 0.015 | 0.772 |
| noise | 4 | 500 | 0.958 | 0.683 ± 0.034 (n=32) | 0.998 | 0.066 | 0.864 |
| resolution | 4 | 500 | 0.968 | 0.699 ± 0.051 (n=32) | 1.000 | 0.057 | 0.888 |

## distort25 (25 distortions, 5 levels, 300 items each)

| Scale | Levels | n test | Acc (full) | Acc (n=~30) | Within 1 | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| brighten | 5 | 150 | 0.433 | 0.359 ± 0.033 (n=30) | 0.867 | 0.678 |
| color_block | 5 | 150 | 0.373 | 0.278 ± 0.033 (n=30) | 0.780 | 0.838 |
| color_diffusion | 5 | 150 | 0.207 | 0.205 ± 0.012 (n=30) | 0.527 | 1.348 |
| color_noise | 5 | 150 | 0.753 | 0.495 ± 0.049 (n=30) | 0.987 | 0.308 |
| color_quantization | 5 | 150 | 0.920 | 0.471 ± 0.061 (n=30) | 1.000 | 0.128 |
| color_saturation_down | 5 | 150 | 0.493 | 0.382 ± 0.046 (n=30) | 0.893 | 0.552 |
| color_saturation_up | 5 | 150 | 0.340 | 0.267 ± 0.040 (n=30) | 0.733 | 0.934 |
| color_shift | 5 | 150 | 0.480 | 0.343 ± 0.052 (n=30) | 0.933 | 0.556 |
| contrast_change | 5 | 150 | 0.860 | 0.640 ± 0.054 (n=30) | 0.993 | 0.173 |
| darken | 5 | 150 | 0.873 | 0.660 ± 0.042 (n=30) | 1.000 | 0.165 |
| denoise | 5 | 150 | 0.727 | 0.457 ± 0.042 (n=30) | 0.993 | 0.298 |
| gaussian_blur | 5 | 150 | 0.827 | 0.536 ± 0.055 (n=30) | 1.000 | 0.205 |
| impulse_noise | 5 | 150 | 0.847 | 0.557 ± 0.061 (n=30) | 0.980 | 0.216 |
| jitter | 5 | 150 | 0.807 | 0.381 ± 0.039 (n=30) | 0.993 | 0.237 |
| jpeg | 5 | 150 | 0.927 | 0.587 ± 0.070 (n=30) | 1.000 | 0.135 |
| jpeg2000 | 5 | 150 | 0.720 | 0.450 ± 0.037 (n=30) | 0.973 | 0.352 |
| lens_blur | 5 | 150 | 0.673 | 0.482 ± 0.056 (n=30) | 1.000 | 0.336 |
| mean_shift | 5 | 150 | 0.547 | 0.391 ± 0.041 (n=30) | 0.960 | 0.487 |
| motion_blur | 5 | 150 | 0.360 | 0.327 ± 0.034 (n=30) | 0.827 | 0.813 |
| multiplicative_noise | 5 | 150 | 0.853 | 0.547 ± 0.058 (n=30) | 0.987 | 0.202 |
| patch_shuffle | 5 | 150 | 0.207 | 0.199 ± 0.029 (n=30) | 0.473 | 1.295 |
| pixelate | 5 | 150 | 0.940 | 0.629 ± 0.054 (n=30) | 1.000 | 0.119 |
| quantization | 5 | 150 | 0.880 | 0.561 ± 0.054 (n=30) | 1.000 | 0.161 |
| sharpen | 5 | 150 | 0.353 | 0.322 ± 0.036 (n=30) | 0.833 | 0.801 |
| white_noise | 5 | 150 | 0.880 | 0.536 ± 0.044 (n=30) | 0.993 | 0.174 |

## KADID-10k (25 distortions, 5 levels, 405 items each; evaluation only)

| Scale | Levels | n test | Acc (full) | Acc (n=~30) | Within 1 | MAE | Spearman vs -DMOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brighten | 5 | 200 | 0.475 | 0.369 ± 0.029 (n=30) | 0.825 | 0.763 | 0.653 |
| color_block | 5 | 200 | 0.230 | 0.210 ± 0.012 (n=30) | 0.495 | 1.247 | 0.113 |
| color_diffusion | 5 | 200 | 0.505 | 0.372 ± 0.034 (n=30) | 0.830 | 0.727 | 0.820 |
| color_noise | 5 | 200 | 0.870 | 0.465 ± 0.036 (n=30) | 1.000 | 0.178 | 0.897 |
| color_quantization | 5 | 200 | 0.980 | 0.472 ± 0.042 (n=30) | 1.000 | 0.079 | 0.799 |
| color_saturation_1 | 5 | 200 | 0.780 | 0.504 ± 0.036 (n=30) | 0.850 | 0.694 | 0.359 |
| color_saturation_2 | 5 | 200 | 0.450 | 0.376 ± 0.036 (n=30) | 0.860 | 0.678 | 0.780 |
| color_shift | 5 | 200 | 0.345 | 0.236 ± 0.017 (n=30) | 0.670 | 1.050 | 0.349 |
| contrast_change | 5 | 200 | 0.520 | 0.408 ± 0.030 (n=30) | 0.885 | 0.602 | 0.007 |
| darken | 5 | 200 | 0.485 | 0.400 ± 0.024 (n=30) | 0.775 | 0.890 | 0.624 |
| denoise | 5 | 200 | 0.800 | 0.480 ± 0.038 (n=30) | 0.975 | 0.270 | 0.871 |
| gaussian_blur | 5 | 200 | 0.830 | 0.554 ± 0.040 (n=30) | 0.990 | 0.211 | 0.921 |
| impulse_noise | 5 | 200 | 0.680 | 0.393 ± 0.029 (n=30) | 0.950 | 0.374 | 0.814 |
| jitter | 5 | 200 | 0.805 | 0.455 ± 0.050 (n=30) | 0.970 | 0.288 | 0.883 |
| jpeg | 5 | 200 | 0.890 | 0.454 ± 0.038 (n=30) | 0.990 | 0.181 | 0.866 |
| jpeg2000 | 5 | 200 | 0.665 | 0.466 ± 0.042 (n=30) | 0.925 | 0.430 | 0.733 |
| lens_blur | 5 | 200 | 0.815 | 0.473 ± 0.047 (n=30) | 0.995 | 0.232 | 0.888 |
| mean_shift | 5 | 200 | 0.400 | 0.361 ± 0.031 (n=30) | 0.770 | 0.922 | 0.230 |
| motion_blur | 5 | 200 | 0.545 | 0.376 ± 0.036 (n=30) | 0.900 | 0.593 | 0.742 |
| multiplicative_noise | 5 | 200 | 0.830 | 0.467 ± 0.037 (n=30) | 1.000 | 0.211 | 0.895 |
| patch_shuffle | 5 | 200 | 0.190 | 0.199 ± 0.014 (n=30) | 0.555 | 1.209 | -0.202 |
| pixelate | 5 | 200 | 0.605 | 0.383 ± 0.040 (n=30) | 0.820 | 0.626 | 0.678 |
| quantization | 5 | 200 | 0.525 | 0.341 ± 0.033 (n=30) | 0.875 | 0.590 | 0.501 |
| sharpen | 5 | 200 | 0.445 | 0.344 ± 0.024 (n=30) | 0.800 | 0.836 | 0.673 |
| white_noise | 5 | 200 | 0.830 | 0.459 ± 0.032 (n=30) | 1.000 | 0.221 | 0.844 |


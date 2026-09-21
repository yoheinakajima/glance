# The one-pass JSON-position read on the harder rating benchmarks (E17, zero-shot)

## creative-QA rubrics

| Scale | n | `jsondigits`, one pass | raw `ens4d`, four passes | `jsondigits` within one | `jsondigits` + 16 unlabeled |
| --- | --- | --- | --- | --- | --- |
| cutoff | 300 | 0.253 | 0.267 | 0.707 | 0.289 |
| occlusion | 300 | 0.510 | 0.350 | 0.913 | 0.380 |
| text_legibility | 300 | 0.403 | 0.433 | 0.673 | 0.425 |
| tilt | 300 | 0.230 | 0.223 | 0.453 | 0.193 |
| watermark | 300 | 0.480 | 0.397 | 0.960 | 0.542 |
| **mean** | | **0.375** | **0.334** | 0.741 | 0.366 |

## KADID-10k (23 severity distortions)

| Scale | n | `jsondigits`, one pass | raw `ens4d`, four passes | `jsondigits` within one | `jsondigits` + 16 unlabeled |
| --- | --- | --- | --- | --- | --- |
| brighten | 200 | 0.280 | 0.285 | 0.635 | 0.293 |
| color_block | 200 | 0.280 | 0.240 | 0.705 | 0.323 |
| color_diffusion | 200 | 0.430 | 0.430 | 0.765 | 0.406 |
| color_noise | 200 | 0.535 | 0.510 | 0.965 | 0.569 |
| color_quantization | 200 | 0.410 | 0.380 | 0.820 | 0.441 |
| color_saturation_1 | 200 | 0.095 | 0.045 | 0.485 | 0.071 |
| color_saturation_2 | 200 | 0.265 | 0.265 | 0.610 | 0.307 |
| color_shift | 200 | 0.335 | 0.345 | 0.625 | 0.327 |
| darken | 200 | 0.220 | 0.215 | 0.505 | 0.269 |
| denoise | 200 | 0.395 | 0.340 | 0.800 | 0.451 |
| gaussian_blur | 200 | 0.510 | 0.515 | 0.960 | 0.606 |
| impulse_noise | 200 | 0.305 | 0.270 | 0.720 | 0.345 |
| jitter | 200 | 0.490 | 0.450 | 0.890 | 0.468 |
| jpeg | 200 | 0.210 | 0.205 | 0.540 | 0.272 |
| jpeg2000 | 200 | 0.200 | 0.200 | 0.415 | 0.214 |
| lens_blur | 200 | 0.345 | 0.335 | 0.585 | 0.323 |
| motion_blur | 200 | 0.350 | 0.285 | 0.630 | 0.329 |
| multiplicative_noise | 200 | 0.530 | 0.455 | 0.950 | 0.638 |
| patch_shuffle | 200 | 0.290 | 0.270 | 0.580 | 0.277 |
| pixelate | 200 | 0.315 | 0.405 | 0.710 | 0.382 |
| quantization | 200 | 0.370 | 0.315 | 0.775 | 0.388 |
| sharpen | 200 | 0.370 | 0.305 | 0.725 | 0.373 |
| white_noise | 200 | 0.455 | 0.480 | 0.945 | 0.546 |
| **mean** | | **0.347** | **0.328** | 0.710 | 0.375 |

Registered predictions and decision (`lab/NOTES.md` entry 43):

- H42: creative-QA, jsondigits >= raw ens4d on the mean and on at least 3 of 5 rubrics: YES
- H43a: KADID, jsondigits beats raw ens4d by >= 5 points: NO
- H43b: KADID, jsondigits stays below 0.50: YES
- DECISION (registered rule): jsondigits becomes the zero-shot default for `score`: YES

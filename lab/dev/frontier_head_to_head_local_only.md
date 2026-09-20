# Frontier models vs Qwen3-VL-4B on the lab scales, same held-out images

Frontier: zero-shot, constrained to the rubric's levels, uncalibrated (a hard pick has no probabilities to calibrate); only whether each pick was right is stored. Local rows are recomputed from the lab's saved logits for the same item ids. The temperature of the v0 row was fit on the calibration split (it cannot change a prediction).

| System | labels used for this rubric | blur | exposure | jpeg | noise | resolution | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B, v0 readout as shipped (single temperature) | 0 | 0.625 | 0.530 | 0.365 | 0.550 | 0.480 | **0.510** |
| Qwen3-VL-4B, + Glance ens4d | 0 | 0.470 | 0.700 | 0.450 | 0.575 | 0.655 | **0.570** |
| Qwen3-VL-4B, + Glance ens4d | 32 | 0.906 | 0.921 | 0.738 | 0.850 | 0.871 | **0.857** |
| Qwen3-VL-4B, + Glance ens4d | 500 | 0.910 | 0.935 | 0.745 | 0.875 | 0.875 | **0.868** |

n per scale: blur 200, exposure 200, jpeg 200, noise 200, resolution 200.

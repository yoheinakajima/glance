# Frontier models vs Qwen3-VL-4B on the lab scales, same held-out images

Frontier: zero-shot, constrained to the rubric's levels, uncalibrated (a hard pick has no probabilities to calibrate); only whether each pick was right is stored. Local rows are recomputed from the lab's saved logits for the same item ids. The temperature of the v0 row was fit on the calibration split (it cannot change a prediction).

| System | labels used for this rubric | blur | exposure | jpeg | noise | resolution | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic/claude-opus-5 | 0 | 0.560 | 0.750 | 0.285 | 0.525 | 0.630 | **0.550** |
| openai/gpt-5.6 | 0 | 0.645 | 0.610 | 0.555 | 0.670 | 0.505 | **0.597** |
| openrouter/google/gemini-3.1-pro-preview | 0 | 0.665 | 0.655 | 0.645 | 0.590 | 0.695 | **0.650** |
| Qwen3-VL-4B, v0 readout as shipped (single temperature) | 0 | 0.625 | 0.530 | 0.365 | 0.550 | 0.480 | **0.510** |
| Qwen3-VL-4B, + Glance ens4d | 0 | 0.470 | 0.700 | 0.450 | 0.575 | 0.655 | **0.570** |
| Qwen3-VL-4B, + Glance ens4d | 0 (plus unlabeled images) | 0.610 | 0.840 | 0.565 | 0.740 | 0.755 | **0.702** |
| Qwen3-VL-4B, + Glance ens4d | 32 | 0.906 | 0.921 | 0.738 | 0.850 | 0.871 | **0.857** |
| Qwen3-VL-4B, + Glance ens4d | 500 | 0.910 | 0.935 | 0.745 | 0.875 | 0.875 | **0.868** |

Paired differences in mean accuracy, same images (bootstrap 95% interval over items):

| Difference | points | 95% interval |
| --- | --- | --- |
| v0 readout as shipped (single temperature), 0 rubric labels minus anthropic/claude-opus-5 | -4.0 | [-8.0, +0.0] |
| + Glance ens4d, 0 labels minus anthropic/claude-opus-5 | +2.0 | [-2.1, +6.0] |
| + Glance ens4d, 0 labels + unlabeled images minus anthropic/claude-opus-5 | +15.2 | [+10.8, +19.5] |
| + Glance ens4d, 32 labels minus anthropic/claude-opus-5 | +30.7 | [+27.3, +34.2] |
| + Glance ens4d, 500 labels minus anthropic/claude-opus-5 | +31.8 | [+28.2, +35.4] |
| v0 readout as shipped (single temperature), 0 rubric labels minus openai/gpt-5.6 | -8.7 | [-13.3, -4.2] |
| + Glance ens4d, 0 labels minus openai/gpt-5.6 | -2.7 | [-7.1, +1.8] |
| + Glance ens4d, 0 labels + unlabeled images minus openai/gpt-5.6 | +10.5 | [+6.1, +14.9] |
| + Glance ens4d, 32 labels minus openai/gpt-5.6 | +26.0 | [+22.6, +29.4] |
| + Glance ens4d, 500 labels minus openai/gpt-5.6 | +27.1 | [+23.3, +30.7] |
| v0 readout as shipped (single temperature), 0 rubric labels minus openrouter/google/gemini-3.1-pro-preview | -14.0 | [-18.7, -9.3] |
| + Glance ens4d, 0 labels minus openrouter/google/gemini-3.1-pro-preview | -8.0 | [-12.4, -3.6] |
| + Glance ens4d, 0 labels + unlabeled images minus openrouter/google/gemini-3.1-pro-preview | +5.2 | [+1.3, +9.3] |
| + Glance ens4d, 32 labels minus openrouter/google/gemini-3.1-pro-preview | +20.7 | [+17.5, +24.1] |
| + Glance ens4d, 500 labels minus openrouter/google/gemini-3.1-pro-preview | +21.8 | [+18.3, +25.2] |

n per scale: blur 200, exposure 200, jpeg 200, noise 200, resolution 200.

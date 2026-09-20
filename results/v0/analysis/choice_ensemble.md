# Combining the two local backends on multiple choice (offline, saved logits)

Fit on the calibration split (two temperatures + one mixing weight, by NLL); reported on the test split.

| Suite | n test | Qwen3-VL-4B | SigLIP2 | Combined | w (VLM share) | Combined ECE (floor) | Claude Opus 5 (n) | Combined sel. acc @80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | 250 | 0.892 | 0.956 | **0.952** | 0.46 | 0.015 (0.014) | 0.932 (250) | 0.990 |
| caltech101 | 221 | 0.919 | 0.932 | **0.937** | 0.28 | 0.026 (0.019) | 0.968 (221) | 1.000 |

Frontier accuracy is on the same test items (all of them for these two suites).

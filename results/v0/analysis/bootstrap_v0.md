# v0 out of the box: bootstrap 95% intervals (test split, uncalibrated decisions, 10,000 resamples)

| Suite | n | Qwen3-VL-4B (primary method) | Claude Opus 5 (zero-shot pick) | Opus minus local, points |
| --- | --- | --- | --- | --- |
| pope | 250 | 0.880 [0.840, 0.920] | 0.920 [0.884, 0.952] | +4.0 [+0.4, +7.6] |
| gqa_yesno | 250 | 0.732 [0.676, 0.784] | 0.732 [0.676, 0.784] | +0.0 [-6.0, +6.0] |
| pets37 | 250 | 0.892 [0.852, 0.928] | 0.932 [0.900, 0.960] | +4.0 [+0.0, +8.0] |
| caltech101 | 221 | 0.919 [0.882, 0.955] | 0.968 [0.941, 0.991] | +5.0 [+1.8, +8.6] |
| blur_ladder | 250 | 0.496 [0.432, 0.556] | 0.536 [0.472, 0.596] | +4.0 [-4.4, +12.0] |

mean gap, all five suites: +3.4 points [+1.0, +5.8] (Opus ahead when positive).

mean gap, yes/no and choice suites only (no blur_ladder): +3.2 points [+1.0, +5.4] (Opus ahead when positive).

pope, other local rows: siglip:statement 0.588 [0.528, 0.648]

pets37, other local rows: siglip:independent 0.956 [0.928, 0.980]; vlm:letter 0.904 [0.864, 0.940]

caltech101, other local rows: siglip:independent 0.932 [0.896, 0.964]; vlm:letter 0.977 [0.955, 0.995]

blur_ladder, other local rows: siglip:statement 0.344 [0.284, 0.404]

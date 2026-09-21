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

Registered predictions and decision (`lab/NOTES.md` entry 43):

- H42: creative-QA, jsondigits >= raw ens4d on the mean and on at least 3 of 5 rubrics: YES

# Label-free batch calibration of `ens4d` on the lab scales: DEV (calibration split only)

| Setting | blur | exposure | jpeg | noise | resolution | mean accuracy | within one | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw (0 labels, no pool) | 0.448 | 0.680 | 0.412 | 0.556 | 0.664 | **0.552** | 0.986 | 0.469 |
| BC, pool = all 250 unlabeled | 0.512 | 0.824 | 0.416 | 0.756 | 0.628 | **0.627** | 0.982 | 0.383 |
| BC, pool = 16 unlabeled (20 draws) | 0.515 | 0.830 | 0.438 | 0.739 | 0.637 | **0.632** | 0.980 | 0.384 |
| BC, pool = 32 unlabeled (20 draws) | 0.538 | 0.818 | 0.422 | 0.744 | 0.651 | **0.635** | 0.981 | 0.380 |
| BC, pool = 64 unlabeled (20 draws) | 0.529 | 0.819 | 0.422 | 0.740 | 0.632 | **0.628** | 0.981 | 0.383 |
| BC, unbalanced pool of 60 (70% one level) | 0.578 | 0.728 | 0.449 | 0.686 | 0.588 | **0.606** | 0.969 | 0.424 |
| BCz, pool = all 250 unlabeled | 0.544 | 0.844 | 0.552 | 0.724 | 0.780 | **0.689** | 0.988 | 0.463 |
| BCz, pool = 16 unlabeled (20 draws) | 0.571 | 0.825 | 0.547 | 0.706 | 0.753 | **0.680** | 0.983 | 0.464 |
| BCz, pool = 32 unlabeled (20 draws) | 0.564 | 0.861 | 0.565 | 0.712 | 0.761 | **0.693** | 0.987 | 0.467 |
| BCz, pool = 64 unlabeled (20 draws) | 0.566 | 0.849 | 0.548 | 0.713 | 0.779 | **0.691** | 0.987 | 0.463 |
| BCz, unbalanced pool of 60 (70% one level) | 0.599 | 0.742 | 0.547 | 0.679 | 0.663 | **0.646** | 0.976 | 0.497 |

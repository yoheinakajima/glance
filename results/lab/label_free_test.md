# Label-free batch calibration of `ens4d` on the lab scales: TEST split, scored once

| Setting | blur | exposure | jpeg | noise | resolution | mean accuracy | within one | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw (0 labels, no pool) | 0.462 | 0.680 | 0.412 | 0.574 | 0.660 | **0.558** | 0.987 | 0.470 |
| BCz, pool = all 500 unlabeled | 0.594 | 0.832 | 0.542 | 0.750 | 0.768 | **0.697** | 0.985 | 0.458 |
| BCz, pool = 16 unlabeled (20 draws) | 0.603 | 0.821 | 0.554 | 0.724 | 0.726 | **0.686** | 0.984 | 0.453 |
| BCz, pool = 32 unlabeled (20 draws) | 0.604 | 0.806 | 0.543 | 0.756 | 0.762 | **0.694** | 0.984 | 0.454 |
| BCz, pool = 64 unlabeled (20 draws) | 0.590 | 0.821 | 0.549 | 0.746 | 0.761 | **0.693** | 0.985 | 0.461 |
| BCz, unbalanced pool of 100 (70% one level) | 0.620 | 0.708 | 0.547 | 0.698 | 0.659 | **0.646** | 0.976 | 0.493 |

# Zero-shot ratings: calibration before and after self-calibration from unlabeled images (lab test split)

| Setting | exact accuracy | ECE (sampling floor) | NLL |
| --- | --- | --- | --- |
| zero-shot (no pool) | 0.558 | 0.327 (0.033) | 1.94 |
| 16 unlabeled images | 0.673 | 0.175 (0.066) | 0.82 |
| 64 unlabeled images | 0.699 | 0.200 (0.067) | 0.81 |
| 500 unlabeled images | 0.697 | 0.208 (0.068) | 0.81 |

A labeled fit (`ens4d` + matrix, 500 labels) has ECE about 0.03 on the same split (`docs/paper/RESULTS_LAB.md`). Pool draws differ from `label_free_test.json`, so the 16-image accuracy differs from it in the third decimal.

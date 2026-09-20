# Content-free prior for zero-shot ratings (E14, lab test split, no labels and no images of the task)

| Setting | blur | exposure | jpeg | noise | resolution | mean exact accuracy | within one level | MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw zero-shot | 0.462 | 0.680 | 0.412 | 0.574 | 0.660 | **0.558** | 0.987 | 0.470 |
| content-free prior, all six null images (registered) | 0.492 | 0.404 | 0.358 | 0.326 | 0.418 | **0.400** | 0.998 | 0.606 |
| content-free prior, flat grey only | 0.412 | 0.414 | 0.412 | 0.472 | 0.418 | **0.426** | 0.998 | 0.590 |
| content-free prior, three noise images only | 0.492 | 0.368 | 0.300 | 0.370 | 0.414 | **0.389** | 0.917 | 0.677 |
| self-calibration from 16 unlabeled images (reference, entry 26) | 0.603 | 0.790 | 0.560 | 0.715 | 0.729 | **0.679** | 0.984 | 0.459 |

Level the null images are read as (argmax of the prior, per member): blur: [3]; exposure: [0, 3]; jpeg: [2, 3]; noise: [2, 3]; resolution: [3].

Verdict: gain over raw -15.8 points; H37 gain >= 3 points: NOT SUPPORTED; below self-calibration: yes; hurts on: ['exposure', 'jpeg', 'noise', 'resolution'].

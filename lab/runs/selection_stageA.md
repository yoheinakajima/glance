# Score lab: method selection on the calibration split only

5-fold cross-validation inside the calibration split of each scale. The test split was not read.
Ranking criterion (fixed in advance): mean cross-validated NLL over the scales, lower is better.

| Rank | Candidate | Passes | Mean CV NLL | Mean CV accuracy | blur acc | exposure acc | jpeg acc | noise acc | resolution acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `ens5(all five)` | 12 | 0.3227 | 0.866 | 0.888 | 0.930 | 0.762 | 0.874 | 0.876 |
| 2 | `ens3(independent+digits+zoom_digits)` | 6 | 0.3412 | 0.859 | 0.882 | 0.926 | 0.754 | 0.838 | 0.894 |
| 3 | `ens2(digits+zoom_digits)` | 2 | 0.3440 | 0.857 | 0.874 | 0.916 | 0.770 | 0.838 | 0.888 |
| 4 | `ens4(independent+zoom_digits)` | 5 | 0.3538 | 0.850 | 0.876 | 0.928 | 0.758 | 0.842 | 0.844 |
| 5 | `zoom_digits` | 1 | 0.3832 | 0.841 | 0.868 | 0.912 | 0.772 | 0.842 | 0.812 |
| 6 | `digits` | 1 | 0.4558 | 0.817 | 0.848 | 0.896 | 0.656 | 0.820 | 0.864 |
| 7 | `zoom_cumulative` | 3 | 0.4767 | 0.785 | 0.766 | 0.906 | 0.710 | 0.848 | 0.694 |
| 8 | `independent` | 4 | 0.4846 | 0.806 | 0.872 | 0.920 | 0.626 | 0.790 | 0.822 |
| 9 | `cumulative` | 3 | 0.5464 | 0.756 | 0.806 | 0.826 | 0.586 | 0.836 | 0.728 |

**Winner, any cost:** `ens5(all five)`.
**Winner within 4 forward passes:** `ens2(digits+zoom_digits)`.

Calibration kind chosen per scale (lowest CV NLL): `ens5(all five)`: blur=matrix, exposure=matrix, jpeg=matrix, noise=matrix, resolution=matrix; `ens2(digits+zoom_digits)`: blur=matrix, exposure=matrix, jpeg=matrix, noise=matrix, resolution=matrix

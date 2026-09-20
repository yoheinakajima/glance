# Score lab: method selection on the calibration split only

5-fold cross-validation inside the calibration split of each scale. The test split was not read.
Ranking criterion (fixed in advance): mean cross-validated NLL over the scales, lower is better.

| Rank | Candidate | Passes | Mean CV NLL | Mean CV accuracy | blur acc | exposure acc | jpeg acc | noise acc | resolution acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `ens7` | 14 | 0.3007 | 0.884 | 0.896 | 0.934 | 0.792 | 0.884 | 0.916 |
| 2 | `ens4d` | 4 | 0.3135 | 0.875 | 0.876 | 0.928 | 0.798 | 0.858 | 0.914 |
| 3 | `ens5` | 12 | 0.3227 | 0.866 | 0.888 | 0.930 | 0.762 | 0.874 | 0.876 |
| 4 | `ens3` | 6 | 0.3412 | 0.859 | 0.882 | 0.926 | 0.754 | 0.838 | 0.894 |
| 5 | `ens2` | 2 | 0.3440 | 0.857 | 0.874 | 0.916 | 0.770 | 0.838 | 0.888 |
| 6 | `ens2r` | 2 | 0.3453 | 0.863 | 0.860 | 0.934 | 0.772 | 0.854 | 0.896 |
| 7 | `zoom_digits` | 1 | 0.3832 | 0.841 | 0.868 | 0.912 | 0.772 | 0.842 | 0.812 |
| 8 | `zoom_digitsrev` | 1 | 0.4196 | 0.837 | 0.798 | 0.920 | 0.732 | 0.862 | 0.872 |
| 9 | `digitsrev` | 1 | 0.4374 | 0.819 | 0.858 | 0.910 | 0.656 | 0.832 | 0.840 |
| 10 | `digits` | 1 | 0.4558 | 0.817 | 0.848 | 0.896 | 0.656 | 0.820 | 0.864 |
| 11 | `zoom_cumulative` | 3 | 0.4767 | 0.785 | 0.766 | 0.906 | 0.710 | 0.848 | 0.694 |
| 12 | `independent` | 4 | 0.4846 | 0.806 | 0.872 | 0.920 | 0.626 | 0.790 | 0.822 |
| 13 | `cumulative` | 3 | 0.5464 | 0.756 | 0.806 | 0.826 | 0.586 | 0.836 | 0.728 |

**Winner, any cost:** `ens7`.
**Winner within 4 forward passes:** `ens4d`.

Calibration kind chosen per scale (lowest CV NLL): `ens7`: blur=matrix, exposure=matrix, jpeg=matrix, noise=matrix, resolution=matrix; `ens4d`: blur=matrix, exposure=matrix, jpeg=matrix, noise=matrix, resolution=matrix

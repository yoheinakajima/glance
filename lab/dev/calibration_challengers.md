# Dev: calibration challengers (lab calibration split only; lab/NOTES.md entry 19)

Fit on n labels drawn from the first half of each scale's calibration split, judged on its second half; mean over 5 scales x 10 draws.

**acc**

| method | n=8 | n=16 | n=32 | n=64 | n=128 | n=240 |
| --- | --- | --- | --- | --- | --- | --- |
| current (matrix, L2 to zero) | 0.801 | 0.826 | 0.843 | 0.858 | 0.863 | 0.864 |
| C1 prior-centred | 0.788 | 0.817 | 0.832 | 0.842 | 0.852 | 0.858 |
| C2 ordinal 1-D | 0.553 | 0.826 | 0.844 | 0.856 | 0.858 | 0.856 |

**nll**

| method | n=8 | n=16 | n=32 | n=64 | n=128 | n=240 |
| --- | --- | --- | --- | --- | --- | --- |
| current (matrix, L2 to zero) | 1.268 | 0.542 | 0.423 | 0.371 | 0.352 | 0.345 |
| C1 prior-centred | 1.037 | 0.698 | 0.499 | 0.441 | 0.422 | 0.409 |
| C2 ordinal 1-D | 1.135 | 0.482 | 0.403 | 0.373 | 0.353 | 0.346 |

**mae**

| method | n=8 | n=16 | n=32 | n=64 | n=128 | n=240 |
| --- | --- | --- | --- | --- | --- | --- |
| current (matrix, L2 to zero) | 0.929 | 0.243 | 0.220 | 0.194 | 0.190 | 0.185 |
| C1 prior-centred | 0.706 | 0.269 | 0.243 | 0.215 | 0.210 | 0.206 |
| C2 ordinal 1-D | 0.562 | 0.248 | 0.225 | 0.205 | 0.202 | 0.200 |

**ece**

| method | n=8 | n=16 | n=32 | n=64 | n=128 | n=240 |
| --- | --- | --- | --- | --- | --- | --- |
| current (matrix, L2 to zero) | 0.516 | 0.094 | 0.072 | 0.057 | 0.059 | 0.055 |
| C1 prior-centred | 0.410 | 0.109 | 0.084 | 0.067 | 0.068 | 0.067 |
| C2 ordinal 1-D | 0.305 | 0.097 | 0.073 | 0.065 | 0.062 | 0.061 |

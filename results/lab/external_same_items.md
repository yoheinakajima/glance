# Outside systems against the four-pass read (`ens4d`) on the same items with the same fit

Both systems: matrix scaling on their own level logits, the same 300 calibration items per scale to fit (or 32 of them, balanced, 20 draws), the same 300 test items per scale, five lab scales, exact-level accuracy, paired bootstrap 95% intervals. Added after the registered comparison (`results/lab/external_systems.md`) was seen; see `lab/NOTES.md` entry 37c.

## q-sit-mini (0.9B, trained for image quality)

| Labels per scale | outside system | Qwen3-VL-4B + Glance `ens4d` | outside minus `ens4d`, points |
| --- | --- | --- | --- |
| 300 | 0.869 [0.852, 0.886] | 0.863 [0.845, 0.881] | +0.6 [-1.7, +2.9] |
| 32 | 0.853 [0.838, 0.867] | 0.846 [0.832, 0.861] | +0.7 [-1.1, +2.5] |

| Scale | outside, 300 labels | `ens4d`, 300 labels | outside, 32 labels | `ens4d`, 32 labels |
| --- | --- | --- | --- | --- |
| blur | 0.963 | 0.887 | 0.956 | 0.883 |
| noise | 0.837 | 0.867 | 0.812 | 0.837 |
| jpeg | 0.707 | 0.757 | 0.671 | 0.738 |
| exposure | 0.883 | 0.930 | 0.881 | 0.919 |
| resolution | 0.957 | 0.877 | 0.945 | 0.855 |

## openjev v2 (4B, trained claim scorer)

| Labels per scale | outside system | Qwen3-VL-4B + Glance `ens4d` | outside minus `ens4d`, points |
| --- | --- | --- | --- |
| 300 | 0.772 [0.751, 0.793] | 0.863 [0.845, 0.881] | -9.1 [-11.4, -6.9] |
| 32 | 0.750 [0.731, 0.768] | 0.846 [0.832, 0.861] | -9.7 [-11.5, -7.9] |

| Scale | outside, 300 labels | `ens4d`, 300 labels | outside, 32 labels | `ens4d`, 32 labels |
| --- | --- | --- | --- | --- |
| blur | 0.837 | 0.887 | 0.826 | 0.883 |
| noise | 0.750 | 0.867 | 0.739 | 0.837 |
| jpeg | 0.653 | 0.757 | 0.637 | 0.738 |
| exposure | 0.877 | 0.930 | 0.831 | 0.919 |
| resolution | 0.743 | 0.877 | 0.715 | 0.855 |


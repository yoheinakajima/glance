# Reading the rating at the JSON answer position (E16): one pass, same 1,000 images as the frontier models

| Setting | passes | written answer | `jsondigits` read | raw `ens4d` read |
| --- | --- | --- | --- | --- |
| zero-shot | 1 / 1 / 4 | 0.672 | 0.669 | 0.570 |
| + 16 unlabeled images | | not possible | 0.758 | 0.694 |
| + 32 labels | | not possible | 0.822 | 0.853 |

`jsondigits` zero-shot: agrees with the written answer on 98.2% of items, within one level 0.979, ECE 0.248, probability mass outside the digits 0.000.

| Scale | written | `jsondigits` | raw `ens4d` |
| --- | --- | --- | --- |
| blur | 0.755 | 0.730 | 0.470 |
| exposure | 0.810 | 0.805 | 0.700 |
| jpeg | 0.345 | 0.345 | 0.450 |
| noise | 0.700 | 0.715 | 0.575 |
| resolution | 0.750 | 0.750 | 0.655 |

Registered predictions (`lab/NOTES.md` entry 42):

- H38a: argmax agrees with the written answer on >= 97% of items: SUPPORTED
- H38b: zero-shot accuracy within 1.5 points of the written answer: SUPPORTED
- H39: beats raw ens4d zero-shot by >= 8 points: SUPPORTED
- H40a: 16 unlabeled images add less than they add to ens4d: SUPPORTED
- H40b: with 16 unlabeled images it reaches >= 0.70: SUPPORTED
- H41: with 32 labels within 3 points of ens4d + matrix: NOT SUPPORTED

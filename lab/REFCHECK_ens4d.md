# Reference-path check for `ens4d`

Calibration fit on the cached calibration split, applied unchanged to cached and reference logits of the same test items.

| Scale | n | max abs logit diff | median abs logit diff | same prediction | accuracy cached | accuracy reference |
| --- | --- | --- | --- | --- | --- | --- |
| blur | 100 | 0.129 | 0.016 | 1.000 | 0.910 | 0.910 |
| exposure | 100 | 0.110 | 0.013 | 1.000 | 0.940 | 0.940 |
| jpeg | 100 | 0.116 | 0.015 | 1.000 | 0.800 | 0.800 |
| noise | 100 | 0.106 | 0.017 | 1.000 | 0.900 | 0.900 |
| resolution | 100 | 0.118 | 0.016 | 0.990 | 0.910 | 0.920 |

Over 500 items: same prediction on 0.998 of them; accuracy 0.892 cached vs 0.894 reference.

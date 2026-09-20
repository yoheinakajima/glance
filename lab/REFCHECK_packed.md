# Reference-path check for `ens4d`

Calibration fit on the cached calibration split, applied unchanged to cached and reference logits of the same test items.

| Scale | n | max abs logit diff | median abs logit diff | same prediction | accuracy cached | accuracy reference |
| --- | --- | --- | --- | --- | --- | --- |
| blur | 20 | 0.160 | 0.016 | 1.000 | 1.000 | 1.000 |
| exposure | 20 | 0.091 | 0.017 | 1.000 | 1.000 | 1.000 |
| jpeg | 20 | 0.088 | 0.017 | 1.000 | 0.900 | 0.900 |
| noise | 20 | 0.094 | 0.018 | 1.000 | 0.800 | 0.800 |
| resolution | 20 | 0.133 | 0.019 | 1.000 | 0.850 | 0.850 |

Over 100 items: same prediction on 1.000 of them; accuracy 0.910 cached vs 0.910 reference.

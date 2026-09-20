# Binary calibrators on the yes/no suites (offline, saved logits)

Source: `results/v0/analysis/noul_n1000/predictions.jsonl.gz`. Fit on the calibration split, judged on the test split. ECE uses 15 equal-mass bins.

| Suite | n cal / test | Calibrator | params | Acc | NLL | Brier | ECE | ECE floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_yesno | 500 / 500 | raw | - | 0.770 | 1.101 | 0.201 | 0.179 | 0.018 |
| gqa_yesno | 500 / 500 | Platt pooled (v0) | a=0.177 b=0.453 | 0.778 | 0.476 | 0.157 | 0.053 | 0.055 |
| gqa_yesno | 500 / 500 | Platt per suite | a=0.217 b=0.265 | 0.760 | 0.477 | 0.157 | 0.065 | 0.051 |
| gqa_yesno | 500 / 500 | asymmetric Platt per suite | a+=0.214 a-=0.221 c=0.247 | 0.760 | 0.477 | 0.157 | 0.069 | 0.051 |
| gqa_yesno | 500 / 500 | isotonic per suite | 24 knots | 0.782 | 0.490 | 0.157 | 0.056 | 0.048 |
| pope | 500 / 500 | raw | - | 0.880 | 1.274 | 0.115 | 0.111 | 0.003 |
| pope | 500 / 500 | Platt pooled (v0) | a=0.177 b=0.453 | 0.882 | 0.273 | 0.085 | 0.061 | 0.030 |
| pope | 500 / 500 | Platt per suite | a=0.174 b=1.058 | 0.886 | 0.247 | 0.077 | 0.056 | 0.033 |
| pope | 500 / 500 | asymmetric Platt per suite | a+=0.157 a-=0.219 c=0.774 | 0.890 | 0.249 | 0.078 | 0.049 | 0.032 |
| pope | 500 / 500 | isotonic per suite | 30 knots | 0.892 | 0.245 | 0.075 | 0.029 | 0.027 |

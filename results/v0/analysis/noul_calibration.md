# Binary calibrators on the yes/no suites (offline, saved logits)

Source: `results/v0/m5_full_eval/predictions.jsonl.gz`. Fit on the calibration split, judged on the test split. ECE uses 15 equal-mass bins.

| Suite | n cal / test | Calibrator | params | Acc | NLL | Brier | ECE | ECE floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gqa_yesno | 250 / 250 | raw | - | 0.732 | 1.302 | 0.232 | 0.204 | 0.026 |
| gqa_yesno | 250 / 250 | Platt pooled (v0) | a=0.184 b=0.557 | 0.744 | 0.529 | 0.178 | 0.093 | 0.074 |
| gqa_yesno | 250 / 250 | Platt per suite | a=0.237 b=0.283 | 0.728 | 0.541 | 0.181 | 0.109 | 0.070 |
| gqa_yesno | 250 / 250 | asymmetric Platt per suite | a+=0.186 a-=0.317 c=-0.009 | 0.724 | 0.548 | 0.183 | 0.096 | 0.069 |
| gqa_yesno | 250 / 250 | isotonic per suite | 22 knots | 0.728 | 0.601 | 0.193 | 0.133 | 0.062 |
| pope | 250 / 250 | raw | - | 0.880 | 1.245 | 0.118 | 0.114 | 0.004 |
| pope | 250 / 250 | Platt pooled (v0) | a=0.184 b=0.557 | 0.884 | 0.264 | 0.084 | 0.066 | 0.040 |
| pope | 250 / 250 | Platt per suite | a=0.198 b=1.623 | 0.884 | 0.237 | 0.073 | 0.057 | 0.041 |
| pope | 250 / 250 | asymmetric Platt per suite | a+=0.207 a-=0.168 c=1.784 | 0.884 | 0.236 | 0.073 | 0.064 | 0.041 |
| pope | 250 / 250 | isotonic per suite | 24 knots | 0.896 | 0.212 | 0.067 | 0.057 | 0.034 |

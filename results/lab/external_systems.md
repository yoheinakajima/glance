# Outside systems on this lab's scales

Registered design: `lab/NOTES.md` entry 37; `docs/paper/COMPARABLE_SYSTEMS.md`, "Four systems", items 1 and 2.
Calibration: `glance.rating.fit_matrix`/`apply_matrix`, fit on the calibration split (`rescale="train"`, `l2=0.05`), reported on the held-out test split.

## 1. OpenJevV2 (`AlexWortega/openjev`, v2 4B)

| Scale | n fit | n test | uncalibrated accuracy | calibrated accuracy | within 1 | MAE | median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur | 300 | 300 | 0.423 | 0.837 | 1.000 | 0.228 | 5443 |
| noise | 300 | 300 | 0.493 | 0.750 | 0.993 | 0.327 | 6271 |
| jpeg | 300 | 300 | 0.390 | 0.653 | 0.953 | 0.436 | 5123 |
| exposure | 300 | 300 | 0.387 | 0.877 | 0.997 | 0.159 | 4969 |
| resolution | 300 | 300 | 0.367 | 0.743 | 0.990 | 0.323 | 5346 |
| **mean** | | | **0.412** | **0.772** | 0.987 | 0.295 | 5297 |

**H31: SUPPORTED**

- yes: uncalibrated mean accuracy < 0.60 (got 0.412)
- yes: calibrated - uncalibrated >= 0.10 points (got +0.360)
- yes: calibrated mean accuracy < ens4d 0.867 (got 0.772)

## 2. QSitMini (`zhangzicheng/q-sit-mini`)

Direction: this lab's `level` field runs low (best) -> high (worst) for every one of these scales, the same direction q-sit-mini's own Excellent -> Bad scale runs in, so its natural high-for-good weighted score is sign-flipped below (`flip_sign=True`) before comparing it with `level`.

| Scale | n fit | n test | SRCC vs. true level | calibrated accuracy | median ms |
| --- | --- | --- | --- | --- | --- |
| blur | 300 | 300 | 0.949 | 0.963 | 2054 |
| noise | 300 | 300 | 0.934 | 0.837 | 2162 |
| jpeg | 300 | 300 | 0.862 | 0.707 | 2455 |
| exposure | 300 | 300 | 0.947 | 0.883 | 2233 |
| resolution | 300 | 300 | 0.925 | 0.957 | 2238 |
| **mean** | | | **0.923** | **0.869** | 2208 |

**H32: NOT SUPPORTED**

- yes: mean SRCC on blur/jpeg/noise >= 0.80 (got 0.915)
- no: mean SRCC on exposure/resolution < 0.80, i.e. "weak" (got 0.936)
- no: calibrated mean accuracy < ens4d 0.867 (got 0.869)


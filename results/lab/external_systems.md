# Outside systems on this lab's scales

Registered design: `lab/NOTES.md` entry 37; `docs/paper/COMPARABLE_SYSTEMS.md`, "Four systems", items 1 and 2.
Calibration: `glance.rating.fit_matrix`/`apply_matrix`, fit on the calibration split (`rescale="train"`, `l2=0.05`), reported on the held-out test split.

## 1. OpenJevV2 (`AlexWortega/openjev`, v2 4B)

not collected yet (`lab/runs/external_openjev.jsonl` has no rows)

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


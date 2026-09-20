# Outside systems on this lab's scales

Registered design: `lab/NOTES.md` entry 37; `docs/paper/COMPARABLE_SYSTEMS.md`, "Four systems", items 1 and 2.
Calibration: `glance.rating.fit_matrix`/`apply_matrix`, fit on the calibration split (`rescale="train"`, `l2=0.05`), reported on the held-out test split.

## 1. OpenJevV2 (`AlexWortega/openjev`, v2 4B)

not collected yet (`lab/runs/external_openjev.jsonl` has no rows)

## 2. QSitMini (`zhangzicheng/q-sit-mini`)

Direction: this lab's `level` field runs low (best) -> high (worst) for every one of these scales, the same direction q-sit-mini's own Excellent -> Bad scale runs in, so its natural high-for-good weighted score is sign-flipped below (`flip_sign=True`) before comparing it with `level`.

not collected yet (`lab/runs/external_qsit.jsonl` has no rows)


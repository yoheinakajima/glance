# Follow-up analyses after the v0 evaluation (2026-09-20)

These analyses were run after the v0 full evaluation and its frontier baseline. All but the last use only saved
logits; none involves training. Every number is copied from the cited file. The rating-scale experiments are
documented separately in `docs/paper/RESULTS_LAB.md` (written when the full lab run finishes) and, as a live
notebook, in `lab/NOTES.md`.

## 1. Combining the two local backends on multiple choice

Script: `tools/analyze_choice_ensemble.py`. Output: `results/v0/analysis/choice_ensemble.{md,json}`.
Input: `results/v0/m5_full_eval/predictions.jsonl.gz`. Combination: weighted sum of each backend's temperature-scaled
log-probabilities over all options; two temperatures and the mixing weight are fit by NLL on the calibration split;
numbers below are on the test split, on the same items the frontier baseline answered.

| Suite | n test | Qwen3-VL-4B | SigLIP2 | Combined | VLM weight | Combined ECE (floor) | Claude Opus 5 | Combined sel. acc @80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pets37 | 250 | 0.892 | 0.956 | 0.952 | 0.46 | 0.015 (0.014) | 0.932 | 0.990 |
| caltech101 | 221 | 0.919 | 0.932 | 0.937 | 0.28 | 0.026 (0.019) | 0.968 | 1.000 |

Reading: the combination does not beat the better single backend on accuracy (pets37: 0.952 vs 0.956 for SigLIP2
alone; caltech101: 0.937 vs 0.932). It is very well calibrated (ECE at its sampling floor) and nearly error-free on
its most confident 80% of items. On pets37 both the dual encoder and the combination are ahead of the frontier
baseline; on caltech101 the frontier baseline stays ahead by 3.1 points. Nine of Qwen3-VL-4B's 18 caltech101 errors
are the dataset's `Faces_easy` -> `Faces` pair (`results/v0/m5_full_eval/report.md`, error breakdown).

## 2. Yes/no calibration: the ECE gate at n = 250 versus n = 1,000

Script: `tools/analyze_noul_calibration.py`. Five binary calibrators, each fit on the calibration split and judged
on the test split: raw sigmoid(z); Platt scaling pooled over suites (what v0 ships); Platt per suite; asymmetric Platt
(three numbers: separate slopes for the yes side and the no side of z); isotonic regression per suite.

At the v0 sample size (250 calibration / 250 test items; `results/v0/analysis/noul_calibration.md`):

| Suite | Platt pooled (v0) | Platt per suite | asymmetric Platt | isotonic | sampling floor |
| --- | --- | --- | --- | --- | --- |
| pope | 0.066 | 0.057 | 0.064 | 0.057 | 0.034-0.041 |
| gqa_yesno | 0.093 | 0.109 | 0.096 | 0.133 | 0.062-0.074 |

At the full manifest size (500 / 500; new reference-path run snapshotted in `results/v0/analysis/noul_n1000/`, table
in `results/v0/analysis/noul_n1000/noul_calibration.md`):

| Suite | Calibrator | Accuracy | NLL | ECE | ECE floor |
| --- | --- | --- | --- | --- | --- |
| pope | raw | 0.880 | 1.274 | 0.111 | 0.003 |
| pope | Platt pooled (v0) | 0.882 | 0.273 | 0.061 | 0.030 |
| pope | Platt per suite | 0.886 | 0.247 | 0.056 | 0.033 |
| pope | asymmetric Platt per suite | 0.890 | 0.249 | 0.049 | 0.032 |
| pope | isotonic per suite | 0.892 | 0.245 | 0.029 | 0.027 |
| gqa_yesno | raw | 0.770 | 1.101 | 0.179 | 0.018 |
| gqa_yesno | Platt pooled (v0) | 0.778 | 0.476 | 0.053 | 0.055 |
| gqa_yesno | Platt per suite | 0.760 | 0.477 | 0.065 | 0.051 |
| gqa_yesno | asymmetric Platt per suite | 0.760 | 0.477 | 0.069 | 0.051 |
| gqa_yesno | isotonic per suite | 0.782 | 0.490 | 0.056 | 0.048 |

Reading: at 250 test items no calibrator could be told apart from another, and on gqa_yesno the sampling floor alone
was above the 0.05 gate. At 500 test items pope meets the gate with two calibrators (0.029 isotonic, 0.049 asymmetric
Platt) and gqa_yesno sits at 0.053-0.056, inside its own floor of 0.048-0.055. The v0 "ECE failure" on yes/no
questions is therefore mostly a sample-size effect of equal-mass ECE with 15 bins. What is real is domain dependence
of the offset: on pope a per-suite fit beats the pooled one at both sample sizes. HANDOFF allows isotonic regression
only with at least 1,000 calibration examples; it had 500 here, so this is analysis, not a change to the harness.

## 3. What this does to the v0 verdict

The v0 go/no-go table (`results/v0/m5_full_eval/report.md`) fails only the ECE gate, on pope (0.066), gqa_yesno
(0.093) and blur_ladder (0.149). Section 2 shows the first two are at or under the gate once there is enough data to
measure them. That leaves rating scales (`score`) as the one question type where post-hoc calibration of the v0
readout is not enough, which is the subject of the score lab.

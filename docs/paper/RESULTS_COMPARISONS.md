# Comparisons with similar systems: accuracy, speed, cost

Branch `score-lab`. Plan and registrations: `lab/NOTES.md` entries 20, 23 and 24. In result rows the configuration is written "Qwen3-VL-4B + Glance"; Glance is a readout and calibration recipe, not a model. Every comparison below uses identical images and labels for all systems in the same table. Sections marked pending fill in as data lands.

## 1. A frozen dual encoder against the VLM readouts (lab scales, identical items)

300 calibration labels and 300 test images per scale, the same for every row. SigLIP2 answers in 38 ms per image.

| System | blur | exposure | jpeg | noise | resolution | mean accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| SigLIP2 base, 256 px, one statement per level (CLIP-IQA-style frozen encoder), as shipped | 0.313 | 0.387 | 0.240 | 0.267 | 0.443 | **0.330** |
| SigLIP2 + the same matrix calibration | 0.627 | 0.487 | 0.647 | 0.543 | 0.637 | **0.588** |
| Qwen3-VL-4B, yes/no per level (v0 / P(True)-style), as shipped | 0.617 | 0.520 | 0.350 | 0.553 | 0.483 | **0.505** |
| Qwen3-VL-4B, yes/no per level + matrix calibration | 0.863 | 0.910 | 0.667 | 0.820 | 0.770 | **0.806** |
| Qwen3-VL-4B + Glance `digits` (1 pass) | 0.857 | 0.887 | 0.680 | 0.837 | 0.833 | **0.819** |
| Qwen3-VL-4B + Glance `fast2` (2 passes) | 0.887 | 0.903 | 0.687 | 0.853 | 0.830 | **0.832** |
| Qwen3-VL-4B + Glance `ens4d` (4 passes) | 0.887 | 0.930 | 0.757 | 0.867 | 0.877 | **0.863** |

Reading: a fitted map on the readout helps a contrastive encoder as well (0.330 to 0.588), so the effect is not specific to the VLM; and the VLM perceives much more of these attributes than the dual encoder does at 256 px.

## 2. Other systems' readouts on the same frozen VLM (option letters, rotated letters, two poles)

Pending: registered in `lab/NOTES.md` entry 24, queued on the GPU.

## 3. Classical no-reference image-quality features with the same labels

29 hand-built no-reference features (sharpness, high-frequency energy, noise estimate, 8-px blockiness, luminance and colour statistics, edge density, ...) with a standardized multinomial logistic regression, fit on the same calibration labels and scored once on the same test items (`tools/classical_baselines.py`, `results/lab/classical_baselines.md`; feature extraction about 13 ms per image on the CPU).

| Lab scales (5 scales, 4 levels, 500 test images each) | about 32 labels per scale | all 500 labels per scale |
| --- | --- | --- |
| Classical features + logistic regression | 0.752 | **0.979** |
| Qwen3-VL-4B + Glance `ens4d` | **0.856** | 0.867 |

Reading, stated plainly: with enough labels, features built for exactly these artifacts beat the VLM on every lab scale, by 6 to 22 points. The VLM readout is the more label-efficient one (ahead by about 10 points at 32 labels) and it saturates early: more labels do not help it. So the lab scales do not show that a VLM is the best tool for blur or JPEG grading; they show how much of a described rubric a frozen general model can deliver from a few dozen labels, with no feature engineering. On the 25-distortion benchmarks the classical baseline reaches 0.651 (`distort25`) and 0.620 (KADID-10k; Spearman with human scores 0.629) with all calibration labels and 0.443 / 0.401 with about 30; it is near chance where the distortion is structural rather than statistical (shuffled patches, colour diffusion). The VLM's numbers on those benchmarks are in `RESULTS_GENERALIZATION.md` when that collection finishes.

## 4. Frontier APIs on the same held-out images

Frontier: zero-shot, constrained to the rubric's levels, uncalibrated (a hard pick has no probabilities to calibrate); only whether each pick was right is stored. Local rows are recomputed from the lab's saved logits for the same item ids. The temperature of the v0 row was fit on the calibration split (it cannot change a prediction).

| System | labels used for this rubric | blur | exposure | jpeg | noise | resolution | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic/claude-opus-5 | 0 | 0.560 | 0.750 | 0.285 | 0.525 | 0.630 | **0.550** |
| Qwen3-VL-4B, v0 readout as shipped (single temperature) | 0 | 0.625 | 0.530 | 0.365 | 0.550 | 0.480 | **0.510** |
| Qwen3-VL-4B, + Glance ens4d | 0 | 0.470 | 0.700 | 0.450 | 0.575 | 0.655 | **0.570** |
| Qwen3-VL-4B, + Glance ens4d | 0 (plus unlabeled images) | 0.610 | 0.840 | 0.565 | 0.740 | 0.755 | **0.702** |
| Qwen3-VL-4B, + Glance ens4d | 32 | 0.906 | 0.921 | 0.738 | 0.850 | 0.871 | **0.857** |
| Qwen3-VL-4B, + Glance ens4d | 500 | 0.910 | 0.935 | 0.745 | 0.875 | 0.875 | **0.868** |

Paired differences in mean accuracy, same images (bootstrap 95% interval over items):

| Difference | points | 95% interval |
| --- | --- | --- |
| v0 readout as shipped (single temperature), 0 rubric labels minus anthropic/claude-opus-5 | -4.0 | [-8.0, +0.0] |
| + Glance ens4d, 0 labels minus anthropic/claude-opus-5 | +2.0 | [-2.1, +6.0] |
| + Glance ens4d, 0 labels + unlabeled images minus anthropic/claude-opus-5 | +15.2 | [+10.8, +19.5] |
| + Glance ens4d, 32 labels minus anthropic/claude-opus-5 | +30.7 | [+27.3, +34.2] |
| + Glance ens4d, 500 labels minus anthropic/claude-opus-5 | +31.8 | [+28.2, +35.4] |

n per scale: blur 200, exposure 200, jpeg 200, noise 200, resolution 200.

Caveats that travel with this table: the frontier model is zero-shot and cannot be calibrated through a hard pick; a few-shot prompt for it was not tried; the scales are synthetic single-factor rubrics with hand-written level texts; with hundreds of labels classical features beat both systems (section 3). The paid calls were run by the project owner with their own key.

## 5. A second model family (SmolVLM2-2.2B), same prompts and recipe

Pending: registered in `lab/NOTES.md` entries 20, 22 and 22b, queued on the GPU.

## 6. Open image-capable typed-decision systems and trained image-quality VLMs, as shipped

Pending: license and feasibility survey in `docs/paper/COMPARABLE_SYSTEMS.md`. Only Apache-2.0 or MIT weights may be run here; the rest are cited with their published numbers and the caveat that blind MOS regression is a different task from level classification with the distortion named.

## 6b. Write the answers or read them? (same frozen VLM, same images, same questions)

`glance/lab/gen_bench.py`: cold start per image, end to end, idle GPU, 40 lab test images. Writing = greedy generation of one JSON object with a token cap (no thinking). Reading = `glance decide`, uncalibrated, which also returns a probability for every answer.

| Request | write p50 ms (tokens written) | read `fast2` p50 ms | read `ens4d` p50 ms | reading is faster by | JSON failures | written = read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 yes/no | 798 (7) | 338 | - | 2.4x | 0 of 40 | 100.0% |
| 5 mixed | 3478 (38) | 1001 | 2230 | 3.5x / 1.6x | 0 of 40 | 92.0% |
| 25 ratings | 20046 (221) | 3261 | 8380 | 6.1x / 2.4x | 0 of 40 | 59.0% |

The single yes/no was written as a small JSON object (7 tokens), not a bare word; a one-token answer would narrow that gap and was not measured. On 25 ratings the written and the read answers differ on four fields in ten; neither is calibrated and there is no ground truth for that request, so this is a difference, not a ranking.

## 7. Cost of 1,000 ratings

Seconds are measured (`lab/PACKING.json`); dollar figures are arithmetic on stated assumptions (`tools/cost_model.py`), not measurements. API figures are list-price upper estimates from `glance baseline --estimate-only`; no call was made.

| Qwen3-VL-4B + Glance, self-hosted | seconds per 1,000 ratings | electricity only | rented cloud GPU at laptop speed | laptop amortized |
| --- | --- | --- | --- | --- |
| ens4d, one rubric per image | 1089 | $0.0005 to $0.0073 | $0.159 to $0.243 | $0.023 to $0.138 |
| ens4d, 5 rubrics per image | 584 | $0.0002 to $0.0039 | $0.085 to $0.130 | $0.012 to $0.074 |
| ens4d, 25 rubrics per image | 341 | $0.0001 to $0.0023 | $0.050 to $0.076 | $0.007 to $0.043 |
| fast2 (no crop), 5 rubrics per image | 240 | $0.0001 to $0.0016 | $0.035 to $0.054 | $0.005 to $0.030 |
| fast2 (no crop), 25 rubrics per image | 133 | $0.0001 to $0.0009 | $0.019 to $0.030 | $0.003 to $0.017 |

| Hosted frontier API (one image per call, no labels) | list-price upper estimate per 1,000 ratings |
| --- | --- |
| anthropic/claude-opus-5 | $18.70 |
| openai/gpt-5.6 | $14.96 |
| gemini/gemini-3.1-pro-preview | $8.28 |
| anthropic/claude-haiku-4-5-20251001 | $3.74 |

For one rubric per image on a rented GPU the APIs are 15 to 118 times more expensive per rating. Not counted: engineering time and the one-time labeling of about 32 images per rubric (2 to 5 minutes). The APIs need no labels and no setup. Assumptions: laptop draw 15 to 60 W (not measured), electricity $0.10 to $0.40 per kWh, cloud GPU $0.526 to $0.8048 per hour (AWS on-demand g4dn.xlarge and g6.xlarge, us-east-1, checked 2026-09-20) at the laptop's own speed, laptop price $2000 to $4000 amortized over 8760 to 26280 hours. TypeSafe's Jev is not in the table: it does not accept images.


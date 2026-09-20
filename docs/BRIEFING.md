# Glance: briefing for outside reviewers (human or AI)

Status date: 2026-09-20. This file is self-contained: you do not need the repository to comment on it. Everything
below is measured unless it says "planned" or "partial". Numbers trace to committed files (named in brackets).
We want criticism more than praise; the questions we most want answered are in section 9.

## 1. One paragraph

Glance is a way to ask a frozen vision-language model for an ordinal score. It reads the digit-token logits of a few
complementary prompts (no text is generated), can pack many rubrics behind one image encode, and maps the logits to
levels with a small per-rubric affine calibration fit on a few dozen labeled images. On Qwen3-VL-4B this moves the
original yes/no readout from 0.500 exact accuracy (single temperature) to 0.867 on five synthetic 4-level image-quality
scales, with the VLM's weights frozen. (A small readout IS fit on labeled images: 500 per rubric in that result, about 32 for most of the gain. It is not zero-shot.) `glance fit` is the product verb. **Glance is how you ask a frozen VLM
for a score. It is not a VLM**, and we do not claim it is faster or cheaper than any hosted model.

## 2. Where it came from

A hand-off spec asked for a local harness with the same interface as TypeSafe's Jev (typed questions `noul` = P(yes),
`choice` = pick one, `score` = ordered rubric; distributions out, no generation), but for images, to test whether a
"v1" is a calibration problem or a data problem. v0 was built in a day (milestones M0 to M5, 236 tests now). Its
weakest result was `score`. A pre-registered "score lab" then attacked that one question type. Everything since is on
one laptop (Apple M5, 32 GB, MPS, float16), one model (Qwen/Qwen3-VL-4B-Instruct, Apache-2.0, pinned revision), and
license-clean data.

## 3. Method, precisely

**Readout.** The prompt shows the question and a numbered scale and ends at the empty assistant turn. We read the
next-token logits over the digits `0..K-1` from the final hidden state with a float32 head (float16 quantizes logits
near 20 to steps of 1/64). Four members, one forward pass each:

| Member | Scale order shown | Images |
| --- | --- | --- |
| `digits` | lowest to highest | the image |
| `digitsrev` | highest to lowest (logits flipped back) | the image |
| `zoom_digits` | lowest to highest | the image + `zoom`: the central third, enlarged 3x with nearest-neighbour sampling |
| `zoom_digitsrev` | highest to lowest | image + `zoom` |

`ens4d` = all four. Reversal makes position and digit biases enter with opposite signs. The magnified crop makes
grain and 8 x 8 compression blocks large enough to see at a 448 px / 196-token image.

**Calibration.** The 4K logits are standardized and mapped by `p = softmax(s (W x + b))`, `W` is K x 4K: matrix
scaling, L2 = 0.05, then one scalar `s` refit without the penalty (it changes no prediction). Without a calibration
the answer is softmax of the mean member logits. A calibration is keyed to the exact rubric text, the model revision
and the image-token budget. There is no pooled fallback, because we measured that a calibration fit on one rating
dimension does not work on another.

**Selection discipline.** Every design choice (readouts, ensembles, calibration kind) was made on a calibration split
by 5-fold cross-validated NLL; the winners were committed to the notebook before the test split was scored, once.
Hypotheses were registered before each experiment; one of ours (threshold questions beat isolated level questions)
was not supported.

## 4. Main result: five synthetic 4-level scales, 500 held-out images each [lab/REPORT.json, lab/BOOTSTRAP.json]

Blur, noise, JPEG artifacts, underexposure, low resolution, generated on Oxford-IIIT Pet photos (CC BY-SA), level
texts written by hand. Mean exact accuracy on the test split, bootstrap 95% intervals:

| Configuration (all Qwen3-VL-4B, frozen) | Forward passes | Mean accuracy |
| --- | --- | --- |
| v0 readout (one yes/no statement per level), single temperature, as shipped | 4 | 0.500 [0.481, 0.519] |
| same readout, best cross-validated calibration (bias + temperature) | 4 | 0.810 [0.796, 0.825] |
| `digits`, one pass, matrix calibration | 1 | 0.814 [0.800, 0.829] |
| `zoom_digits`, one pass | 1 | 0.838 [0.822, 0.852] |
| **+ Glance (`ens4d`)** | 4 | **0.867 [0.854, 0.881]** |
| `ens7`, all seven readouts we tried | 14 | 0.876 [0.864, 0.889] |

Per scale for `ens4d`: blur 0.888, exposure 0.924, JPEG 0.772, noise 0.864, resolution 0.888; ECE 0.018 to 0.048;
mean absolute error 0.179 levels; almost every error is an adjacent level. A bar fixed in advance (accuracy >= 0.85,
MAE <= 0.25, ECE <= 0.05, on at least 4 of 5 scales) is met on 4 of 5; **JPEG misses** for every method. Paired
differences: `ens4d` minus best-calibrated v0 readout +5.7 points [4.4, 7.1]; minus `zoom_digits` +3.0 [1.8, 4.1].
About 32 labels per rubric bring `ens4d` within two points of its final accuracy. On the uncached reference path the
prediction is the same on 499 of 500 images. For context only: a frontier model (Claude Opus 5, uncalibrated,
constrained output) scored 0.536 on v0's comparable blur suite.

## 5. Things we got wrong and fixed in public

1. **Latency erratum** [lab/NOTES.md entry 16, lab/PACKING.json]. We first reported `ens4d` at 609 ms next to 432 ms
   for v0 and wrote "at the cost of the baseline". That 609 ms was the marginal cost of four readouts on an image
   whose prefix was already cached; it contained no image prefill, the v0 number contained one. Cold start, end to
   end, idle GPU, p50:

   | | one image |  per rating |
   | --- | --- | --- |
   | v0 readout, prefix cache | 441 ms | 441 ms |
   | v0 readout as shipped (no cache) | 918 ms | 918 ms |
   | `ens4d`, one readout at a time | 1,248 ms | 1,248 ms |
   | `ens4d` packed (2 prefills, 2 batched reads) | 1,089 ms | 1,089 ms |
   | `ens4d` packed, 5 rubrics in one request | 2,918 ms | 584 ms |
   | `ens4d` packed, 25 rubrics | 8,529 ms | 341 ms |
   | `digits` + `digitsrev` packed, no crop, 25 rubrics | 3,333 ms | 133 ms |

   So `ens4d` is 2.5 to 2.8 times slower than the cached v0 readout for a single score, and comparable per score only
   when several rubrics share an image. Packing changed 0 of 100 predictions. The saving is modest because a
   196-token image is short next to a 100-token rating prompt that must be read once per readout.
2. **Small fits were overconfident** [lab/NOTES.md entry 18]. The first real `glance fit` (40 labels) showed it. Cause:
   the sharpness scalar `s` was fit on the training points, which are nearly separable at that size. Fitting `s` on
   held-out folds: NLL 1.320 -> 0.601 at 16 labels, 0.749 -> 0.478 at 32, equal from about 128; accuracy identical
   by construction. "32 labels are enough" was a statement about accuracy, not about probabilities.
3. **The v0 "yes/no calibration failure" was mostly sample size.** Equal-mass ECE has a sampling floor (what a
   perfectly calibrated predictor with the same confidences would measure). At n = 1,000 the gate is met on one suite
   (0.029) and sits at its floor on the other. We now print that floor next to every ECE.
4. **A badly designed pre-registered criterion** [entry 15b]: "ECE <= 0.05 on 20 of 25 distortions" cannot be met at
   200 test items per distortion (floor about 0.08). Recorded before the results; a pooled ECE was added.

## 6. The harness [glance/rating.py, glance/fit.py, results/lab/harness_rating_check.json]

`glance decide` (CLI, local HTTP server, Python class) answers `noul`, `choice`, `score`. For `score` the default on
the VLM is now `ens4d`; all rating questions of a request are packed into one backend call per view. Additive to the
Jev-shaped contract: `options.score_method`, `options.calibrated: "auto"`, and `method` / `calibration` on score
answers. `glance fit --data labels/ --rubric rubric.json` reads four logit vectors per labeled image, fits the map,
reports cross-validated accuracy, MAE, ECE and the ECE floor at the user's sample size, and writes a 4 KB JSON.
Acceptance check: ordinary requests with the shipped calibrations reproduce the lab's own predictions on 200 of 200
held-out images, also when five rubrics go in one request. v0 context for the other two types: yes/no 0.884 (POPE)
and 0.744 (GQA subset), 37-way breeds 0.892, 101-way objects 0.919; 3.1 points behind Claude Opus 5 on average;
`independent` multiple choice is exactly permutation-invariant by construction.

## 7. The adult test, in progress: 25 distortions, 5 levels, human scores [lab/NOTES.md entries 14 to 15c]

Registered before any data: the method fixed as selected above, **one generic question for all 25 distortions**
("How strong is the <one-line description> in `img0`?", levels Barely noticeable / Slight / Moderate / Strong / Very
strong), no re-selection, baselines with their best calibration, and hypotheses with numeric targets. Two benchmarks:
a license-clean rebuild on new photos (`distort25`, 7,500 items) and KADID-10k itself (evaluation only, split by
reference image, nothing from the database redistributed).

**Partial, not a result** (6 of 25 KADID distortions, 17 calibration references each): exact accuracy 0.537, within one
level 0.880, MAE 0.597, per-distortion Spearman correlation with human DMOS 0.805 against 0.866 for the true level
itself; single-pass readouts 0.465 and 0.449 on the same items. Our registered targets (exact >= 0.70, within one
>= 0.97, MAE <= 0.40) will probably **fail** on KADID; the ensemble gain (+7 to +9 points) and the rank correlation
look like the parts that survive. The correlation numbers are type-aware (the question names the distortion) and
are not comparable with blind IQA results. Full results replace these when collection ends.

## 7b. Update, 2026-09-20 midday: comparisons run since the first version of this briefing

All on identical images and labels per table (`docs/paper/RESULTS_COMPARISONS.md`, `lab/NOTES.md` entries 19 to 25).

1. **Classical features beat us when labels are plentiful.** 29 hand-built no-reference features (sharpness, noise
   estimate, 8-px blockiness, luminance and colour statistics) + logistic regression, same labels, same test images:
   lab scales 0.979 mean accuracy with all 500 labels per scale, above `ens4d` (0.867) on every scale (JPEG 0.990
   against 0.772). With 32 labels it is 0.752 against 0.856 for `ens4d`: the VLM readout is label-efficient and
   saturates early. On the 25-distortion sets the classical baseline scores 0.651 (`distort25`) and 0.620 (KADID-10k,
   Spearman with human scores 0.629), near chance on structural distortions (shuffled patches, colour diffusion). So the
   lab scales do NOT show a VLM is the best tool for blur or JPEG grading; they show what a frozen general model
   delivers from a rubric in words and a few dozen labels, with no feature engineering.
2. **The "remap the readout" effect is not VLM-specific.** SigLIP2 (frozen dual encoder, 38 ms per image): 0.330 as
   shipped (chance 0.250), 0.588 with the same matrix map. Same items: v0 VLM readout 0.505 -> 0.806, `ens4d` 0.863.
3. **Two calibration challengers reviewers asked for lose or tie** (calibration split only): shrinking toward the
   uncalibrated answer is worse at every label count; a 19-parameter ordinal (cumulative-link) readout ties the
   68-parameter matrix up to 64 labels (NLL 0.403 against 0.423 at 32 labels) and is slightly behind at 240. The
   evidence in the member logits is essentially one-dimensional.
4. **`fast2`** (digits + reversed digits, no crop; 133 to 240 ms per rating at 5 to 25 rubrics per image): 0.833 on the
   held-out split, JPEG 0.674 (so the magnified crop is what helps compression artifacts: 0.772 with it).
5. **Cost** (`results/lab/cost_model.md`; seconds measured, dollars are arithmetic on stated assumptions): 1,000
   ratings cost $0.159 to $0.243 on a rented cloud GPU no faster than the laptop ($0.019 to $0.030 with `fast2` at 25
   rubrics per image), against list-price upper estimates of $3.74 to $18.70 for frontier APIs. Electricity alone is
   under a cent.
6. **Running now or queued:** a fitted readout on the VLM's final hidden states (is the 0.867 ceiling the token
   interface or the 196-token image?); other systems' readouts on our model (option letters, rotated letters, two
   poles); a second model family (SmolVLM2-2.2B, Apache-2.0); two runnable external systems (`openjev` v2, `q-sit-mini`);
   a non-quality rubric benchmark with exact ground truth (cut-off, occlusion, tilt, caption legibility, watermark);
   frontier APIs on the same images (needs the owner's key). Most trained IQA VLMs (Q-Align, DeQA-Score, ...) sit on
   LLaMA-2-derived bases and cannot be run under our Apache/MIT-weights rule; they are cited, not run
   (`docs/paper/COMPARABLE_SYSTEMS.md`).

New question for reviewers: given item 1, is the right headline "label-efficient, rubric-in-words grading from a frozen
general model" with classical features as the honest ceiling for low-level artifacts, and should the paper's main
benchmark move to rubrics where no classical feature exists (framing, occlusion, legibility)?

## 8. What we claim and what we do not

Claim: on a frozen 4B VLM, how you ask and how you remap the logits is worth +37 points on described rating scales
(+6 over the best calibration of the naive readout); temperature scaling is not enough for ordinal scales; the
calibration is rubric-specific; small-n probability quality needs an out-of-sample sharpness estimate; packing
rubrics behind one image prefill is prediction-safe.

Not claimed: a new model; "Jev for vision" (same interface shape, different object: a recipe on someone else's
weights, calibration is the user's job, slower than Jev's advertised latency for a single score); faster or cheaper
than anything hosted (no dollar numbers); "works on any VLM" (one model measured; `fit` is the transfer mechanism, not
a transfer result); "0.87 on image quality assessment" (five synthetic single-factor scales with hand-written level
texts; KADID is harder). Naming: result rows read "Qwen3-VL-4B + Glance", never a model-style name.

Known limits: task-specific classical features beat the method on the lab scales when hundreds of labels are available (section 7b); one model, one machine, synthetic degradations for the headline, level texts tuned by eye on the
calibration split, no comparison with trained IQA scorers (Q-Align, DeQA-Score) or classical no-reference metrics
(which would solve blur and noise ladders trivially), no second VLM family yet, prefix-cached path fails the spec's
strict logit-equality acceptance (|dz| 0.075 > 0.05) although it changes no decision, DistortBench could not be run
(no public data link found).

## 9. Questions for you

1. **Is the core finding already known?** Closest work we verified: VQAScore (P(yes) readout), Kadavath et al.
   P(True), Zheng et al. on selection bias / PriDe, Q-Bench, Q-Align and DeQA-Score (trained level-token scorers),
   DistortBench, ViCrop. We have not found "digit-logit ensemble + per-rubric matrix scaling on a frozen VLM,
   pre-registered". What are we missing?
2. **KADID follow-up.** If exact accuracy lands in the mid 50s with generic wording: is the reference-anchored
   question (pristine image beside the distorted one) the right next experiment, or per-distortion level texts, or
   more crops, or a sequence readout (sum of log-probabilities of a short phrase per level)? Which would you register?
3. **Is type-aware rank correlation worth reporting at all**, given that blind IQA is the community's metric? What is
   the fairest comparison a frozen, untrained 4B model can be put through?
4. **Second model.** Which open VLM is the most informative replication: same family at 2B / 8B, or a different
   family (which)? What result would change your mind about the method?
5. **Calibration.** Is matrix scaling with a held-out sharpness scalar the right small-n recipe, or should we use an
   ordinal model (cumulative link on a learned 1-D projection), Dirichlet calibration, or a Bayesian shrinkage toward
   the uncalibrated mean-logit answer?
6. **JPEG.** Every readout misses on compression artifacts. Perception limit at 196 image tokens, or still an
   elicitation problem? (Five crops instead of one lifted cross-validated JPEG accuracy 0.798 -> 0.840 in a pilot.)
7. **Cost.** The question text, not the image, dominates packed cost. Is a tree-shaped KV cache (image -> question ->
   two level orders) worth building, or is dropping the crop for cheap rubrics (`digits` + `digitsrev`, 133 to 240 ms
   per score) the better product default?
8. **Positioning.** We plan to lead with the elicitation gap and the two errata, publish the KADID miss as hard as
   the lab win, and describe the interface as a "Jev-like Score readout for VLMs" in related work only. What would
   make you distrust this write-up?

## 10. If you have the repository

`README.md` (quick start), `STATUS.md` (decisions D1 to D36), `lab/NOTES.md` (notebook with every registration and
erratum), `docs/paper/` (METHODS, RESULTS_V0, RESULTS_LAB, RELATED_WORK, REPRODUCE, OUTLINE, figures),
`tools/verify_docs_numbers.py` (every 3-decimal number in the docs must appear in a committed result file).

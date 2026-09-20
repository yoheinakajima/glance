# Claims ledger: what we can say, what it rests on, and what we must not say

Status date 2026-09-20. One row per claim. "Supported" means measured on held-out data with the evidence file named;
"with caveats" means the sentence is only true together with the caveat next to it; "pending" means the experiment is
registered or running and the claim must not be made yet. Model everywhere: Qwen3-VL-4B-Instruct, frozen, on one Apple
M5 laptop. If a number here disagrees with a results file, the results file wins and this ledger is wrong.

## A. General use out of the box (yes/no and pick-one questions, zero labels, zero setup)

| Claim | Status | Evidence | Required caveats |
| --- | --- | --- | --- |
| Any yes/no or pick-one question about an image gets a typed answer with probabilities from one forward pass per statement, no generation, no per-task setup. | Supported (it is how the harness works) | `glance/scorer.py`, v0 tests, `HANDOFF.md` section 5 | None for the interface. Accuracy claims are the rows below. |
| Yes/no accuracy out of the box: POPE 0.880 [0.840, 0.920], GQA yes/no 0.732 [0.676, 0.784] (n = 250 each, uncalibrated decisions). | Supported | `results/v0/analysis/bootstrap_v0.json` | Two public benchmarks on everyday photos. Both are old and public, so the model (and the frontier model) may have seen them in training. We have no uncontaminated check: `gold/human_gold.jsonl` is still empty. |
| Pick-one accuracy out of the box: 37 pet breeds 0.892 [0.852, 0.928], 101 object classes 0.919 [0.882, 0.955]; with the option-letter method on a 26-option subset 0.904 and 0.977. | Supported | same file; `results/v0/m5_full_eval/metrics.json` | Same contamination caveat. `independent` costs one forward pass PER OPTION: p50 8.4 s per image for 37 options and 12.3 s for 101 on the default (uncached) path, against 1.7 s and 1.0 s for `letter`, which is capped at 26 options and is not order-invariant. A plain frozen SigLIP2 dual encoder beats the VLM on the pets suite (0.956). |
| A frontier model (Claude Opus 5, zero-shot, constrained picks) is ahead by 3.4 points on average [1.0, 5.8] over five suites; 3.2 [1.0, 5.4] on the four yes/no and choice suites; tied on GQA, 5.0 ahead on Caltech-101. | Supported | `results/v0/analysis/bootstrap_v0.json` (paired bootstrap, same items) | One frontier model, one run, 221 to 250 items per suite. The report's own figure, 3.1 points, uses calibrated local decisions; both are correct for what they measure. |
| The `independent` pick-one method is exactly invariant to option order. | Supported | `results/v0/m5_full_eval/report.md` (max probability shift 0.0e+00) | By construction (canonical statement order), verified empirically. `letter` is not invariant; it averages rotations. |
| Probabilities out of the box are calibrated. | NOT supported | `results/v0/analysis/noul_n1000/` | Raw yes/no probabilities are overconfident (ECE 0.111 and 0.179, NLL above 1). A pooled two-number Platt map fit once on labeled data brings ECE to 0.061 and 0.053 (at the sampling floor on GQA; 0.029 on POPE with an isotonic map at n = 1,000). That map was fit on these same two suites; transfer to a new yes/no task is untested. |
| It knows when none of the options fits. | NOT supported as shipped | `STATUS.md` stretch section | Held-out breeds land on an explicit `other` option only 7.7% of the time. The raw logits do carry the signal (AUROC 0.977 for the negative max logit); no threshold is shipped. |
| It is robust to text written on the image. | NOT supported | `results/v0/stretch/injection/report.md` | "Answer Yes" rendered onto the image flips 5.2% of correct POPE negatives. |

## B. Ratings on an ordered rubric (`score`)

| Claim | Status | Evidence | Required caveats |
| --- | --- | --- | --- |
| Out of the box a rating is a good ranking and a weak grade: 0.558 exact, 0.987 within one level (five lab scales). | Supported | `results/lab/label_free_test.json` | Five synthetic single-factor scales with hand-written level texts. |
| With 16 or more UNLABELED images of the domain (`glance fit --unlabeled`): 0.697 exact. | Supported, with caveats | same file; `lab/NOTES.md` entry 26b | Registered target was 0.70: narrowly missed. Falls to 0.646 if the unlabeled pool is 70% one level. Cannot check itself. Expected-value error does not improve; use the top level or the rank. |
| With about 32 labeled images per rubric (`glance fit`): 0.856; with 500: 0.867 [0.854, 0.881]. | Supported | `lab/EXTRAS_ens4d.json`, `lab/BOOTSTRAP.json` | Not zero-shot. The 500-label figure is the headline; say "synthetic single-factor scales" in the same sentence. JPEG artifacts miss the pre-registered bar (0.772). |
| The gain decomposes: shipped readout 0.500 -> same readout, properly calibrated 0.810 -> digit readout 0.814 -> with magnified crop 0.838 -> four-readout ensemble 0.867. | Supported | `docs/paper/RESULTS_LAB.md` sections 2, 3, 8 | Most of the +36.8 is "the shipped single-temperature calibration was badly specified". The method's own effect over the best-calibrated naive readout is +5.7 points [4.4, 7.1]. |
| A calibration fit on one rubric does not work on another. | Supported | `RESULTS_LAB.md` section 5 | Shown between the five lab scales only. A universal map across many rubrics is being tested (entry 26, pending). |
| Task-specific classical features beat the method when labels are plentiful. | Supported (against us) | `results/lab/classical_baselines.json` | 0.979 against 0.867 with 500 labels per scale; 0.752 against 0.856 with 32. Say this whenever the lab scales are quoted. |
| The "remap the readout" effect is not specific to the VLM. | Supported | `results/lab/dual_encoder_vs_vlm_same_items.json` | SigLIP2 0.330 -> 0.588 with the same map, far below the VLM. |
| It generalizes to 25 distortion types, 5 levels, generic wording, and tracks human opinion scores. | PENDING | `lab/NOTES.md` entries 14 to 15c | Interim look says exact accuracy will be well below the lab scales and below our registered target; do not quote interim numbers as results. |
| It works on another model family / on non-quality rubrics / against other systems' readouts / against frontier APIs on the same images / from hidden states. | PENDING (all five) | entries 20, 22, 24 | Registered with expectations that can fail. |

## C. Calibration findings (these are safe to share as lessons)

| Claim | Status | Evidence |
| --- | --- | --- |
| A single temperature cannot fix an ordinal readout: it never changes a prediction, and the shipped readout's errors are boundary errors. | Supported | `RESULTS_LAB.md` section 3 |
| Small fits are overconfident unless the sharpness scalar is estimated on held-out folds (NLL 0.749 -> 0.478 at 32 labels); accuracy is unaffected. "32 labels are enough" is a statement about accuracy, not about probabilities. | Supported (dev experiment on the calibration split) | `lab/NOTES.md` entry 18 |
| Equal-mass ECE has a sampling floor; several of our own "failures" and one of our own pre-registered criteria were artifacts of it. Every ECE should be printed next to its floor. | Supported | `glance/evals/metrics.py::ece_noise_floor`; entries 9, 15b |
| A 19-parameter ordinal map ties the 68-parameter matrix up to 64 labels; shrinking toward the uncalibrated answer is worse. | Supported (calibration split only; registration and result share a commit) | `lab/dev/calibration_challengers.md`, entry 19b |

## D. Speed and cost

| Claim | Status | Evidence | Required caveats |
| --- | --- | --- | --- |
| One rating of a fresh image: 1,089 ms with `ens4d` (441 ms for the naive readout). Per rating when rubrics share an image: 584 ms at 5, 341 ms at 25; `fast2` 240 and 133 ms. | Supported | `lab/PACKING.json` | One laptop. `ens4d` is SLOWER than the naive readout for a single rating. An earlier 609 ms figure was wrong (entry 16). |
| Packing rubrics behind a shared image prefill changes no prediction. | Supported | `lab/REFCHECK_packed.json` (100 of 100), `results/lab/harness_rating_check.json` (200 of 200) | Our packing is independent sequences sharing a cached prefix, not several questions in one sequence. |
| Self-hosted cost per 1,000 ratings is $0.019 to $0.243 on a rented GPU at laptop speed; frontier APIs are $3.74 to $18.70 at list price. | With caveats | `results/lab/cost_model.json` | Dollars are arithmetic on stated assumptions (cloud prices checked 2026-09-20; laptop power NOT measured); API figures are upper estimates, no call was made. Labels and engineering time are not counted. |
| Faster or cheaper than Jev. | Do not claim | | Jev does not accept images; no measurement exists. |

## E. Process claims

| Claim | Status | What git proves |
| --- | --- | --- |
| Winners were chosen on a calibration split and committed before the test split was scored once. | Supported | `lab/NOTES.md` entry 12 commit precedes the test scoring commit. |
| Hypotheses were registered before data. | Supported for entries 14, 15, 15b, 20, 22, 24, 26 (own commits before the data existed); NOT provable for entries 19 and 21 (registration and result share a commit) | `lab/NOTES.md`, timestamp note |
| We publish our misses. | Supported | Latency erratum (16), small-fit overconfidence (18), flawed ECE criterion (15b), failed hypothesis on threshold questions (13), H19 narrow miss (26b), classical features beating us (25), wrong notebook timestamps (timestamp note). |
| Every number in the docs is machine-checked. | With caveats | `tools/verify_docs_numbers.py` checks that each 3-decimal number appears in SOME committed source; with tens of thousands of source numbers a wrong number can match by coincidence. Generated result documents do not have this weakness. |

## F. Do not say

- "A new model", "Jev for vision" as a headline, any model-style name. Glance is a readout and calibration recipe with a runtime.
- "Zero-shot" or "no training" for any number that used a fitted readout. The VLM is frozen; the readout is fit on labels.
- "0.87 on image quality assessment." It is five synthetic single-factor scales.
- "Works on any VLM." One model measured; the second is running.
- "Calibrated out of the box." Decisions are usable out of the box; probabilities need a fitted map.
- "Better than classical methods." On low-level artifacts with enough labels they are better.
- Interim KADID numbers as results.

## G. What we learned that is worth sharing, in order of how sure we are

1. How a frozen VLM is asked and how its logits are remapped changes graded-judgment accuracy by tens of points; the
   remapping matters more than the choice of answer tokens (pending the letter-readout comparison for the second half).
2. Temperature scaling is the wrong tool for ordinal readouts; an affine map per rubric is the right one and it does not
   transfer between rubrics.
3. Few-label fits need out-of-sample sharpness; ECE needs its sampling floor printed next to it.
4. About half of the zero-label error on ratings is removable bias that unlabeled images reveal.
5. Sharing an image prefill across independent questions is prediction-safe; with short images the question text, not
   the image, dominates cost.
6. Cold-start and warm-cache latencies are different quantities; we mixed them once.
7. For low-level artifacts, thirty lines of classical features and enough labels beat a 4B VLM; the VLM's advantages are
   label efficiency, rubrics in words, and (to be shown) attributes no classical feature captures.
8. Most "MIT / Apache" trained image-quality VLMs inherit LLaMA-2 terms from their base model; check the base, not the tag.

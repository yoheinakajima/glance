<!-- FIRST DRAFT, written by an assistant model from the repository's result files on 2026-09-20 and only partly reviewed.
     Known issues are listed at the end ("Draft notes for the editor"). Speed and cost statements were corrected after
     the fair photo-size timing (lab/NOTES.md entry 48). Not yet covered: entries 38c (8B verdicts are in; check section 5),
     46 to 48b, and the harder insect-order test (entry 47). -->

# Reading Typed Decisions from a Frozen Open Vision-Language Model

## Abstract

A typed question is one whose legal answers form a closed set known before the model runs: yes or no, one of a list, a level
on a rubric. We put such questions about images to a frozen open vision-language model (Qwen3-VL-4B-Instruct, Apache-2.0, on
a laptop) and read the answer from the logits of one forward pass; nothing is generated. On photographs taken after every
model's release, labelled by people outside this project, the open model is statistically indistinguishable from six hosted
models (three flagships, three low-cost) on coarse yes/no and pick-one questions (n = 131 and 65; every 95% interval overlaps
every other). The questions are easy and the labels imperfect: 3 to 5% of items are answered "wrongly" by all seven systems.
Reading gives the same answer as the same model writing JSON. Ratings behave differently. Zero-shot, every system orders
images correctly (within one level on 0.99 of images) and places the level boundaries wrongly, by a constant offset per
rubric; a low-cost hosted model leads (0.763 against 0.669), and an 8B model is no better than a 4B one. Because a read
answer is a vector of logits it can be fitted: 16 unlabeled images of the rubric remove most of the offset (0.758) and 32
labels reach 0.857. The hosted models were not given examples, so this is a comparison of products, not of models. On a real
image-quality benchmark (KADID-10k) the approach missed every target we registered, and hand-built features remain better for
low-level artefacts. The open model's cost is of the same order as the low-cost hosted models and one to two orders below the
flagships; no image leaves the machine. The readout is shared with other training-free tools and is not claimed as new; what
is offered is the measurement, registered before it was run, with its misses.
<!-- src: results/lab/matrix.md, results/lab/label_noise_proxy.md, results/lab/jsondigits.md, results/lab/scaling.md, docs/paper/RESULTS_GENERALIZATION.md; wording aligned with the page after the outside critique of 2026-09-21 (lab/NOTES.md entry 51) -->

## 1. Introduction

The question behind this paper is plain: what does a frozen open vision-language model already do for typed image
decisions, measured honestly against hosted models, before anything is trained or fitted. `glance` reads three typed
questions, a yes/no judgment, a pick-one over a list of options, a level on an ordered rating scale, from the logits
of a single forward pass of Qwen3-VL-4B-Instruct rather than generating text. This paper reports how good that
reading is against hosted flagship and cheapest-tier models, against the same model's own generated answer, and
against itself once it has seen unlabeled or labeled examples of a rating rubric.
<!-- src: docs/paper/METHODS.md sections 1-2 -->

The contributions below are measured properties of the frozen model plus this readout, not claims about the readout's
originality; Section 6 places the readout among a family of training-free tools that already do the same thing for
text and, in two cases, for images.

1. A fresh-photograph, cross-provider measurement. On photographs taken after every evaluated model's release,
   labelled by people outside this project (Commons "depicts" statements, iNaturalist community identifications), the
   open model read this way is statistically indistinguishable (overlapping 95% intervals, n = 131 and 65) from three hosted flagships and each provider's cheapest current model on
   yes/no and pick-one (Section 3).
2. Reading against writing on the same model. The same frozen model asked to generate a JSON answer gives identical
   yes/no and pick-one decisions on the items measured; reading is 2.4 to 6.1 times faster and returns probabilities
   the generated answer does not (Section 3).
3. Ratings told as one arc, because this is where most of what is ours lives: order versus exact level, who leads
   zero-shot, a one-pass read at the JSON answer position that closes most of the zero-shot gap to writing,
   self-calibration from unlabeled images, a content-free prior that fails, label curves, and a hidden-state ceiling
   (Section 4).
4. A pre-registered notebook with published misses. Every experiment behind this paper has a hypothesis written down
   before its data existed; several were not supported, and we report them next to the ones that were (Section 7).

We do not claim the readout itself is new. Scoring a candidate statement's logit at a forced answer position, with
nothing decoded, is the same primitive that VQAScore, Kadavath et al.'s probes of a language model's stated
confidence, and several open reproductions of a commercial typed-decision interface (Simple Jev, jev-visual, LitJev)
already use, in two of the latter cases on images (Section 6). What we measure is what that primitive delivers on a
frozen open model, honestly compared with hosted alternatives, and what a small amount of calibration adds.
<!-- src: docs/paper/RELATED_WORK.md -->

## 2. Setup

Every question is one of three typed forms: a yes/no judgment (`noul`), a pick-one over 2 to 128 named options
(`choice`), or a level on an ordered 2- to 10-level scale (`score`). Each candidate answer, one yes/no statement, one
option, one scale level, is scored in its own forward pass: the user turn holds the image(s) and the question, the
assistant turn is opened and left empty, and we read the logits of the allowed answer tokens ("Yes"/"No" variants, or
a rating's digit tokens) at that position through a float32 copy of the output head, never decoding a token. A yes/no
answer is the sigmoid of the Yes-minus-No logit; a pick-one or rating answer is a softmax over the per-option or
per-level logits. Several questions about one image can share the cost of reading it.
<!-- src: docs/paper/METHODS.md sections 1-2 -->

The open model is Qwen3-VL-4B-Instruct (Apache-2.0), frozen, float16, on one Apple M5 laptop (32 GB RAM, MPS). A
frozen dual-encoder baseline, SigLIP2-base (Apache-2.0), appears alongside it on pick-one questions; two other sizes
of the same family (2B, 8B) and one model of a different family (SmolVLM2-2.2B, Apache-2.0) are read the same way in
Section 5. <!-- src: docs/paper/METHODS.md section 12 -->

Three tests, all zero-shot unless stated otherwise, use the same items across every system compared in a table: (a)
yes/no, 131 fresh Wikimedia Commons photographs (262 questions, one "yes" and one seeded "no" per photograph); (b)
pick one of 13, the same 131 Commons photographs (hosted models answered a 65-photograph test half, so headline
comparisons restrict the open model to that half); (c) rating, the exact level of 4, on five synthetic single-factor
degradation scales (blur, underexposure, JPEG artifacts, noise, low resolution), 1,000 held-out images, 200 per scale.
A second, more cleanly labelled set, 200 iNaturalist observations (400 yes/no questions, pick one of 10 organism
groups), repeats (a) and (b). The Commons set holds photographs captured on or after 2026-08-15, after every
evaluated model's release, CC0/CC BY/CC BY-SA "own work" uploads labelled only by uploaders' and editors' structured
"depicts" statements; the iNaturalist set holds research-grade observations uploaded the day of the test, labelled by
the iconic taxon at least two community identifiers agreed on. No label in either set was made by us or by any model.
<!-- src: results/lab/matrix.md, results/lab/fresh_commons.md, results/lab/fresh_inat.md, docs/paper/RESULTS_LAB.md, docs/paper/DATASETS.md -->

Hosted models were queried once each, zero-shot, through a JSON schema enumerating every question's allowed answers,
at temperature 0 where a provider accepts it; a failed call counts as wrong. Only whether the pick was correct is
stored, never the pick itself, so no hosted output becomes a training label. Hosted wall time is the median seconds
per call from the laptop; hosted cost is the provider's logged bill where recorded, otherwise a list-price upper
estimate marked as such throughout. Open-model cost is arithmetic on measured seconds and an assumed cloud GPU price,
not a bill. Every experiment behind Sections 3 to 5 has a hypothesis recorded, with its own timestamp, in a lab
notebook before the data that would test it existed, and a verdict recorded afterward, including every case not
supported; we report both, and collect the misses in Section 7.
<!-- src: docs/paper/METHODS.md section 11, results/lab/cost_model.md, results/lab/frontier_cost_measured.md, lab/NOTES.md -->

## 3. Yes/no and pick-one

### 3.1 The headline matrix

On the same items, across three tests and eight systems (three hosted flagships, three of the same providers'
cheapest current models, and the open model both read and written), accuracy intervals on yes/no and pick-one overlap
almost everywhere:

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 0.947 [0.908, 0.985] | 0.923 [0.846, 0.985] | 0.650 [0.621, 0.679] |
| Claude Opus 5 | 0.924 [0.878, 0.969] | 0.908 [0.831, 0.969] | 0.550 [0.519, 0.581] |
| GPT-5.6 | 0.893 [0.840, 0.947] | 0.892 [0.815, 0.954] | 0.597 [0.567, 0.627] |
| Claude Haiku 4.5 (cheapest Anthropic) | 0.939 [0.893, 0.977] | 0.846 [0.754, 0.923] | 0.609 [0.579, 0.639] |
| GPT-5.6 Luna (cheapest OpenAI) | 0.924 [0.878, 0.969] | 0.908 [0.831, 0.969] | 0.686 [0.657, 0.715] |
| Gemini 3.1 Flash-Lite (cheapest Google) | 0.954 [0.916, 0.985] | 0.908 [0.831, 0.969] | 0.763 [0.737, 0.789] |
| Qwen3-VL-4B, written | 0.931 [0.885, 0.969] | 0.892 [0.815, 0.954] | 0.672 [0.643, 0.701] |
| Qwen3-VL-4B, read (Glance) | 0.931 [0.885, 0.969] | 0.862 [0.769, 0.938] | 0.669 [0.640, 0.698] |

The yes/no column uses the 131 test-half questions and the pick-one column the 65 test-half photographs the hosted models
were shown, for every system including the open model (Section 3.2 gives the
open model's numbers on the full 131 instead). Every yes/no and pick-one interval above overlaps every other. Timed on these
same photographs with the GPU otherwise idle, reading takes 1.08 s per yes/no (writing 1.65 s) and 1.43 s per 13-way
pick-one (writing 1.87 s): faster than five of the six hosted models on yes/no (Claude Haiku 4.5 is faster, 0.87 s), and
\$0.16 to \$0.24 per 1,000 yes/no answers on a rented GPU against \$0.31 for the cheapest hosted model (Gemini 3.1
Flash-Lite). That is somewhat cheaper, not an order of magnitude: each provider's cheapest current model is itself many
times cheaper than its flagship, and on a full-size photograph most of the open model's time goes into reading the image.
An earlier version of these cells (0.33 s, "about five times cheaper") was timed on 448 px test images and is withdrawn
(erratum, entry 48).
<!-- src: results/lab/matrix.md, lab/PHOTO_TIMING.json, lab/NOTES.md entries 45b, 48, 48b -->

### 3.2 Both photo sets in full

The open model's own Commons numbers on all 131/262 items (rather than the 65-item hosted subset of 3.1) are 0.931
[0.901, 0.958] yes/no and 0.885 [0.824, 0.939] pick-one. A second, more cleanly labelled set, iNaturalist, repeats
both tests:

| System | iNaturalist yes/no (n=400 open, 200 hosted) | iNaturalist pick-one (n=200 open, 100 hosted) |
| --- | --- | --- |
| Qwen3-VL-4B, read (Glance) | 0.945 [0.922, 0.968] | 0.940 [0.905, 0.970] |
| SigLIP2 (dual encoder) | - | 0.880 [0.835, 0.925] |
| Gemini 3.1 Pro | 0.960 [0.930, 0.985] | 0.910 [0.850, 0.960] |
| Claude Opus 5 | 0.945 [0.910, 0.975] | 0.930 [0.880, 0.980] |
| GPT-5.6 | 0.935 [0.900, 0.965] | 0.910 [0.850, 0.960] |
| Claude Haiku 4.5 | 0.830 [0.780, 0.880] | 0.790 [0.710, 0.870] |
| GPT-5.6 Luna | 0.935 [0.900, 0.965] | 0.860 [0.790, 0.920] |
| Gemini 3.1 Flash-Lite | 0.965 [0.935, 0.990] | 0.930 [0.880, 0.980] |

Only whether each hosted pick was correct is stored, never the pick itself. Almost every interval overlaps, with one
exception: Claude Haiku 4.5 trails here (0.830/0.790, neither overlapping the open model's 0.945/0.940), 11 to 14
points under its own flagship, unexpected when we registered this comparison (Section 7). On four older public
benchmarks that may sit inside every model's training data (POPE, GQA yes/no, Oxford Pets, Caltech-101) the same
reading trails Claude Opus 5's zero-shot pick by 3.2 points on average [1.0, 5.4]; on fresh photographs, which no
model could have trained on, that gap disappears into overlapping intervals, so it is not obviously a contamination
artifact working in the open model's favor.
<!-- src: results/lab/fresh_commons.md, results/lab/fresh_inat.md, docs/paper/RESULTS_ZEROSHOT.md section 1, lab/NOTES.md entries 33, 45b -->

### 3.3 A finer test that separates the hosted models

The two photo sets above sit near ceiling for almost every system, which leaves open whether "level with hosted models"
only means "the task is easy". We registered a deliberately harder closed-set test before collecting it (entry 47): 210
iNaturalist research-grade photographs taken after every model's release, labelled by insect order (seven orders that look
alike), with every "no" question naming another insect order rather than an unrelated subject.

| System | yes/no (n=420 open, 210 hosted) | pick one of 7 (n=210 open, 105 hosted) |
| --- | --- | --- |
| Qwen3-VL-4B, read (Glance) | 0.948 [0.926, 0.967] | 0.962 [0.933, 0.986] |
| SigLIP2 (dual encoder) | - | 0.710 [0.648, 0.771] |
| Gemini 3.1 Pro | 0.967 [0.943, 0.990] | 0.971 [0.933, 1.000] |
| Claude Opus 5 | 0.938 [0.905, 0.967] | 0.962 [0.924, 0.990] |
| Gemini 3.1 Flash-Lite | 0.962 [0.933, 0.986] | 0.952 [0.905, 0.990] |
| GPT-5.6 | 0.943 [0.910, 0.971] | 0.905 [0.848, 0.952] |
| GPT-5.6 Luna | 0.867 [0.819, 0.910] | 0.886 [0.819, 0.943] |
| Claude Haiku 4.5 | 0.786 [0.729, 0.838] | 0.743 [0.657, 0.829] |

This test does separate systems: hosted pick-one accuracy spans 23 points (0.743 to 0.971) and the dual encoder falls to
0.710. The open 4B model stays within one point of the best hosted system on pick-one and two points on yes/no, with
overlapping intervals, and above three of the six hosted models on both. So the level result is not only a ceiling
effect, at least at this grain; we have not tested species-level or expert-level distinctions.
<!-- src: results/lab/fresh_inat_orders.md, lab/NOTES.md entries 47, 47b, 47c -->

Labels on all three photo sets are community or uploader labels, not a human audit. As a rough floor on label noise we
count the items that every one of the seven systems answers "wrongly": 4 of 131 and 3 of 65 on Commons, 5 of 200 and 3 of
100 on iNaturalist (2.5% to 4.6%). Seven independent systems making the same mistake is less likely than a wrong or
ambiguous label, so part of every system's distance from 1.0 is the labels, not the models.
<!-- src: results/lab/label_noise_proxy.md, lab/NOTES.md entry 51 -->

### 3.4 Written against read, and other sizes

Reading changes nothing about what the frozen model knows on yes/no and pick-one. Asked to generate a JSON object
instead, the same model gives the identical decision on the fresh Commons items: agreement +0.0 points [-1.1, +1.1]
on 262 yes/no questions and +0.0 [-3.8, +3.8] on 131 pick-one photographs, zero invalid or unparsable outputs.
<!-- src: results/lab/gen_accuracy.md --> What reading changes is what surrounds the answer: cold start on an idle
GPU, reading is 2.4 times faster than writing for one yes/no question, 3.5 times for five mixed questions, 6.1 times
for 25 rating questions (40 images per shape, one laptop), and it returns a probability at no extra decoding cost.
<!-- src: lab/GENBENCH.md, results/lab/cost_model.md --> A smaller open model does as well on yes/no out of the box:
Qwen3-VL-2B reaches 0.939 [0.908, 0.966] on Commons yes/no and 0.925 [0.898, 0.950] on iNaturalist yes/no, both
overlapping the 4B model above; pick-one is more size-sensitive (Section 5).
<!-- src: results/lab/other_models.json -->

## 4. Ratings, the hard case

### 4.1 Order versus exact level

A rating asks for a level on a four-level rubric described in words: blur, underexposure, JPEG artifacts, noise, low
resolution; 1,000 held-out images, 200 per scale. Zero-shot, the frozen model gets the order right and the exact
level often wrong: the raw, uncalibrated read is exactly right on 0.558 of images but within one level on 0.987 (mean
absolute error 0.47 levels, rank agreement 0.934). Its misses are not evenly spread: it is, for example, about half a
level too harsh on blur and almost never uses the JPEG scale's top level, a constant per-rubric offset rather than
random noise, which a small calibration can remove and a content-free correction (4.4) cannot.
<!-- src: results/lab/label_free_test.md, results/lab/scaling.md, docs/paper/METHODS.md section 14.1 -->

### 4.2 Who leads, zero-shot

| System | labels used | mean exact accuracy |
| --- | --- | --- |
| Claude Opus 5 | 0 | 0.550 [0.519, 0.581] |
| GPT-5.6 | 0 | 0.597 [0.567, 0.627] |
| Gemini 3.1 Pro | 0 | 0.650 [0.621, 0.679] |
| Claude Haiku 4.5 | 0 | 0.609 [0.579, 0.639] |
| GPT-5.6 Luna | 0 | 0.686 [0.657, 0.715] |
| Gemini 3.1 Flash-Lite | 0 | 0.763 [0.737, 0.789] |
| Qwen3-VL-4B, v0 readout as shipped (single temperature) | 0 | 0.510 |
| Qwen3-VL-4B, four-pass `ens4d` (the shipped fitting target) | 0 | 0.570 |
| Qwen3-VL-4B, written JSON answer | 0 | 0.672 [0.643, 0.701] |
| Qwen3-VL-4B, one-pass read at the JSON position (`jsondigits`) | 0 | 0.669 [0.640, 0.698] |

Same 1,000 images for every row (n=200 per scale). Each provider's cheapest current model scores higher, zero-shot,
than its own flagship here, the opposite of what we had expected: Claude Haiku 4.5 above Claude Opus 5, GPT-5.6 Luna
above GPT-5.6, Gemini 3.1 Flash-Lite above Gemini 3.1 Pro, the strongest zero-shot system we measured on this task.
<!-- src: results/lab/matrix.md, results/lab/frontier_head_to_head.md, lab/NOTES.md entry 45b -->

### 4.3 One pass, four passes, or writing

The readout the harness ships as the fitting target (`ens4d`, four passes, chosen because a calibration in the loop
forgives a constant offset) is a poor zero-shot elicitation: 0.570, ten points below the same model's own generated
JSON answer on the same images (0.672). A written answer is itself a logit read in disguise: greedy decoding of
`{"answer": 2}` takes the argmax after `{"answer": `. Reading the digit logits at that forced position in one pass
(`jsondigits`) reaches 0.669, agrees with the written answer on 98.2% of items, within one level on 0.979 (ECE 0.248,
uncalibrated). Paired on the same images it is ahead of Claude Opus 5 by 11.9 points [7.8, 16.0] and GPT-5.6 by 7.2
[2.7, 11.8], and level with Gemini 3.1 Pro (+1.9 [-2.5, 6.3]); it is not yet the harness default for zero-shot
ratings, pending a registered check on harder benchmarks (Section 7).
<!-- src: results/lab/gen_accuracy.md, results/lab/jsondigits.md, docs/paper/METHODS.md section 14.2, docs/paper/RESULTS_ZEROSHOT.md section 3 -->

### 4.4 Zero labels, not zero-shot: self-calibration from unlabeled images

A pool of unlabeled images of the rubric lets each readout's level logits be z-scored against that pool's own mean and
standard deviation, with no parameter fit to any label: zero labels, never zero-shot, since it still needs images
covering the rubric's range. On the four-pass readout it raises mean exact accuracy from 0.558 to 0.686 with 16
unlabeled images and to 0.697 with the full 500-image pool; a badly unbalanced pool (70% of images at one level) keeps
only part of the gain (0.646). Calibration error falls too: mean ECE (floor in parentheses) goes from 0.327 (0.033)
zero-shot to 0.175 (0.066) at 16 images, negative log-likelihood 1.94 to 0.82, not calibrated in the sense a labeled
fit is (about 0.03 ECE), so usable for the level, not yet for a confidence threshold. On the one-pass JSON-position
read the same idea reaches 0.758 at 16 unlabeled images, level with Gemini 3.1 Flash-Lite's zero-shot pick (0.763) and
ahead of the five other hosted systems measured.
<!-- src: results/lab/label_free_test.md, results/lab/label_free_ece.md, results/lab/jsondigits.md, lab/NOTES.md entry 45b -->

A correction that needs no images of the task at all, subtracting the model's own reading of content-free inputs
(flat grey, black, white, three seeded noise images) as a prior, does NOT work: it takes mean exact accuracy from
0.558 to 0.400, because the frozen model reads a blank or noise image as the worst level of most quality rubrics, so
subtracting that reading as a "prior" pushes every real image toward the mild end of the scale. The hypothesis that
this would gain at least 3 points was not supported; it lost ground on four of five scales. For an image rubric there
appears to be no content-free image. <!-- src: results/lab/null_prior.md, docs/paper/METHODS.md section 14.1 -->

### 4.5 If you have labels

| Setting | mean exact accuracy | note |
| --- | --- | --- |
| v0 readout as shipped, single temperature, 0 labels | 0.500 | zero-shot |
| v0 readout, best-fitted bias and temperature | 0.810 | fitted, not zero-shot |
| Four-pass `ens4d` + matrix calibration, 32 labels | 0.857 | `glance fit` |
| Four-pass `ens4d` + matrix calibration, 500 labels | 0.867 [0.854, 0.881] | same items, bootstrap interval |
| Seven-readout ensemble + matrix calibration, 500 labels | 0.876 | research variant, not shipped |
| Hidden-state readout, one pass, about a hundred labels | 0.965 | research result, not shipped |

Same 1,000 images throughout. With about 32 labels per rubric the shipped four-pass readout is ahead of every hosted
zero-shot system by 9.4 to 30.7 points (paired 95% intervals in `results/lab/frontier_head_to_head.md`); a
calibration is specific to its own rubric (fit on blur and applied to JPEG it reaches 0.262 against 0.772 fit on JPEG
itself). Hand-built classical no-reference image features (29 of them, logistic regression) reach 0.979 with the full
label set, ahead of this readout's 0.867, though the readout leads with only 32 labels (0.856 against 0.752). The
table's hidden-state row needs on the order of a hundred labels to overtake the digit readout, trailing it below
about 32 (0.826 against 0.855 at n=32); it suggests the frozen model already represents severity almost as well as
the hand-built features do, so the zero-shot gap looks like a readout and convention problem more than a perception
one. <!-- src: docs/paper/RESULTS_LAB.md sections 3, 4, 5, 8, docs/paper/RESULTS_COMPARISONS.md sections 1, 3, lab/READOUT_LADDER.md, lab/NOTES.md entry 40 -->

### 4.6 KADID-10k: the misses

On KADID-10k (23 severity distortion types at 5 levels, human DMOS scores, evaluation only, no photograph shared with
the lab scales), every registered target was missed: exact accuracy 0.527 with the full per-distortion label set
(target >= 0.70), within one level 0.880 (target >= 0.97), mean absolute error 0.642 levels (target <= 0.40), mean
per-type Spearman correlation with human scores 0.763 (target >= 0.85). Pooled ECE (0.032) cleared its 0.05 target
only once the small-sample floor (0.020) was accounted for, because the criterion as registered, ECE <= 0.05 on at
least 20 of 23 distortions individually, could not be met even by a perfectly calibrated predictor at this sample
size. The zero-shot read is weaker still (0.328 exact); a calibration learned on 22 other distortions barely helps a
distortion it has not seen (0.350 against 0.328 raw), while removing the readout's bias with unlabeled images of the
held-out distortion helps more (0.433 combined). Where zero-shot ratings must place an exact severity level across
many distortion types with no per-rubric fit, this method is not solved.
<!-- src: docs/paper/RESULTS_GENERALIZATION.md, docs/paper/RESULTS_ZEROSHOT.md section 3 -->

## 5. Does it depend on the model?

The same questions and readouts, unchanged, were run on two other sizes of the same open family (Qwen3-VL-2B,
Qwen3-VL-8B) and on one model of a different family (SmolVLM2-2.2B: a SigLIP vision tower, an SmolLM2 language model,
Apache-2.0). <!-- src: results/lab/scaling.md, results/lab/other_models.json, lab/SMOLVLM2_REPORT.md -->

| Open model | Commons yes/no | Commons pick-one | iNaturalist yes/no | iNaturalist pick-one |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.939 [0.908, 0.966] | 0.832 [0.763, 0.893] | 0.925 [0.898, 0.950] | 0.940 [0.905, 0.970] |
| Qwen3-VL-4B | 0.931 [0.901, 0.958] | 0.885 [0.832, 0.939] | 0.945 [0.923, 0.968] | 0.940 [0.905, 0.970] |
| Qwen3-VL-8B | 0.924 [0.889, 0.954] | 0.878 [0.817, 0.931] | 0.935 [0.910, 0.958] | 0.955 [0.925, 0.980] |

Yes/no is flat across size, every interval overlapping every other; pick-one is more sensitive, with the 2B model
trailing on the 13-way Commons task while matching larger models on the 10-way iNaturalist task.
<!-- src: results/lab/other_models.json -->

| Open model | zero-shot exact, four-pass read | within one level | rank agreement | + 16 unlabeled images | + 32 labels |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.388 [0.358, 0.417] | 0.893 | 0.893 | 0.644 [0.618, 0.669] | 0.842 [0.825, 0.858] |
| Qwen3-VL-4B | 0.570 [0.540, 0.601] | 0.988 | 0.934 | 0.694 [0.669, 0.718] | 0.853 [0.835, 0.870] |
| Qwen3-VL-8B | 0.537 [0.506, 0.568] | 0.979 | 0.915 | 0.714 [0.690, 0.737] | 0.839 [0.820, 0.857] |
| SmolVLM2-2.2B | 0.419 [0.389, 0.449] | 0.837 | 0.833 | 0.519 [0.493, 0.547] | 0.814 [0.794, 0.833] |

Same 1,000 images the hosted models saw, n=200 per scale. Going from 2B to 4B raises zero-shot exact accuracy by 18.2
points; 4B to 8B loses 3.3, and the hypothesis that accuracy rises monotonically with size was not supported, nor was
the hypothesis that rank agreement rises with size (0.893, 0.934, 0.915 for 2B, 4B, 8B); a third, that 8B would still
stay under 0.70 zero-shot, was supported. Once labels exist the three sizes converge, within 1.4 points of each other
at 32 labels, so the fitted recipe travels across sizes even though zero-shot quality does not, and the gain from
those 32 labels did not shrink with size as expected (45.4, 28.3, 30.2 points for 2B, 4B, 8B).
<!-- src: results/lab/scaling.md, lab/NOTES.md entry 38c -->

The fitted recipe also travels to that different family without a wording change: the v0 readout as shipped scores
0.384, the same readout with its best calibration 0.762, the one-pass digit readout with a matrix calibration 0.775,
and the four-pass ensemble with a matrix calibration 0.854 (200 labels/scale), the same registered ordering as on
Qwen3-VL-4B. Zero-shot it is much weaker (raw four-pass read 0.419 exact, within one level 0.837, against 0.570 and
0.987 for the 4B model): zero-shot quality is a property of the model being read, not of the harness. Two families
and three sizes is not "any model"; SmolVLM2's own yes/no and pick-one accuracy on the fresh photographs was still
collecting when this draft was written (Section 7).
<!-- src: lab/SMOLVLM2_REPORT.md, lab/NOTES.md entries 44, 46 -->

## 6. Related work and positioning

Reading a candidate statement's logit at a forced answer position, with nothing decoded, is not a primitive we
invented. VQAScore reads P(Yes) to "Does this figure show {text}?" from a vision-language model for image-text
alignment; Kadavath et al. read a language model's P(True) on a proposed answer; and a family of open reproductions of
a commercial typed-decision interface already do the same thing, in two cases on images: jev-visual shares one vision
prefill across up to 64 question suffixes on a 0.8B model, and LitJev serves that interface's own request and response
schema from a Hugging Face checkpoint, reading option logits instead of generating, with a vision checkpoint when one
is loaded. On yes/no and pick-one, the forward pass here is the same object as those, and the accuracy we report
belongs to the open model that is read, not to the harness around it.
<!-- src: docs/paper/RELATED_WORK.md, "Verified today" and "Open reproductions" sections -->

As far as the sources we checked go, what we did not find already published: self-calibration of a rating readout
from unlabeled images and a labeled matrix fit for rating levels (`glance fit --unlabeled`, `glance fit`); a
fresh-photograph board against three hosted flagships and three hosted cheapest-tier models with measured dollars and
milliseconds; and the same frozen model's generated answer measured against its own read answer on identical items.
<!-- src: docs/paper/RELATED_WORK.md, "Statements we stand behind" -->

This project is also not the same kind of system as entries in the same family that train something: YOFO fine-tunes
2B-scale Qwen-VL checkpoints with LoRA so a template of binary requirements is judged in one pass; Laya Vision trains
a decision head (about 150M parameters) on a frozen 256M vision-language backbone, and its own rating type is
untrained; OpenJev v2 fine-tunes a 4B model as an NLI-style claim scorer. None of these train the model we read: no
weights are distributed or updated here, only fitted readout files of a few hundred numbers per rubric. A trained head
can be calibrated by construction but is tied to its own backbone; what this harness reads is whatever open
vision-language model is attached, and it moves when a better one ships, with no retraining.
<!-- src: docs/paper/RELATED_WORK.md, "Typed-decision models" table and "Statements we stand behind" -->

We do not claim this works on any open vision-language model (two families, three sizes, Section 5), that it is
faster or cheaper than the commercial interface it is modelled on (that interface is text-only; no image-to-image
cost comparison exists), or that its probabilities are calibrated for ratings out of the box (Section 4). We do not
call the harness a model, and we avoid describing it as a vision counterpart of that commercial interface:
image-input, Jev-shaped open servers already exist, and what a calibration and measurement layer adds on top of the
shared primitive is exactly what Sections 3 and 4 report.
<!-- src: docs/paper/RELATED_WORK.md, "Lead with the ask" and the pitch paragraph -->

## 7. Limitations and misses

We registered a hypothesis before most experiments in this paper and report every verdict, including the following
that this paper's evidence did not support.

| Registered claim (short form) | Measured | Entry |
| --- | --- | --- |
| Self-calibration from 16 unlabeled images reaches >= 0.70 exact | 0.686 at 16 images, 0.697 with the full pool | 26b |
| A universal calibration across KADID distortions exceeds raw by >= 5 points and is within 8 of a per-rubric fit | +2.2 points; 17.7 points below the per-rubric fit | 28 |
| Adaptive escalation keeps >= 90% of the four-pass gain | 88% kept | 27b |
| Written and raw-read ratings stay within 5 points; unlabeled/32-label reads beat written by >= 10/25 points | trailed by 9.5-28.5 points on four scales; beat written by only +3.0 and +18.5 | 32b |
| SigLIP2 stays within 5 points of the open model on iNaturalist pick-one | 6.0 points behind | 35b |
| Rating accuracy, rank agreement and the 32-label gain all rise with model size; fresh-photo accuracy orders 8B >= 4B >= 2B | none held: 8B (0.537) below 4B (0.570); rank agreement 0.893/0.934/0.915; label gain 45.4/28.3/30.2 pts; every fresh-photo gap inside its interval | 38c |
| A content-free prior improves zero-shot rating accuracy by >= 3 points | -15.8 points | 39b |
| A one-pass fitted read stays within 3 points of the four-pass fitted read at 32 labels | 3.1 points behind | 42b |
| Every cheap hosted model stays within 5 points of its own flagship; cheap models score at or below their flagships on ratings | Claude Haiku 4.5 11-14 points behind Opus 5 on iNaturalist; every cheap model scored ABOVE its flagship on ratings, Flash-Lite reaching 0.763 | 45b |
| KADID-10k: exact accuracy >= 0.70, within-one >= 0.97, MAE <= 0.40, per-type SRCC >= 0.85 | 0.527, 0.880, 0.642, 0.763 | 28 |

<!-- src: lab/NOTES.md entries 26b, 27b, 28, 32b, 35b, 38c, 39b, 42b, 45b; docs/paper/RESULTS_GENERALIZATION.md -->

Two published errata sit behind this paper's cost numbers. First, a latency erratum: an earlier latency table reported
the four-pass rating readout's cost as 609 ms next to 432 ms for the shipped v0 readout, implying the two costs were
about the same; both figures mixed a warm, prefix-cached measurement with a cold one. On a fresh image the four-pass
readout costs 1,089 ms against 441 ms for the cached v0 readout, 2.5 to 2.8 times more, not the same; every derived
document has since been corrected, and accuracy and calibration results were unaffected, since the method was
selected by cross-validated negative log-likelihood, not by latency. Second, a small-fit overconfidence erratum: the
fitting procedure originally estimated a calibration's sharpness scalar on its own training data, nearly separable at
a few dozen labels, which left fitted probabilities overconfident (negative log-likelihood 0.749 against 0.478 at 32
labels); the scalar is now estimated on held-out folds instead, changing no accuracy number (it multiplies every
logit alike) but changing how much the reported probabilities can be trusted at small label counts.
<!-- src: lab/NOTES.md entries 16, 18 -->

Other limits, stated plainly: every rating scale here, including KADID-10k, is a synthetic single-factor degradation,
none a naturally occurring multi-factor judgment; no hosted model was given a few-shot prompt or a rubric example
before scoring, so Section 4 compares each provider's zero-shot, not best achievable, performance; yes/no and pick-one
on fresh photographs sit near ceiling for every system (all near 0.93), so "level with hosted models" partly reflects
that the task is currently easy, and on older, possibly contaminated benchmarks a hosted flagship led by 3 to 5 points
(Section 3.2) - the harder insect-order test of Section 3.3 separates the hosted models and the open model holds, but
species-level and expert distinctions are untested; labels on the photo sets are community labels with an estimated
2.5% to 4.6% noise floor and no human audit; timings are from one laptop GPU (Section 3.1); written answers on iNaturalist were not collected, so Section 3.4 uses the Commons
set only, iNaturalist being queued (entry 35c); the one-pass JSON-position read of Section 4.3 has not been checked on
KADID-10k or the creative-QA rubrics, so it is not yet the harness default for zero-shot ratings (entry 43); two open,
MIT-licensed outside systems selected for a head-to-head on the lab scales have not yet been run (entry 37); and
SmolVLM2's own yes/no and pick-one accuracy on the fresh photographs was still being collected when this draft was
written (entry 46). <!-- src: lab/NOTES.md entries 35c, 37, 38c, 43, 46, 47 -->

## 8. Reproducibility

Every result in this paper can be regenerated from this repository.

```bash
uv sync && uv run glance doctor && uv run glance ask photo.jpg "Is there a dog?"     # setup, one question

uv run python tools/fetch_fresh_inat.py                                # 200 photographs, about ten API calls
uv run glance eval --suite inat_choice --suite inat_yesno --model vlm
uv run python tools/make_results_zeroshot.py                           # every table on the project page

uv run glance eval                                                     # all suites, siglip + vlm, 4 h budget
uv run glance fit --unlabeled --data folder_of_your_images/            # 16 or more images, no labels
uv run glance fit --data labels/ --rubric rubric.json                  # about 32 labeled images
```

<!-- src: README.md "Try it", "How much setup does a question need?", "Evaluate"; site/page.html "Reproduce" -->

`lab/NOTES.md` is the notebook of Section 2; frontier outputs are never written to any file that could serve as a
training label, only whether each answer was right, and the call log redacts the rest.
<!-- src: README.md "Where things are"; docs/paper/METHODS.md section 11 -->

Licensing: the open model (Qwen3-VL-4B-Instruct) and the dual-encoder baseline (SigLIP2-base) are both Apache-2.0,
and only Apache-2.0 or MIT weights are used anywhere in this project. Photographs are CC0, CC BY or CC BY-SA,
attributed per file in source manifests, and not redistributed (manifests hold item identifiers, hashes and licence
fields; images are re-fetched into a local, gitignored cache). Two older benchmarks (POPE, Caltech-101) were
re-sourced from official MIT- or CC-BY-licensed releases rather than an ambiguous mirror; RVL-CDIP was skipped
outright over an unclear licence. KADID-10k (Section 4.6) carries no formal licence, only a statement that it is
"freely available to the research community"; it is used under an explicit, evaluation-only exception, never to fit
anything the harness ships, and nothing from it is redistributed beyond item identifiers, distortion levels,
DMOS-derived metrics and model logits. Hand-labeled private data never leaves the machine and was empty in every run
behind this paper. <!-- src: docs/paper/DATASETS.md -->

---

## Draft notes for the editor

**(a) Places where two source files disagreed on a number.**

1. Claude Opus 5's Commons yes/no accuracy 95% interval: `results/lab/fresh_commons.md` gives 0.924 [0.878, 0.962];
   `results/lab/matrix.md` (and `site/page.html`) give 0.924 [0.878, 0.969]. Same point estimate, different upper
   bound. This draft uses the matrix.md figure in Section 3.1 and the fresh_commons.md figure in Section 3.2.
2. GPT-5.6's Commons yes/no accuracy 95% interval: `results/lab/fresh_commons.md` gives 0.893 [0.840, 0.939];
   `results/lab/matrix.md` (and `site/page.html`) give 0.893 [0.840, 0.947]. Same pattern as (1).
3. "Sixteen unlabeled images" self-calibration exact accuracy for the four-pass readout has three different values
   across three files that all cite the same underlying experiment (entry 26): 0.686 in `results/lab/label_free_test.md`
   (pool = 16, 20 draws), 0.673 in `results/lab/label_free_ece.md`, and 0.679 in `results/lab/null_prior.md`.
   `label_free_ece.md` itself notes the draws differ from `label_free_test.md` and calls the gap a "third decimal"
   difference, but 0.686 versus 0.673 differs in the second decimal.
4. `docs/CLAIMS.md` states "0.697 exact" for "16 or more UNLABELED images," which is actually the value for the full
   500-image unlabeled pool in `results/lab/label_free_test.md`; the same file's row for exactly 16 images gives
   0.686, which is the number `docs/paper/OUTLINE.md`'s own results map cites for "16 images." We used 0.686 in
   Section 4.4 as the entry-16 figure and 0.697 as the full-pool figure, sourced separately.
5. The 500-label, four-pass rating accuracy is 0.867 [0.854, 0.881] in `docs/paper/RESULTS_LAB.md` section 8 and in
   `lab/READOUT_LADDER.md`'s "full" column, but 0.868 in `results/lab/frontier_head_to_head.md` and
   `docs/paper/RESULTS_COMPARISONS.md` section 4. We used 0.867 [0.854, 0.881] in Section 4.5 because it carries the
   only sourced interval.
6. The raw four-pass readout's accuracy at 32 labels is 0.853 in `results/lab/jsondigits.md` (used as the comparison
   point for the one-pass read's H41 verdict) but 0.857 in `results/lab/frontier_head_to_head.md` and
   `docs/paper/RESULTS_COMPARISONS.md`. The 3.1-point miss we report for H41 in Section 7 uses 0.822 against 0.853
   (jsondigits.md's own comparison); against 0.857 the miss would be 3.5 points, which would not change the verdict.
7. `results/lab/matrix.md` and `lab/NOTES.md` entry 45b report measured (not "est.") costs for the three cheapest
   hosted models (Claude Haiku 4.5, GPT-5.6 Luna, Gemini 3.1 Flash-Lite), but `results/lab/frontier_cost_measured.md`
   and `docs/paper/RESULTS_COMPARISONS.md` section 7, the files that otherwise document measured hosted cost, list
   only the three flagships. We could not cross-check the cheap-model cost figures against a second numeric source
   file beyond the notebook entry itself.
8. Commons-photo hosted costs in `results/lab/matrix.md` (Gemini 3.1 Pro \$2.62 yes/no, GPT-5.6 \$7.55 yes/no) differ
   from the "fresh photos" figures in `results/lab/frontier_cost_measured.md` (Gemini 3.1 Pro \$3.03, GPT-5.6 \$7.64,
   one blended "yes/no or pick-one" figure over 196 calls). The difference is plausibly explained by
   `frontier_cost_measured.md` blending yes/no and pick-one calls into one number while `matrix.md` reports them
   separately, but neither file states this explicitly.
9. `results/lab/matrix.md`'s own prose gives the open model's yes/no cost as "\$0.05 to \$0.07" per 1,000 answers,
   while `lab/NOTES.md` entry 45b's prose for the same comparison says "\$0.05 to \$0.08."
10. `results/lab/matrix.md`'s column header reads "Yes/no, 131 fresh Commons questions," but the accuracy figures
    under it (also in `results/lab/fresh_commons.md`) are computed over n=262 questions, one "yes" and one "no" per
    photograph; "131" appears to count photographs, not questions, and we kept the header wording in Section 3.1 for
    consistency with the source table while stating n=262 explicitly in the text.

**(b) Claims we wanted to make but could not find a source for, and therefore left out.**

1. `site/page.html`'s "why read" section states an 80%-coverage selective accuracy of 0.975 on the iNaturalist set. We
   could not find this figure in any required numeric source file (it is not in `results/lab/fresh_inat.md`,
   `results/lab/fresh_inat.json`'s visible top-level fields, or `results/lab/gen_accuracy.md`), so Section 3 does not
   repeat it.
2. We wanted a single "about five times cheaper" dollar comparison with a 95% interval, matching the style of this
   paper's accuracy claims; the cost files (`results/lab/cost_model.md`, `results/lab/frontier_cost_measured.md`,
   `results/lab/matrix.md`) report only point ranges, never confidence intervals, so Section 3.1 reports a range
   without one.
3. `docs/paper/OUTLINE.md`'s own contributions list states the marginal cost of one more rating readout on an
   already-prefilled image as "about 150 ms." The only figure we could confirm against a numeric source file is 151
   ms, the marginal cost of one more `digits` readout, in `lab/NOTES.md` entry 16 and `docs/paper/RESULTS_LAB.md`
   section 7; we did not use OUTLINE's rounded figure directly since we could not find its own source citation.
4. We wanted to report which options the hosted models actually picked on pick-one disagreements, to characterize the
   kinds of errors each system makes. Frontier picks are never stored by design (Section 2), so this is not
   available from any file in this repository.

**(c) Sentences in `site/page.html` we believe are no longer fully supported by the result files.**

1. The abstract's claim that "the open model needs 16 unlabeled images ... to draw level" with "the cheapest Google
   model (0.763 against 0.669)" rests on the one-pass read reaching 0.758 with 16 unlabeled images
   (`results/lab/jsondigits.md`), which is 0.005 below Flash-Lite's 0.763. `lab/NOTES.md` entry 45b itself uses the
   word "level" for this comparison, so we kept the same word in our abstract, but the two point estimates do not
   cross, and a reader checking the numbers directly would find the open model still trailing by half a point.
2. Site Table 1 gives the open model's "read (Glance)" rating speed as 0.45 s. The closest sourced timing we found
   for a single rating read, outside `results/lab/matrix.md` itself, is in `lab/GENBENCH_SINGLE.md`: the `fast2`
   two-pass readout at 440 ms, or the four-pass `ens4d` readout at 1,083 ms. Neither file names a timing specifically
   for the one-pass `jsondigits` read that the accuracy column next to it represents. We used matrix.md's 0.45 s
   directly in Section 3.1 since it is the designated headline source, but flag that its exact provenance among the
   three candidate readouts is not stated in any file we read.
3. Site section 2.1's Table 2 gives the open model's Commons pick-one accuracy as 0.885 [0.824, 0.939] (131 items),
   while the same-looking "pick-one" column of the headline Figure 1 and Table 1 (`results/lab/matrix.md`) gives
   0.862 [0.769, 0.938] (the 65-item subset the hosted models saw). The page's Table 2 caption explains the
   difference ("the frontier models answered the test half"), but nothing next to Figure 1 itself flags that its own
   "read" row is already restricted to that same smaller subset; a reader comparing the two figures side by side
   could reasonably read them as the same statistic. This draft states the item counts explicitly in Sections 3.1 and
   3.2 to avoid the same ambiguity.

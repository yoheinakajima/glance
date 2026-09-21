<!-- Working draft, written with AI assistance from the repository's result files; every three-decimal number is checked against
     a committed result file by tools/verify_docs_numbers.py. The project page (site/) is generated from the same files. -->

# Reading Typed Decisions from a Frozen Open Vision-Language Model

## Abstract

A typed question is one whose legal answers form a closed set known before the model runs: yes or no, one of a list, a level
on a rubric. We put such questions about images to a frozen open vision-language model (Qwen3-VL-4B-Instruct, Apache-2.0, on
a laptop) and read the answer from the logits of one forward pass; nothing is generated. On photographs taken after every
model's release, labelled by people outside this project (three sets, 541 yes/no questions and 270 pick-one photographs put
to every system), the open model is level with the best hosted models on pick-one (0.933 against 0.937) and about two points
behind the best on yes/no (0.939 against 0.961; the paired difference excludes zero for both Gemini models); it is
no difference is detected from Claude Opus 5 or GPT-5.6 on either, and it is ahead of two of the three low-cost hosted models on both. The questions are easy and the labels imperfect: 3 to 5% of items are answered "wrongly" by all seven systems.
Reading is as accurate as the same model writing JSON. Ratings behave differently. On five image-quality scales, zero-shot, every system orders
images correctly (within one level on 0.99 of images) and places the level boundaries wrongly, by a constant offset per
rubric; a low-cost hosted model leads (0.763 against 0.669), and an 8B model is no better than a 4B one. Because a read
answer is a vector of logits it can be fitted: 16 unlabeled images of the rubric remove most of the offset (0.758) and 32
labels reach 0.857. The hosted models were not given examples, so this is a comparison of products, not of models. On a real
image-quality benchmark (KADID-10k) the approach missed every target we registered, hand-built features and a small trained
quality model do as well or better on low-level artefacts, and on rubrics that are not image quality (tilt, how much of a
subject is cut off) even 300 labels reach only 0.55. On images drawn by program the open model reads text, coarse position and
small counts reliably and fails on mirror-image direction and relative size. The open model's cost is of the same order as the low-cost hosted models and one to two orders below the
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
originality; Section 8 places the readout among a family of training-free tools that already do the same thing for
text and, in two cases, for images.

1. A fresh-photograph, cross-provider measurement. On photographs taken after every evaluated model's release,
   labelled by people outside this project (Commons "depicts" statements, iNaturalist community identifications), the
   open model read this way is, pooled over three sets and paired on the same items, level with the best hosted models on
   pick-one and about two points behind the two Gemini models on yes/no, with no difference detected from Claude Opus 5 or GPT-5.6,
   and ahead of two of the three low-cost hosted models; one of the sets is a harder insect-order test on which the hosted
   models spread over 23 points (Section 3).
2. Where that stops. On images drawn by program and on synthetic interface screens, with labels exact by construction,
   the open model reads text, finds the element that serves a goal (0.993) and counts to five, and fails on mirror-image
   direction (0.30) and relative size (0.52): its limit is geometry, not reading (Section 4).
3. Reading against writing on the same model. The same frozen model asked to generate a JSON answer is exactly as
   accurate on yes/no and pick-one (the same right/wrong outcome on 95% to 99.5% of items); reading is about 1.5 times
   faster on a full-size photograph, 2.4 to 6.1 times on small images, and returns probabilities the generated answer
   does not (Section 5).
4. Ratings told as one arc, because this is where most of what is ours lives: order versus exact level, who leads
   zero-shot, a one-pass read at the JSON answer position that closes most of the zero-shot gap to writing,
   self-calibration from unlabeled images, a content-free prior that fails, label curves, a hidden-state ceiling, and
   what lies outside the quality scales, where the pattern does not hold: a real quality benchmark, a small trained
   quality model that does as well, and rubrics that a 300-label fit does not rescue (Section 6).
5. A pre-registered notebook with published misses. Every experiment behind this paper has a hypothesis written down
   before its data existed; several were not supported, and we report them next to the ones that were (Section 9).

We do not claim the readout itself is new. Scoring a candidate statement's logit at a forced answer position, with
nothing decoded, is the same primitive that VQAScore, Kadavath et al.'s probes of a language model's stated
confidence, and several open reproductions of a commercial typed-decision interface (Simple Jev, jev-visual, LitJev)
already use, in two of the latter cases on images (Section 8). What we measure is what that primitive delivers on a
frozen open model, honestly compared with hosted alternatives, and what a small amount of calibration adds.
<!-- src: docs/paper/RELATED_WORK.md -->

## 2. Setup

Every question is one of three typed forms: a yes/no judgment (`noul`), a pick-one over 2 to 128 named options
(`choice`), or a level on an ordered 2- to 10-level scale (`score`). Each candidate answer, one yes/no statement, one
option, one scale level, is scored in its own forward pass: the user turn holds the image(s) and the question, the
assistant turn is opened and left empty, and we read the logits of the allowed answer tokens ("Yes"/"No" variants, or
a rating's digit tokens) at that position through a float32 copy of the output head, never decoding a token. A yes/no
answer is the sigmoid of the Yes-minus-No logit; a pick-one or rating answer is a softmax over the per-option or
per-level logits. Several questions about one image share the cost of reading it: they reuse the image's key-value
prefix and branch into independent suffixes, so they cannot influence each other, and packing them this way changed 0 of
100 checked predictions.
<!-- src: docs/paper/METHODS.md sections 1-2, docs/paper/RESULTS_LAB.md section 9 -->

The open model is Qwen3-VL-4B-Instruct (Apache-2.0), frozen, float16, on one Apple M5 laptop (32 GB RAM, MPS). A
frozen dual-encoder baseline, SigLIP2-base (Apache-2.0), appears alongside it on pick-one questions; two other sizes
of the same family (2B, 8B) and one model of a different family (SmolVLM2-2.2B, Apache-2.0) are read the same way in
Section 7. <!-- src: docs/paper/METHODS.md section 12 -->

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
supported; we report both, and collect the misses in Section 9.
<!-- src: docs/paper/METHODS.md section 11, results/lab/cost_model.md, results/lab/frontier_cost_measured.md, lab/NOTES.md -->

## 3. Yes/no and pick-one

### 3.1 The headline matrix

On the same items, across three tests and eleven rows (three hosted flagships, three of the same providers'
cheapest current models, and the open models: Qwen3-VL at 2B, 4B and 8B read with Glance, and the 2B and 4B models writing
JSON without it; a written answer that does not parse counts as wrong for every system, which is what sinks the 2B written
row (scored leniently it reaches 0.915, 0.907 and 0.495, Section 5):

| System | Yes/no, 541 questions, three fresh photo sets pooled | Pick one, 270 photographs, three sets pooled | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 0.959 [0.941, 0.976] | 0.937 [0.907, 0.963] | 0.650 [0.621, 0.679] |
| Claude Opus 5 | 0.937 [0.914, 0.958] | 0.937 [0.907, 0.963] | 0.550 [0.519, 0.581] |
| GPT-5.6 | 0.928 [0.904, 0.951] | 0.904 [0.867, 0.937] | 0.597 [0.567, 0.627] |
| Claude Haiku 4.5 (cheapest Anthropic) | 0.839 [0.809, 0.869] | 0.785 [0.733, 0.833] | 0.609 [0.579, 0.639] |
| GPT-5.6 Luna (cheapest OpenAI) | 0.906 [0.881, 0.930] | 0.881 [0.841, 0.919] | 0.686 [0.657, 0.715] |
| Gemini 3.1 Flash-Lite (cheapest Google) | 0.961 [0.944, 0.977] | 0.933 [0.904, 0.963] | 0.763 [0.737, 0.789] |
| Qwen3-VL-2B, writing JSON (strict scoring) | 0.584 [0.548, 0.621] | 0.841 [0.796, 0.885] | 0.056 [0.043, 0.071] |
| Qwen3-VL-2B + Glance | 0.904 [0.877, 0.930] | 0.907 [0.874, 0.941] | 0.492 [0.461, 0.524] |
| Qwen3-VL-4B, writing JSON | 0.945 [0.925, 0.963] | 0.930 [0.900, 0.959] | 0.672 [0.643, 0.701] |
| Qwen3-VL-4B + Glance | 0.939 [0.917, 0.959] | 0.933 [0.904, 0.959] | 0.669 [0.640, 0.698] |
| Qwen3-VL-8B + Glance | 0.933 [0.910, 0.955] | 0.926 [0.893, 0.956] | 0.643 [0.614, 0.671] |

The yes/no and pick-one columns pool the three fresh photo sets on the items the hosted models were shown (the test
halves), for every system including the open model; Section 3.2 gives the paired differences and Section 3.3 the sets
one by one. The same model writing its answer scores 0.945 and 0.930 against 0.939 and 0.933 read: reading costs no
accuracy. Timed on these
same photographs with the GPU otherwise idle, reading takes 1.08 s per yes/no (writing 1.65 s) and 1.43 s per 13-way
pick-one (writing 1.87 s): faster than five of the six hosted models on yes/no (Claude Haiku 4.5 is faster, 0.87 s), and
\$0.16 to \$0.24 per 1,000 yes/no answers on a rented GPU against \$0.31 for the cheapest hosted model (Gemini 3.1
Flash-Lite). That is somewhat cheaper, not an order of magnitude: each provider's cheapest current model is itself many
times cheaper than its flagship, and on a full-size photograph most of the open model's time goes into reading the image.
The two sides are not measured the same way: hosted cost is the provider's bill per call and hosted latency includes the
network and the provider's queue, while open-model cost is measured laptop seconds at an on-demand cloud GPU price with
no batching, idle time or operator counted; a different GPU price or image resolution moves it by more than the gap to the
low-cost hosted models. Hosted cost also depends on image size: GPT-5.6 measured \$7.55 per 1,000 answers on the
1,280-pixel Commons files and \$1.86 on the roughly 500-pixel iNaturalist files. The durable differences are not the
cents: no image leaves the machine, there is no per-call bill, it works offline, and the answer can be fitted.
An earlier version of these cells (0.33 s, "about five times cheaper") was timed on 448 px test images and is withdrawn
(erratum, entry 48).
<!-- src: results/lab/matrix.md, results/lab/frontier_cost_measured.md, lab/PHOTO_TIMING.json, lab/NOTES.md entries 45b, 48, 48b -->

### 3.2 Three photo sets pooled

Within any one photo set every 95% interval overlaps every other, which says as much about sample size as about the
systems. Pooling the three fresh sets (Commons, iNaturalist ten groups, iNaturalist insect orders; the items the hosted
models were asked; a failed call counts as wrong) and pairing on the same items gives 541 yes/no questions and 270
pick-one photographs per system. This pooling was added after the per-set results had been seen; its rule (every set,
every system, nothing dropped) was fixed before it was computed (entry 54). Intervals come from a bootstrap stratified by set that
resamples photographs, all questions of a photograph together (entry 62); an interval that spans zero is reported as "no difference
detected", never as equivalence.

| System | yes/no, n=541 | open minus this, points | pick-one, n=270 | open minus this, points |
| --- | --- | --- | --- | --- |
| Qwen3-VL-4B + Glance | 0.939 [0.917, 0.959] | - | 0.933 [0.904, 0.959] | - |
| Claude Opus 5 | 0.937 [0.914, 0.958] | +0.2 [-1.7, +2.1] | 0.937 [0.907, 0.963] | -0.4 [-3.0, +2.2] |
| GPT-5.6 | 0.928 [0.904, 0.951] | +1.1 [-0.7, +3.0] | 0.904 [0.867, 0.937] | +3.0 [+0.0, +6.3] |
| Gemini 3.1 Pro | 0.959 [0.941, 0.976] | -2.0 [-3.9, -0.2] | 0.937 [0.907, 0.963] | -0.4 [-3.0, +2.2] |
| Claude Haiku 4.5 | 0.839 [0.809, 0.869] | +10.0 [+7.3, +12.8] | 0.785 [0.733, 0.833] | +14.8 [+10.4, +19.3] |
| GPT-5.6 Luna | 0.906 [0.881, 0.930] | +3.3 [+1.1, +5.7] | 0.881 [0.841, 0.919] | +5.2 [+1.5, +8.9] |
| Gemini 3.1 Flash-Lite | 0.961 [0.944, 0.977] | -2.2 [-4.0, -0.6] | 0.933 [0.904, 0.963] | +0.0 [-3.0, +3.0] |

On pick-one we detect no difference from the four best hosted models, and for three of them the whole paired interval lies
within three points. On yes/no we detect no difference from Claude Opus 5 or GPT-5.6 (both intervals within three points); the open model is about two points behind both Gemini models (a
small lead that no single set could show), and ahead of GPT-5.6 Luna and Claude Haiku 4.5. The equal-weight-per-set means
tell the same story (0.938 against 0.958 and 0.960), so the result is not an artefact of two of the three sets being
nature photographs.
<!-- src: results/lab/pooled_photos.md, lab/NOTES.md entries 54, 55 -->

### 3.3 The photo sets one by one

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
points under its own flagship, unexpected when we registered this comparison (Section 9). On four older public
benchmarks that may sit inside every model's training data (POPE, GQA yes/no, Oxford Pets, Caltech-101) the same
reading trails Claude Opus 5's zero-shot pick by 3.2 points on average [1.0, 5.4]; on fresh photographs, which no
model could have trained on, that gap disappears into overlapping intervals, so it is not obviously a contamination
artifact working in the open model's favor.
<!-- src: results/lab/fresh_commons.md, results/lab/fresh_inat.md, docs/paper/RESULTS_ZEROSHOT.md section 1, lab/NOTES.md entries 33, 45b -->

### 3.4 A finer test that separates the hosted models

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

## 4. Where coarse recognition ends: geometry, not reading

Six sets of 150 images drawn by program, with labels exact by construction, mark where this stops for the open model
read this way (registered as entry 50; zero-shot, uncalibrated, shipped readouts).

| Question | options | Qwen3-VL-4B, read (n=150) |
| --- | --- | --- |
| Which of six look-alike words is printed? | 1 of 6 | 1.000 [1.000, 1.000] |
| Is the ball left of / right of / above / below the square? | 1 of 2 | 0.940 [0.900, 0.973] |
| How many balls? (1 to 8) | 1 of 8 | 0.913 [0.867, 0.953] |
| How many red balls? (0 to 6) | 1 of 7 | 0.887 [0.833, 0.933] |
| Which way do the stripes run? | 1 of 4 | 0.653 [0.573, 0.727] |
| Which of four shapes is the largest? | 1 of 4 | 0.520 [0.440, 0.600] |

Reading text, coarse position and small counts are fine: one to five balls are counted almost without error, then 0.89 at
six and seven and 0.56 at eight. Two things are not. Horizontal and vertical stripes are told apart perfectly (76 of 76)
while the two diagonal directions are confused (22 of 74 correct, below a coin flip: a mirror-image confusion). And the
largest of four like shapes is found on 0.73 of images when it has twice the area of the others, falling to 0.59, 0.45
and 0.32 at area ratios 1.5, 1.3 and 1.15 (chance 0.25). We had predicted at least 0.90 on stripes and on the twofold
size difference, and a fall below 0.70 for six to eight balls (measured 0.78); all three predictions were wrong
(Section 9). Hosted models had not been run on these sets when this draft was written, so whether they share these
weaknesses is not known.
<!-- src: results/lab/probes.md, results/lab/probes.json, lab/NOTES.md entries 50, 50b, 50c -->

Five hundred synthetic interface screens (five kinds of page, invented content, rendered from generated HTML so every
label is exact; registered as entry 49) ask what an agent would ask of a screen.

| Question | options | Qwen3-VL-4B, read |
| --- | --- | --- |
| Goal in words: which numbered mark should be clicked? | 1 of 6 to 8 | 0.993 [0.983, 1.000] (n=300) |
| The same, after one step of reasoning (cheaper plan, out-of-stock item, earliest date) | 1 of 2 to 6 | 0.910 [0.850, 0.960] (n=100) |
| Is the page in this state? (error shown, dialog open, signed in, ...) | 1 of 2 | 0.922 [0.898, 0.944] (n=500) |
| Is this goal already done? | 1 of 2 | 0.825 [0.770, 0.875] (n=200) |
| What kind of page is this? | 1 of 5 | 0.823 [0.780, 0.867] (n=300) |

Finding the element that serves a stated goal is nearly perfect, and one step of reasoning costs eight points (out of
stock 1.00, cheaper plan 0.94, earliest date 0.79). The weak spots are one-sided. The disabled state of the main button
is reported on only 0.07 of the screens that have it (the other five states 0.92 to 1.00); our screens draw it as a
pale tint of the accent colour while the question says "greyed out", so part of that miss may belong to the test, which
we did not change after seeing the result. "Already done" is right on every not-done screen and on 0.65 of done
screens. Every page-type error is another page called an article, an uncalibrated bias toward one option of the kind a
fit removes. We had predicted at least 0.90 on page type, and that a single forward pass would trail the best hosted
model by ten points on the reasoning screens; at 0.910 the second cannot hold whatever the hosted models score. They
had not been run on these screens when this draft was written.
<!-- src: results/lab/ui_screens.md, results/lab/ui_screens.json, lab/NOTES.md entries 49, 49b, 49c -->

## 5. Reading against writing

Reading changes nothing about what the frozen model knows on yes/no and pick-one. Asked to generate a JSON object
instead, the same model is exactly as accurate: read minus written is +0.0 points [-1.1, +1.1] on 262 Commons yes/no
questions, +0.0 [-3.8, +3.8] on 131 Commons pick-one photographs, +0.0 [-0.8, +0.8] on 400 iNaturalist yes/no questions
and -0.5 [-2.5, +1.0] on 200 iNaturalist pick-one photographs, with zero invalid or unparsable outputs. The answers are
not identical item by item: an item comes out the same way, right or wrong, on 99.2%, 95.4%, 99.5% and 98.5% of the
four sets, the rest reflecting the two prompts, which differ (a JSON request against one statement per option).
<!-- src: results/lab/gen_accuracy.md, lab/NOTES.md entries 32b, 32c, 32d --> Where the prompt IS the same (the one-pass
rating read of Section 6.3, taken at the position where the written answer puts its digit) 98.2% of answers are
identical, as expected when a greedy written answer is an argmax over the same logits. The two diverge further when
the prompts differ more (Section 6.3), and when one written JSON object
carries 25 ratings, each conditioned on the fields already written, it matches 25 independent reads on only 59% of fields
(no ground truth for that request, so a difference, not an error rate). What reading changes is cost and form. For one
question about a full-size photograph, where encoding the image dominates, reading is about 1.5 times faster (1.08 s
against 1.65 s, Section 3.1); on small 448-pixel test images it is 2.4 times faster for one yes/no question, 3.5 times for
five mixed questions and 6.1 times for 25 rating questions (40 images per shape, one laptop, idle GPU); and the answer is
a probability vector that can be thresholded, ranked and fitted, at no extra decoding cost.
<!-- src: lab/GENBENCH.md, lab/PHOTO_TIMING.json, results/lab/cost_model.md, lab/NOTES.md entries 27c, 48, 48b -->

## 6. Ratings, the hard case

Scope first. "Ratings" in Sections 6.1 to 6.5 means five synthetic, single-factor, four-level image-quality scales
(blur, exposure, JPEG artifacts, noise, resolution) on 1,000 held-out images, the same for every system. It does not mean
aesthetic judgement or any naturally occurring, multi-factor score. The pattern reported here holds on these scales and
does not hold outside them; Section 6.6 collects what lies outside: a real image-quality benchmark, specialised tools
that do as well or better, and rubrics that are not image quality.

### 6.1 Order versus exact level

A rating asks for a level on a four-level rubric described in words: blur, underexposure, JPEG artifacts, noise, low
resolution; 1,000 held-out images, 200 per scale. Zero-shot, the frozen model gets the order right and the exact
level often wrong: the raw, uncalibrated read is exactly right on 0.558 of images but within one level on 0.987 (mean
absolute error 0.47 levels, rank agreement 0.934). Its misses are not evenly spread: it is, for example, about half a
level too harsh on blur and almost never uses the JPEG scale's top level, a constant per-rubric offset rather than
random noise, which a small calibration can remove and a content-free correction (6.4) cannot.
<!-- src: results/lab/label_free_test.md, results/lab/scaling.md, docs/paper/METHODS.md section 14.1 -->

### 6.2 Who leads, zero-shot

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

### 6.3 One pass, four passes, or writing

The readout the harness ships as the fitting target (`ens4d`, four passes, chosen because a calibration in the loop
forgives a constant offset) is a poor zero-shot elicitation: 0.570, ten points below the same model's own generated
JSON answer on the same images (0.672). A written answer is itself a logit read in disguise: greedy decoding of
`{"answer": 2}` takes the argmax after `{"answer": `. Reading the digit logits at that forced position in one pass
(`jsondigits`) reaches 0.669, agrees with the written answer on 98.2% of items, within one level on 0.979 (ECE 0.248,
uncalibrated). Paired on the same images it is ahead of Claude Opus 5 by 11.9 points [7.8, 16.0] and GPT-5.6 by 7.2
[2.7, 11.8], and level with Gemini 3.1 Pro (+1.9 [-2.5, 6.3]). A registered check on harder benchmarks decided whether
it becomes the harness default for a rubric with nothing fitted: on five rubrics that are not image quality it scores
0.375 against 0.334 for the four-pass read (ahead on 3 of 5), and on KADID-10k 0.347 against 0.328; the rule fixed in
advance (at least as good on the first, not worse on the second) was met, although the 5-point gain we had predicted
on KADID-10k was not (+1.9). It is now the default; the four-pass read remains the method for labeled fits.
<!-- src: results/lab/gen_accuracy.md, results/lab/jsondigits.md, docs/paper/METHODS.md section 14.2, docs/paper/RESULTS_ZEROSHOT.md section 3 -->

### 6.4 Zero labels, not zero-shot: self-calibration from unlabeled images

A pool of unlabeled images of the rubric lets each readout's level logits be z-scored against that pool's own mean and
standard deviation, with no parameter fit to any label: zero labels, never zero-shot, since it still needs images
covering the rubric's range. On the four-pass readout it raises mean exact accuracy from 0.558 to 0.686 with 16
unlabeled images and to 0.697 with the full 500-image pool; a badly unbalanced pool (70% of images at one level) keeps
only part of the gain (0.646). Calibration error falls too: mean ECE (floor in parentheses) goes from 0.327 (0.033)
zero-shot to 0.175 (0.066) at 16 images, negative log-likelihood 1.94 to 0.82, not calibrated in the sense a labeled
fit is (about 0.03 ECE), so usable for the level, not yet for a confidence threshold. On the one-pass JSON-position
read the same idea reaches 0.758 at 16 unlabeled images, half a point short of Gemini 3.1 Flash-Lite's zero-shot pick
(0.763 [0.737, 0.789], well inside its interval) and ahead of the five other hosted systems measured.
<!-- src: results/lab/label_free_test.md, results/lab/label_free_ece.md, results/lab/jsondigits.md, lab/NOTES.md entry 45b -->

A correction that needs no images of the task at all, subtracting the model's own reading of content-free inputs
(flat grey, black, white, three seeded noise images) as a prior, does NOT work: it takes mean exact accuracy from
0.558 to 0.400, because the frozen model reads a blank or noise image as the worst level of most quality rubrics, so
subtracting that reading as a "prior" pushes every real image toward the mild end of the scale. The hypothesis that
this would gain at least 3 points was not supported; it lost ground on four of five scales. For an image rubric there
appears to be no content-free image. <!-- src: results/lab/null_prior.md, docs/paper/METHODS.md section 14.1 -->

### 6.5 If you have labels

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

### 6.6 Outside the quality scales: KADID-10k, specialised tools, other rubrics

On KADID-10k (23 severity distortion types at 5 levels, human DMOS scores, evaluation only, no photograph shared with
the lab scales), every registered target was missed: exact accuracy 0.527 with the full per-distortion label set
(target >= 0.70), within one level 0.880 (target >= 0.97), mean absolute error 0.642 levels (target <= 0.40), mean
per-type Spearman correlation with human scores 0.763 (target >= 0.85). Pooled ECE (0.032) cleared its 0.05 target
only once the small-sample floor (0.020) was accounted for, because the criterion as registered, ECE <= 0.05 on at
least 20 of 23 distortions individually, could not be met even by a perfectly calibrated predictor at this sample
size. The zero-shot read is weaker still (0.328 exact); a calibration learned on 22 other distortions barely helps a
distortion it has not seen (0.350 against 0.328 raw), while removing the readout's bias with unlabeled images of the
held-out distortion helps more (0.433 combined). Where zero-shot ratings must place an exact severity level across
many distortion types with no per-rubric fit, this method is not solved. On the same KADID items the one-pass read
of Section 6.3 scores 0.347 zero-shot against 0.328 for the four-pass read.

Specialised tools do as well or better on low-level artefacts. With plentiful labels 29 hand-built features score 0.979 on
the synthetic scales against 0.867 for the fitted model. A 0.9B model trained for image quality (Q-SiT-mini), given our
matrix scaling on its five level-word logits, the same 32 labels per scale and the same test items, scores 0.853 against
0.846 for the frozen 4B model (difference +0.7 points [-1.1, +2.5]; with 300 labels 0.869 against 0.863): no difference detected,
at a quarter of the size. We had predicted it would rank only the three distortions it was trained for and stay below the
4B model; it ranks all five scales (Spearman 0.86 to 0.95). A general model read this way earns its place by answering any
typed question with one set of frozen weights, not by being the best quality meter. The other outside system we ran, a 4B
open model trained to score claims (openjev v2), behaved as predicted: 0.412 on its own, 0.772 with our fit, and on the same
items with the same 32-label fit 0.750 against 0.846 for the frozen 4B model (-9.7 points [-11.5, -7.9]). The fit is a part
that transfers to another system's readout.

Nor does the pattern of Section 6.1 extend to every rubric. On five synthetic rubrics that are not image quality (subject
cut off by the frame, occlusion, tilt, caption legibility, watermark; exact ground truth from segmentation masks and
drawing parameters) the zero-shot one-pass read is exactly right on 0.375 of images and within one level on 0.741 (chance
0.25; tilt and cut-off at chance), the four-pass read 0.334, and 16 unlabeled images do not help (0.366). A labeled fit
does not rescue them either: with 300 labels per rubric the four-pass read reaches 0.548 exact and 0.897 within one level,
where we had predicted at least 0.75 and 0.95 (occlusion 0.727, caption legibility 0.650, watermark 0.557, cut-off 0.477,
tilt 0.330). A fit removes an offset; it cannot supply a judgement the model does not make, and the weakest rubrics are
the geometric ones, in line with the drawn probes of Section 4.
<!-- src: docs/paper/RESULTS_GENERALIZATION.md, results/lab/classical_baselines.md, results/lab/jsondigits_hard.md, lab/SEMANTIC_REPORT.md, results/lab/external_same_items.md, results/lab/external_systems.md, lab/NOTES.md entries 20, 25, 28, 36c, 37c, 37e, 43b, 43c -->
<!-- src: docs/paper/RESULTS_GENERALIZATION.md, docs/paper/RESULTS_ZEROSHOT.md section 3 -->

## 7. Does it depend on the model?

The same questions and readouts, unchanged, were run on two other sizes of the same open family (Qwen3-VL-2B,
Qwen3-VL-8B) and on one model of a different family (SmolVLM2-2.2B: a SigLIP vision tower, an SmolLM2 language model,
Apache-2.0). <!-- src: results/lab/scaling.md, results/lab/other_models.json, lab/SMOLVLM2_REPORT.md -->

| Open model | Commons yes/no | Commons pick-one | iNaturalist yes/no | iNaturalist pick-one |
| --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.939 [0.908, 0.966] | 0.832 [0.763, 0.893] | 0.925 [0.898, 0.950] | 0.940 [0.905, 0.970] |
| Qwen3-VL-4B | 0.931 [0.901, 0.958] | 0.885 [0.832, 0.939] | 0.945 [0.923, 0.968] | 0.940 [0.905, 0.970] |
| Qwen3-VL-8B | 0.924 [0.889, 0.954] | 0.878 [0.817, 0.931] | 0.935 [0.910, 0.958] | 0.955 [0.925, 0.980] |
| SmolVLM2-2.2B (another family) | 0.931 [0.901, 0.962] | 0.870 [0.809, 0.924] | 0.892 [0.863, 0.923] | 0.815 [0.755, 0.865] |

Within the Qwen3-VL family yes/no is flat across size, every interval overlapping every other; pick-one is more
sensitive, with the 2B model trailing on the 13-way Commons task while matching larger models on the 10-way iNaturalist
task. The 2.2B model of another family, read through the same scorer with nothing reworded, is level with the 4B model
on everyday photographs (tied on Commons yes/no to the item count) and trails it on nature photographs by 5 points on
yes/no and 12 on pick-one, the pick-one intervals not overlapping. We had predicted floors of 0.85 and 0.75, which held,
and that it would sit below the 4B model on every cell, which held on three of four. So the readout carries to another
family unchanged, and the comparison with hosted models in Section 3 is a statement about Qwen3-VL, not about every
small open model.
<!-- src: results/lab/other_models.json, lab/NOTES.md entries 46, 46b -->

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

The one-pass read of Section 6.3 shows the same picture and is at or above the four-pass read zero-shot on every model
measured: 0.492, 0.669 and 0.643 for Qwen3-VL-2B, 4B and 8B (against 0.388, 0.570, 0.537), and 0.440 against 0.419 for
SmolVLM2-2.2B; with 16 unlabeled images it reaches 0.623, 0.758, 0.677 and 0.609. The 8B model is again not above the
4B model. <!-- src: results/lab/scaling.md, lab/NOTES.md entries 38c, 42b, 44b -->

The fitted recipe also travels to that different family without a wording change: the v0 readout as shipped scores
0.384, the same readout with its best calibration 0.762, the one-pass digit readout with a matrix calibration 0.775,
and the four-pass ensemble with a matrix calibration 0.854 (200 labels/scale), the same registered ordering as on
Qwen3-VL-4B. Zero-shot it is much weaker (raw four-pass read 0.419 exact, within one level 0.837, against 0.570 and
0.987 for the 4B model): zero-shot quality is a property of the model being read, not of the harness. Two families
and three sizes is not "any model". The harness loads any Hugging Face image-text model with a chat template through the same
backend (`--model-id`); that path was checked end to end on SmolVLM2-2.2B only (entry 46c), and any other model is untested.
<!-- src: lab/SMOLVLM2_REPORT.md, lab/NOTES.md entries 44, 46 -->

Speed and cost by size. Timed the same way at every size (one laptop, GPU otherwise idle), a read yes/no about a full-size
photograph takes 0.66, 1.08 and 2.12 s at 2B, 4B and 8B: each doubling of the model roughly doubles the time and the cost, and
above 4B it buys no accuracy. Whether reading saves more as the model grows depends on what dominates. On a full-size
photograph the image is encoded either way, and writing costs 1.6, 1.5 and 1.4 times a read for yes/no (1.1, 1.3, 1.1 for
pick-one): a steady saving, not a growing one. On 448-pixel images, where the answer tokens are most of the work, the saving
grows with size, 1.4, 2.0 and 3.2 times for a rating, because every generated token costs a full pass of a larger model while a
read stays one pass. The cost model uses one rented-GPU price for every size, and the 2B model's written answers are short and
often invalid, which flatters its writing time. This is a description of timings already collected, not a registered test.
<!-- src: results/lab/size_speed.md, lab/NOTES.md entry 60 -->

## 8. Related work and positioning

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

We do not claim this works on any open vision-language model (two families, three sizes, Section 7), that it is
faster or cheaper than the commercial interface it is modelled on (that interface is text-only; no image-to-image
cost comparison exists), or that its probabilities are calibrated for ratings out of the box (Section 6). We do not
call the harness a model, and we avoid describing it as a vision counterpart of that commercial interface:
image-input, Jev-shaped open servers already exist, and what a calibration and measurement layer adds on top of the
shared primitive is exactly what Sections 3 and 4 report.
<!-- src: docs/paper/RELATED_WORK.md, "Lead with the ask" and the pitch paragraph -->

Speed, in context. Hosted Jev (TypeSafe, September 2026) is the trained product of this family; it takes text, not images.
Its launch material quotes 40 to 200 times faster than frontier language models. Independent timings show what that figure
is made of. TrueStandard (Agrahri 2026) timed one three-way classification of a support ticket at 477 ms of server time
against 790 ms for Gemini 3.1 Flash Lite and 928 ms for Claude Haiku 4.5, 1.7 and 1.9 times, and reached 100 times only when
one call replaced six sequential calls to a thinking model; in its words, the multiple is a property of the comparison, not
of the model. Goedecke (2026) measured 2 to 3 times from having a small open model emit one constrained token instead of
written structured output, and dorarep (2026) found that going from one question to a hundred per request cost Jev 1.5 times
the latency where generating models paid 6 to 28 times. Our ratios sit in the same modest band, with the image as a fixed
cost that text systems do not pay: 1.5 times against the same model writing JSON on a full-size photograph, 2.4 to 6.1 times
on small images as questions per image grow, 1.5 times against Gemini 3.1 Flash-Lite, and slower than Claude Haiku 4.5 (1.08 s
against 0.87 s). We did not run a sequential thinking-model workflow and claim nothing about one. None of the outside figures
is our measurement, and raw milliseconds do not transfer between a hosted text model and a 4B vision model on a laptop.
<!-- src: docs/paper/RELATED_WORK.md "Speed context", results/lab/matrix.md, lab/NOTES.md entry 56 -->

## 9. Limitations and misses

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
| Rubrics that are not image quality: the fitted four-pass read reaches at least 0.75 exact and 0.95 within one | 0.548 and 0.897 with 300 labels per rubric; tilt 0.330 (tilt worst, as predicted) | 36c |
| The one-pass read beats the four-pass read zero-shot on KADID-10k by at least 5 points | +1.9 points (0.347 against 0.328) | 43c |
| Interface screens: at least 0.90 on page type; one forward pass at least 10 points behind the best hosted model on one-step reasoning | page type 0.823 (every error is another page called an article); reasoning 0.910, so a 10-point gap is impossible | 49c |
| Rendered probes: at least 0.90 on stripe direction; under 0.70 for six to eight balls; at least 0.90 for the largest shape at twice the area | stripes 0.653 (the two diagonals are confused); 0.782 for six to eight balls; 0.73 at twice the area, 0.32 at 1.15 times | 50c |
| A 0.9B model trained for image quality (Q-SiT-mini) is weak on exposure and resolution, and with our fit stays below the four-pass read (0.867) | it ranks all five scales (Spearman 0.86 to 0.95) and scores 0.869; on the same items with the same fit 0.869 against 0.863 (300 labels) and 0.853 against 0.846 (32 labels), both differences within 3 points of zero | 37c |

<!-- src: lab/NOTES.md entries 26b, 27b, 28, 32b, 35b, 37c, 38c, 39b, 42b, 45b; docs/paper/RESULTS_GENERALIZATION.md; results/lab/external_same_items.md -->

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
before scoring, so Section 6 compares each provider's zero-shot, not best achievable, performance; yes/no and pick-one
on fresh photographs sit near ceiling for every system (all near 0.93), so "level with hosted models" partly reflects
that the task is currently easy, and on older, possibly contaminated benchmarks a hosted flagship led by 3 to 5 points
(Section 3.3) - the harder insect-order test of Section 3.4 separates the hosted models and the open model holds, but
species-level and expert distinctions are untested; labels on the photo sets are community labels with an estimated
2.5% to 4.6% noise floor and no human audit; timings are from one laptop GPU (Section 3.1); the one-pass JSON-position read of Section 6.3, now the default, is still poor
in absolute terms on KADID-10k (0.347) and on rubrics that are not image quality (0.375), and its request wording for a question about several images at once has not been measured (entry 43c); and two open,
MIT-licensed outside systems were selected for a head-to-head on the lab scales (entry 37): the first, a 0.9B quality model, matched the fitted four-pass read, and the second, a 4B trained claim scorer, stayed 9 to 10 points behind it (Section 6.6). <!-- src: lab/NOTES.md entries 35c, 37, 38c, 43, 46, 47 -->

## 10. Reproducibility

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
outright over an unclear licence. KADID-10k (Section 6.6) carries no formal licence, only a statement that it is
"freely available to the research community"; it is used under an explicit, evaluation-only exception, never to fit
anything the harness ships, and nothing from it is redistributed beyond item identifiers, distortion levels,
DMOS-derived metrics and model logits. Hand-labeled private data never leaves the machine and was empty in every run
behind this paper. <!-- src: docs/paper/DATASETS.md -->

---

# Related work and how this project compares (written 2026-09-20)

Two tiers. "Verified" entries were looked up today and the link was read; the comparison uses only what those pages
say. "From memory" entries are well known to the author of this note but were NOT re-checked today: confirm title,
venue and claims before citing. Nothing here is a claim of novelty; it is a map of who did what.

## Verified today

| Work | What it does | Relation to glance |
| --- | --- | --- |
| VQAScore: Lin et al., "Evaluating Text-to-Visual Generation with Image-to-Text Generation", ECCV 2024. https://linzhiqiu.github.io/papers/vqascore/ | Scores image-text alignment as the probability of "Yes" to "Does this figure show {text}?", one forward pass of a VQA model. State of the art on 8 alignment benchmarks; their CLIP-FlanT5 model is fine-tuned. | Same primitive as glance's `noul` statement (read P(Yes), generate nothing). glance did not invent this. Differences: glance turns the primitive into typed answers (choice, score), calibrates it post hoc and reports calibration; VQAScore reports ranking quality, not calibration. |
| Kadavath et al., "Language Models (Mostly) Know What They Know", 2022. https://arxiv.org/abs/2207.05221 | Asks a model whether a proposed answer is true and reads P(True); finds large models well calibrated on multiple choice and true/false in the right format. | glance's candidate template ("Candidate answer: X. Is this candidate the correct answer? Yes/No") is P(True) applied per option to images. Their finding is for large text models; on a 4B VLM glance found raw P(True) far too sharp (Platt slope 0.18). |
| Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024 (spotlight). https://arxiv.org/abs/2309.03882 | Shows option-ID token bias makes lettered multiple choice sensitive to option order; proposes PriDe, a label-free prior debiasing estimated from permutations. | Explains glance's `letter` result (probability shifts up to 0.97 under reordering, even with 4 cyclic rotations). glance's alternative is to avoid option IDs altogether (`independent`), which is invariant by construction at the cost of one pass per option. A VLM-specific follow-up exists (arXiv 2509.16805, not read). |
| Q-Bench: Wu et al., ICLR 2024 (spotlight). https://arxiv.org/abs/2309.14181 | Benchmark of MLLM low-level vision. Introduces a softmax over the logits of "good" and "poor" as a zero-shot quality score; reports SRCC against human opinion on 7 IQA datasets (up to 0.541 in the wild per the summary read today). | The closest precedent for reading quality from logits without training. glance's `digits` readout generalizes the two-word softmax to a described K-level scale and adds calibration; Q-Bench measures rank correlation with human scores on real IQA data, which glance has not done. |
| Q-Align: Wu et al., "Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels", ICML 2024. https://proceedings.mlr.press/v235/wu24ah.html | Fine-tunes an LMM on rating levels (text words), then at inference takes the closed-set probabilities of the level tokens and their weighted average as the score. State of the art on IQA, aesthetics and video quality. | Same inference-time readout idea as glance's `digits` + expected score, but TRAINED on human-rated datasets. glance is the training-free counterpart with a few dozen labels per scale; it has not been compared with Q-Align on any shared benchmark. |
| DeQA-Score: You et al., "Teaching Large Language Models to Regress Accurate Image Quality Scores using Score Distribution", CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/You_Teaching_Large_Language_Models_to_Regress_Accurate_Image_Quality_Scores_CVPR_2025_paper.pdf | Trains with soft labels over level tokens so the predicted distribution matches the human score distribution. | Trained; distribution-aware. glance's matrix calibration is a tiny post-hoc version of "make the level distribution right". |
| DistortBench, arXiv 2604.19966 (April 2026). https://arxiv.org/abs/2604.19966 | 13,500 four-choice questions on 27 distortion types and 5 severity levels (KADID-10k calibrations) for 18 VLMs. Best VLM 61.9%; human majority vote 65.7%; average individual human 60.2%. | The nearest benchmark to the score lab's scales. It shows distortion judgments are hard for VLMs and for people. NOT comparable to glance's numbers: their task is a 4-choice question across 27 distortions without calibration data; glance rates the severity of one known distortion on 4 levels with a per-scale calibration fit on labels. Running glance's method on DistortBench (or directly on KADID-10k levels) is the obvious next experiment. |
| Zhang et al., "MLLMs Know Where to Look: Training-free Perception of Small Visual Details", ICLR 2025. https://arxiv.org/abs/2502.17422 | Training-free automatic cropping (from attention / gradients) improves MLLM accuracy on small details. | glance's `zoom` (a fixed, pixel-magnified centre crop as a second image) is a cruder, question-agnostic relative of this. Their attention-guided crop could replace the fixed one. |
| Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | Shows inference kernels are not batch-invariant: the same request gives numerically different outputs at different batch sizes; provides batch-invariant kernels at a throughput cost. | Exactly the effect behind glance's failed prefix-cache acceptance: on MPS float16 the reference path differs from itself by up to 0.093 in z when only its batch size changes. glance's workaround for decisions is canonical statement ordering, not invariant kernels. |

## From memory, to verify before citing

- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017: temperature, vector and matrix scaling; ECE. glance's calibrations are these, unchanged.
- Zhao et al., "Calibrate Before Use", ICML 2021, and later batch/prototypical calibration: affine correction of label-token probabilities in few-shot prompting. Closest text-LLM precedent for glance's per-level bias.
- Liu et al., "G-Eval", 2023: probability-weighted score over rating tokens for text evaluation. Text-domain precedent for the expected-score readout.
- Wang et al., "Exploring CLIP for Assessing the Look and Feel of Images" (CLIP-IQA), AAAI 2023: antonym prompt pairs with a dual encoder. Precedent for glance's SigLIP `noul` with true/false criteria.
- Compare2Score (NeurIPS 2024) and other comparison-to-anchor IQA methods: related to glance's `anchors_*` readouts, which did not help here.
- OrdinalCLIP (NeurIPS 2022), CORAL/CORN ordinal regression: related to the `cumulative` readout, which did not help here.
- Wang et al., "Large Language Models are not Fair Evaluators", and MT-Bench position-bias analyses: swapping positions and averaging, the judge-model analogue of glance's forward + reversed scale.
- Kumar et al. 2019 ("Verified Uncertainty Calibration"), Roelofs et al. 2022, Nixon et al. 2019: bias of binned ECE estimators. glance's "ECE sampling floor" is a simulation-based way of showing the same bias; a debiased estimator would be the standard alternative.
- Kwon et al. (vLLM / PagedAttention), Zheng et al. (SGLang / RadixAttention), Juravsky et al. (Hydragen): prefix sharing in LLM serving. glance's prefix cache is a small single-request version with mRoPE position continuation; Hydragen-style attention decomposition would avoid copying the prefix KV per statement.
- Whitehead et al., "Reliable Visual Question Answering", ECCV 2022: risk-coverage evaluation for VQA. Same selective-accuracy framing as glance's go/no-go row 3.
- POPE (Li et al., 2023), GQA, Oxford-IIIT Pet, Caltech-101: datasets used, see `DATASETS.md`.
- Tooling with a similar interface but no calibration: constrained-decoding libraries (Outlines, Guidance, SGLang `select`), provider structured outputs with logprobs, Hugging Face zero-shot image classification pipelines, `t2v_metrics` (VQAScore), `pyiqa` (Q-Align, CLIP-IQA).

## Honest positioning

What is NOT new: reading Yes/No or level-token logits instead of generating; independent per-option scoring; temperature,
vector and matrix scaling; position-swap debiasing; cropping to help perception; prefix caching.

What this project adds, as far as the checked literature goes:
1. One typed interface (yes/no, choice, rating) over the same logit primitive, with calibration, logging and an
   evaluation harness, measured against a frontier model on identical items (within 3.1 points on average; selective
   accuracy at 80% coverage above the frontier model's full accuracy).
2. A training-free recipe for described rating scales: digit-logit readout, forward and reversed, with and without a
   magnified crop, plus matrix calibration from about 32 labels per scale: 0.500 -> 0.867 on five synthetic scales at
   equal forward passes, selected before the test split was scored.
3. Measurement care that changes conclusions: the ECE sampling floor (the v0 yes/no "failure" disappears at n = 1,000),
   exact permutation invariance including numerical noise, and the batch-shape noise analysis of the cache.

What it lacks compared with the IQA line of work (Q-Bench, Q-Align, DeQA-Score): no standard human-opinion benchmarks
(KonIQ-10k, SPAQ, KADID-10k), no SRCC/PLCC, one model, one machine, synthetic scales only, and no comparison with
trained scorers or with classical no-reference metrics, which would solve blur and noise ladders trivially. The rating
result is a statement about asking and calibrating, not yet about image quality assessment.

## Typed-decision ("System One") models: Jev and open reproductions

Jev is a commercial API from TypeSafe AI, launched in mid-September 2026 (15 or 16 September depending on the
source), that answers typed questions (Noul, Choice, Score) against a state in one forward pass and returns typed
answers plus probabilities instead of generated text. glance's interface is modelled on it on purpose: the hand-off
(`HANDOFF.md`, first section) says "The interface mirrors TypeSafe's Jev, which is text-only today, so results are
directly comparable." Every page below was fetched and read on 2026-09-20. It is kept apart from the paper list above
because it is a product plus a fast-moving open-source ecosystem, not a paper; most sources are repositories and blog
posts, not peer-reviewed work, and should be re-checked before submission.

### Vendor claims (TypeSafe's own docs and posts; not independently checked)

| Claim | Source |
| --- | --- |
| Three primitives. Noul: "is this statement true?", returns a probability near 1/0/0.5, no confidence field. Choice: pick from up to 255 options, returns `choice`, `probabilities`, `confidence`. Score: rate on an ordered list of 2 to 10 levels, returns `score`, `legend`, `probabilities`, `confidence`. Request is a `state` (string, JSON object, or JSON array of text) plus one or more questions with `instructions` and `criteria`. | docs.typesafe.ai/introduction, docs.typesafe.ai/primitives |
| "Jev reads text only ... Images, audio and video are not supported yet." State plus the longest question must fit about 32k tokens; state plus all questions in a request about 64k. (docs.typesafe.ai/limits 404'd today; this is a third-party summary of it, not the page itself.) | flaviocopes.com/jev/ |
| Calibration method is called RLCD, "Reinforcement Learning for Calibrated Decisions": trained so that among decisions where the model states probability p, about p of them are correct, unlike RLHF (preference) or RLVR (verifiable reward). Third-party reconstruction; the article itself says no independent architecture paper exists. | explainx.ai/blog/how-does-jev-work-rlcd-system-one-model-explained-2026 |
| Latency 70 to 500 ms end to end; one demo showed 0.114 s vs 8.566 s for a compared frontier model. Price $0.042/MTok input, $0 output, claimed "193.6x faster" and "444.6x cheaper" than a compared GPT model. Architecture, parameter count and weights undisclosed; hosted API in early access behind a waitlist as of 19 Sept 2026. | theregister.com (16 Sept 2026), marktechpost.com (19 Sept 2026) |
| TypeSafe's own documented failure modes: literal reading, weak counting and date math, indirection, distraction by irrelevant state, adversarial framing, contradictory instructions, no cross-question consistency guarantee, poor generation. No abstain option, no rationale, questions in one batch cannot see each other's answers. | docs.typesafe.ai/model-jaggedness/jev-1.13, reticle.sh/blog/what-jev-cannot-do |

Neither the Register nor MarkTechPost piece independently verified the speed, cost or accuracy numbers; MarkTechPost
says the figures come from "TypeSafe's own workflow evals."

### Independent and third-party measurements (not run by TypeSafe)

| Measurement | Result | Source |
| --- | --- | --- |
| JevBench: 534 text-only decisions (easy/standard/judge/hard tiers), scored on intelligence, calibration, speed and cost. No vision tasks. | Jev 1.13.0 leads at 75.4; open reproduction "SemIf" (Qwen3.5-4B) scores 74.7. | github.com/fstandhartinger/jevbench, benchmarkheaven.com/jev-models |
| Event-listing validation, 50 held-out cases. Authors call it "a use-case study, not a general model ranking," tuned to their own prompts, so third-party but not neutral. | Jev 96% (48/50) at 0.59 s median, $0.043/1,000 decisions; Gemini 3.5 Flash-Lite 86% at 3.40 s; Mistral Small 4 84% at 2.90 s. | nearhere.events/blog/typesafe-jev-mistral-gemini-event-validation |
| Prompt-injection detection, 662 labeled messages (deepset/prompt-injections), run on a public corpus, results committed to the repo. | 96.5% accuracy, ROC-AUC 0.9927, ECE 0.0588, latency p50 325 ms. | github.com/Gaurav-Gosain/jev-sec-bench |

### Open reproductions, including two with images

None of these are affiliated with TypeSafe. None claims to reproduce Jev's undisclosed weights or exact RLCD training.

- OpenJev (AlexWortega, HF): Qwen3.5 fine-tuned as an NLI cross-encoder. v2 (4B) is trained and multimodal: ANLI
  0.42 -> 0.63, image-based claims 0.52 -> 0.84 after tuning, per its model card. Vision: yes (v2, trained).
  huggingface.co/AlexWortega/openjev
- open-jev (Dasein Labs): Gemma 3 4B on Apple silicon via MLX, zero-shot or a small trained head. Its "Doom demo"
  converts game frames to a text description before scoring; the repo states no images are accepted as input, so
  despite the demo it is text-only. Vision: no, contrary to what the demo name suggests. github.com/daseinlabs/open-jev
- jev-visual (hr98w): Qwen3.5-0.8B on Apple silicon (MLX), zero-shot, genuinely takes an image and shares one vision
  prefill across up to 64 question suffixes by forking the KV cache. Reports independent-vs-shared scoring at 64
  decisions: 37.30 s -> 2.40 s (medians); author calls it "a scaling experiment, not an accuracy evaluation."
  Vision: yes, training-free. github.com/hr98w/jev-visual
- OpenJev-Vision (IamBusy): two trained tracks, a small CNN plus prior or frozen DINOv2 features for vision, a
  Qwen3-0.6B LoRA scorer for text, prefill-once evaluation. Reports "8 questions x 4 candidates, warm median: 0.70 s"
  against one image, vs "1 question x 4 candidates: 104 ms." Vision: yes, trained. github.com/IamBusy/OpenJev-Vision

Text-only reproductions also checked directly: qwen27b-jev (single-logit read 94.8% accuracy either way; grammar-
constrained is 15% faster than free generation, plain logit read is 28% slower; github.com/sueszli/qwen27b-jev),
jev-single-decode (llama.cpp HTTP adapter, `max_tokens=1` plus renormalized logprobs, 88.40% accuracy on 10,000
samples, p50 537 ms, ECE 0.1067 on Qwen3-4B-Q4_K_M; github.com/siren2345/jev-single-decode), jev-on-a-laptop
(Qwen2.5/3, 7 to 8x speedup from parallel logit reads vs naive generation on an M5 MacBook Air; github.com/
rorshopping/jev-on-a-laptop), and Simple Jev (Hugging Face Transformers server; its own README says "text only;
images, audio, video, and tool calls are unsupported"; github.com/featherless-ai/simple-jev).

### ECE noise floor: an independent re-analysis

An open GitHub issue re-examines several Jev-adjacent benchmarks (jev-benchmark, jev-phishing-bench, jev-spam-eval,
jev-rerank-bench, openjev) and argues published ECE numbers at small n cannot be told apart from a perfectly
calibrated model's sampling noise: the noise floor is about 0.061 ECE at n = 60, 0.025 at n = 500, 0.012 at
n = 2,000, 0.004 at n = 18,514. It reframes a published n = 60 ECE of 0.0505-0.0712 as indistinguishable from
perfect calibration and proposes recomputing ECE for five repos on one consistent binning alongside each one's own
noise floor. This is the same failure mode glance's own "ECE sampling floor" targets (the v0 yes/no "failure"
disappears at n = 1,000, see Honest positioning above); this issue is independent confirmation, from a source with
no connection to glance, that the same bias affects Jev-adjacent evaluations. github.com/SamuelSacco/jev-exploration/issues/2

### Not verified

- An official `jev` branch or PR in `ggml-org/llama.cpp`: not found. The closest match is a feature request, "Fast
  Tool Gating & Single-Pass Selection via Prefill Logit Slicing" (issue #29022): prefill-logit tool routing, no
  vision, not merged, no benchmark numbers. github.com/ggml-org/llama.cpp/issues/29022
- "8 questions on one photo, 1.6 s vs 4.6 s on Qwen3.5-2B": not found despite targeted search. Closest related
  numbers: OpenJev-Vision's "8 questions x 4 candidates, warm median 0.70 s" against one image (different model,
  different numbers, no baseline ratio given) and jev-visual's 37.30 s -> 2.40 s at 64 decisions on Qwen3.5-0.8B
  (different model, different question count). Neither matches. Treat the 1.6 s / 4.6 s / Qwen3.5-2B figure as
  unconfirmed.

### How glance relates

glance reads the same three question types (yes/no probability, choice, ordered score) but works on images
natively, using a frozen open-weights vision-language model with no training of any kind, unlike the Jev ecosystem's
trained vision entries (OpenJev v2, OpenJev-Vision) and unlike Jev itself, whose docs state text only. glance's
calibration is post hoc, fit per task on a few dozen to a few hundred labels; a fit on one rating dimension does not
transfer to another, about 32 labels per scale are enough, and a single temperature is not enough for ordered
scales. On five synthetic 4-level image-quality rating scales the shipped v0 readout reached mean accuracy 0.500,
the same readout with a fitted bias and temperature reached 0.810, and a 4-pass ensemble of digit readouts with
matrix scaling (`ens4d`) reached 0.867, mean ECE about 0.03. On an Apple-silicon laptop one such rating of a fresh
image takes 1.09 s (the v0 readout: 0.44 s); when five ratings share the image prefill it is 0.58 s per rating, and
0.34 s with 25 (`docs/paper/RESULTS_LAB.md`, section 9). The multi-question-per-prefill idea that jev-visual and
OpenJev-Vision both measure is what glance's prefix cache does; packing changed none of 100 checked predictions. The
saving is smaller here than in their settings because a 196-token image is short next to a 100-token rating prompt
that has to be read once per readout.

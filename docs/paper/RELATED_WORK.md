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

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
| Q-Align: Wu et al., "Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels", ICML 2024. https://proceedings.mlr.press/v235/wu24ah.html | Fine-tunes an LMM on rating levels (text words), then at inference takes the closed-set probabilities of the level tokens and their weighted average as the score. State of the art on IQA, aesthetics and video quality. | Same inference-time readout idea as glance's `digits` + expected score, but TRAINED on human-rated datasets. glance's base VLM is frozen instead; a small readout (matrix-scaling calibration) is fit on a few dozen labels per scale. It has not been compared with Q-Align on any shared benchmark. |
| DeQA-Score: You et al., "Teaching Large Language Models to Regress Accurate Image Quality Scores using Score Distribution", CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/You_Teaching_Large_Language_Models_to_Regress_Accurate_Image_Quality_Scores_CVPR_2025_paper.pdf | Trains with soft labels over level tokens so the predicted distribution matches the human score distribution. | Trained; distribution-aware. glance's matrix calibration is a tiny post-hoc version of "make the level distribution right". |
| DistortBench, arXiv 2604.19966 (April 2026). https://arxiv.org/abs/2604.19966 | 13,500 four-choice questions on 27 distortion types and 5 severity levels (KADID-10k calibrations) for 18 VLMs. Best VLM 61.9%; human majority vote 65.7%; average individual human 60.2%. | The nearest benchmark to the score lab's scales. It shows distortion judgments are hard for VLMs and for people. NOT comparable to glance's numbers: their task is a 4-choice question across 27 distortions without calibration data; glance rates the severity of one known distortion on 4 levels with a per-scale calibration fit on labels. Running glance's method on DistortBench (or directly on KADID-10k levels) is the obvious next experiment. |
| Zhang et al., "MLLMs Know Where to Look: Training-free Perception of Small Visual Details", ICLR 2025. https://arxiv.org/abs/2502.17422 | Training-free automatic cropping (from attention / gradients) improves MLLM accuracy on small details. | glance's `zoom` (a fixed, pixel-magnified centre crop as a second image) is a cruder, question-agnostic relative of this. Their attention-guided crop could replace the fixed one. |
| Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | Shows inference kernels are not batch-invariant: the same request gives numerically different outputs at different batch sizes; provides batch-invariant kernels at a throughput cost. | Exactly the effect behind glance's failed prefix-cache acceptance: on MPS float16 the reference path differs from itself by up to 0.093 in z when only its batch size changes. glance's workaround for decisions is canonical statement ordering, not invariant kernels. |
| G-Eval: Liu, Iter, Xu, Wang, Xu, Zhu, "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment", EMNLP 2023 (main conference, pp. 2511-2522). https://arxiv.org/abs/2303.16634 | Scores text on a 1-5 (etc.) rubric by asking GPT-4 for a rating, then takes "score = sum_i p(s_i) * s_i", the probability-weighted summation over the rating-token probabilities, as the final score instead of the raw generated number. DeepEval's docs describe this as G-Eval's default scoring mode: "take the probabilities of the output tokens from the LLM to normalize the score and take their weighted summation as the final result" (https://deepeval.com/docs/metrics-llm-evals, fetched today). | Direct precedent, in the text domain, for reading score-token probabilities instead of trusting a generated number. glance's `digits` readout is the same primitive on images, plus matrix-scaling calibration fit on labels and an evaluation harness; G-Eval reports correlation with human judges on summarization/dialogue, not calibration, and uses the raw softmax weighting with no fitted map. |
| Q-Bench+: Zhang, Wu, Zhang, Zhai, Lin, "Q-Bench+: A Benchmark for Multi-modal Foundation Models on Low-level Vision from Single Images to Pairs", TPAMI (arXiv 2402.07116, Feb 2024). https://arxiv.org/abs/2402.07116 | Extends Q-Bench's good/poor softmax with a prompt ensemble (Section II-D3): positive token set {good, fine, high}, negative set {poor, bad, low}, summed into the softmax instead of one word per side. Reports up to 5% accuracy improvement, 1.7% on average (SRCC+PLCC)/2, across 7 quality datasets and the top MLLMs tested. | Confirms the reviewer's claim: this is exactly a prompt ensemble, and it is not new. glance's own ensemble (`ens4d`) varies scale direction (forward/reversed) and view (plain image / magnified crop) rather than synonym sets at a fixed prompt; the two ideas are compatible and neither has been tried combined. |
| Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017. https://arxiv.org/abs/1706.04599 | Defines temperature scaling (one scalar T), vector scaling (diagonal W) and matrix scaling (full "Wz+b" before softmax) as extensions of Platt scaling. States plainly that matrix scaling's parameter count grows quadratically with class count K, reports it performing poorly on hundreds of classes (Birds, Cars, CIFAR-100) and failing to converge on 1000-class ImageNet, and concludes "any calibration model with tens of thousands (or more) parameters will overfit to a small validation set." | glance's calibration IS matrix scaling, unchanged, on a 4K-dimensional logit feature ("W" of shape K x 4K, fit by L2-regularized NLL: `docs/paper/METHODS.md`). This is exactly the small-data overfitting regime the paper warns about, which is why glance fits the softmax temperature separately on held-out folds rather than the training fold, and why per-rubric calibrations are never pooled (see Honest positioning). |
| Kull, Perello-Nieto, Kangsepp, de Menezes e Silva Filho, Song, Flach, "Beyond Temperature Scaling: Obtaining Well-Calibrated Multi-Class Probabilities with Dirichlet Calibration", NeurIPS 2019. https://arxiv.org/abs/1910.12656 | Dirichlet calibration is matrix scaling on log-probabilities (log-transform, one linear layer, softmax). Proposes ODIR (off-diagonal and intercept regularization): penalize the off-diagonal and bias terms of that matrix more than the diagonal, specifically to fight matrix scaling's overfitting on many classes or little data ("L = logloss + lambda * mean(off-diagonal w_ij^2) + mu * mean(b_j^2)"). | A more targeted regularizer than glance's flat L2 penalty on the whole W and b. glance has not tried ODIR-style differential regularization on its 4K-dimensional matrix; it is a plausible next step, not something already done here. |
| CLIP-IQA: Wang, Chan, Loy, "Exploring CLIP for Assessing the Look and Feel of Images", AAAI 2023. https://arxiv.org/abs/2207.12396 | Frozen CLIP, antonym prompt pairs ("Good photo." / "Bad photo."), softmax over the two cosine similarities as the score. Swaps in other antonym pairs for abstract attributes: brightness ("Bright"/"Dark"), noisiness ("Clean"/"Noisy"), sharpness ("Sharp"/"Blurry"), colorfulness ("Colorful"/"Dull"). | Precedent for frozen-encoder, prompt-pair attribute scoring with no labels at all. glance differs in three ways: a VLM that attends over image and text jointly rather than a dual-encoder cosine similarity; a described K-level digit scale rather than a 2-way antonym softmax; and a fitted matrix-scaling map rather than a raw softmax. CLIP-IQA needs zero labels; glance needs a few dozen per rubric, which is a real cost CLIP-IQA does not have. |
| Dog-IQA: Liu, Zhang, Li, Pei, Song, Liu, Kong, Zhang, "Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessment", arXiv 2410.02505 (Oct 2024). https://arxiv.org/abs/2410.02505 | Training-free MLLM IQA with explicit rating standards written into the prompt, plus local (segment-level) and global (whole-image) analysis aggregated into one score. The MLLM is prompted to "score in [1, 2, ..., 7]" and the number is read off its generated text; the paper reports under 0.1% of outputs come back as words instead of a number, handled by a fallback. | Not a probability-reading precedent, despite being training-free: Dog-IQA generates the score as text, the exact free-generation approach glance's `digits` readout is built to avoid. Its local/global segment aggregation is a different idea from anything glance does, which scores one image (or image plus one fixed centre crop) per rating. |
| McCullagh, "Regression Models for Ordinal Data", Journal of the Royal Statistical Society: Series B (Methodological), vol. 42(2), 1980, pp. 109-127. https://academic.oup.com/jrsssb/article/42/2/109/7027621 | Introduces the proportional-odds (cumulative-link) model for ordinal regression. | Background for the cumulative-link idea behind the `cumulative` readout glance tried (see OrdinalCLIP/CORAL/CORN below), which did not help here. |
| Alain & Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes", arXiv 1610.01644 (2016, rev. 2018); and Hu, Ding, Wang, Liu, Wang, Li, Wu, Sun, "Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification", ACL 2022. https://arxiv.org/abs/1610.01644 , https://arxiv.org/abs/2108.02035 | Alain & Bengio: train a small linear classifier on a frozen network's intermediate activations to read out what is already represented, without touching the network's weights. Hu et al.: the "verbalizer" (the mapping from class labels to output vocabulary words in prompt-based classification) can be built or expanded from a knowledge base instead of hand-picked. | Both are the same shape of idea as glance's calibration: fit a small readout on top of a frozen model instead of touching its weights. Neither is about VLMs, ordinal scales, or images. glance's W/b matrix is closer to a linear probe (it reads a 4K-dimensional logit vector) than to a verbalizer (which picks single label words); we found no VLM-specific ordinal-rubric verbalizer paper to cite instead. |
| vLLM, "Automatic Prefix Caching", project docs. https://docs.vllm.ai/en/stable/design/prefix_caching/ | Standard serving optimization: KV-cache blocks are content-hashed and reused across requests sharing a prefix. For multimodal inputs, the frontend image processor's hash is folded into the block hash as an extra key, "we encode the image hash generated by the frontend image processor," so the cache tells two different images apart even when surrounding text tokens match. | Confirms glance's prefix cache is infrastructure, not a contribution: vLLM (and Kwon et al./Zheng et al./Juravsky et al. below) do this in production for many concurrent requests and images. What this doc does not measure, and glance did: exact prediction invariance under packing (0 of 100 predictions changed, `docs/paper/RESULTS_LAB.md` section 9) and the wall-clock cost of one rating with and without a shared image prefill. |
| VL-Calibration: Xiao, Xu, Gan, "VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning", ACL 2026 (arXiv 2604.09529, April 2026). https://arxiv.org/abs/2604.09529 | Calibrates VLM reasoning confidence on Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-30B and InternVL3.5-4B-MPO; reports ECE falling from 0.421 to 0.098 on the 4B model while accuracy improves, and similar gains at 8B and 30B. | A reviewer named this as prior VLM-calibration work on the same base-model family; confirmed, it exists and does use Qwen3-VL-4B. Different problem: it calibrates whether a multi-step reasoning answer is correct on math/knowledge benchmarks (DynaMath, MathVerse, MMMU-Pro), not an ordinal rating scale, and its calibration method is decoupled confidence estimation, not matrix scaling on rating-token logits. |

## From memory, to verify before citing

- Zhao et al., "Calibrate Before Use", ICML 2021, and later batch/prototypical calibration: affine correction of label-token probabilities in few-shot prompting. Closest text-LLM precedent for glance's per-level bias.
- Compare2Score (NeurIPS 2024) and other comparison-to-anchor IQA methods: related to glance's `anchors_*` readouts, which did not help here.
- OrdinalCLIP (NeurIPS 2022), CORAL/CORN ordinal regression: related to the `cumulative` readout, which did not help here.
- Wang et al., "Large Language Models are not Fair Evaluators", and MT-Bench position-bias analyses: swapping positions and averaging, the judge-model analogue of glance's forward + reversed scale.
- Kumar et al. 2019 ("Verified Uncertainty Calibration"), Roelofs et al. 2022, Nixon et al. 2019: bias of binned ECE estimators. glance's "ECE sampling floor" is a simulation-based way of showing the same bias; a debiased estimator would be the standard alternative.
- Kwon et al. (vLLM / PagedAttention), Zheng et al. (SGLang / RadixAttention), Juravsky et al. (Hydragen): prefix sharing in LLM serving. glance's prefix cache is a small single-request version with mRoPE position continuation; Hydragen-style attention decomposition would avoid copying the prefix KV per statement.
- Whitehead et al., "Reliable Visual Question Answering", ECCV 2022: risk-coverage evaluation for VQA. Same selective-accuracy framing as glance's go/no-go row 3.
- POPE (Li et al., 2023), GQA, Oxford-IIIT Pet, Caltech-101: datasets used, see `DATASETS.md`.
- Tooling with a similar interface but no calibration: constrained-decoding libraries (Outlines, Guidance, SGLang `select`), provider structured outputs with logprobs, Hugging Face zero-shot image classification pipelines, `t2v_metrics` (VQAScore), `pyiqa` (Q-Align, CLIP-IQA).

## Honest positioning

### What is not new, stated plainly

- Token-probability scoring instead of generating an answer: G-Eval (rating tokens, text) and Q-Bench (good/poor
  tokens, images).
- Prompt ensembling to steady that readout: Q-Bench+.
- Matrix scaling as the calibration map, and its tendency to overfit with little data: Guo et al.
- Frozen-encoder attribute scoring from antonym or described prompts, no labels needed: CLIP-IQA.
- Sharing an image prefill across requests: vLLM's automatic prefix caching (and Kwon et al./Zheng et al./Juravsky
  et al. below).

All five predate this project. Also not new on their own: independent per-option scoring; position-swap debiasing;
cropping to help perception; the base model being frozen while a small map is fit on top (linear probes,
verbalizers).

### What this project adds, as far as the checked literature goes

1. The composition for K-level ordinal rubrics on a frozen VLM: a digit readout counter-biased by asking the scale
   forward and reversed, a magnified-crop view alongside the plain image, and a matrix-scaling map fit per rubric on
   labeled examples. The base VLM is never updated; only this small readout is fit.
2. A pre-registered measurement of how much each step is worth, on five synthetic 4-level scales: mean accuracy
   0.500 with the shipped v0 readout, 0.810 with a fitted bias and temperature on that same readout, 0.867 with the
   four-pass ensemble and matrix scaling (`ens4d`).
3. The small-n probability failure and its fix: fitting the softmax temperature on the same data used to fit the
   matrix `W` and bias `b` makes the model overconfident once labels are scarce (NLL 0.749 vs. 0.478 at 32 labels
   per rubric); estimating that temperature on held-out folds instead removes the effect, and the gap closes by
   about 128 labels (`docs/paper/METHODS.md`).
4. The finding that calibrations do not transfer between rubrics: a matrix fit on one rating dimension does not work
   on another, so there is no pooled fallback (`docs/paper/RESULTS_LAB.md` section 5).
5. The packing-invariance check: packing several rating questions into one request against a shared image prefix
   changed 0 of 100 checked predictions (`docs/paper/RESULTS_LAB.md` section 9). This is worth stating because a
   related open project found in-sequence question packing moves 6-9% of answers (see "How glance relates" below).
6. Published errata: the project's own latency and small-n sharpness claims were wrong at one point and were
   corrected in the open (`docs/BRIEFING.md`) rather than left standing.
7. Also carried over from the "Verified today" table: one typed interface (yes/no, choice, rating) over the same
   logit primitive, with calibration, logging and an evaluation harness, measured against a frontier model on
   identical items (within 3.1 points on average; selective accuracy at 80% coverage above the frontier model's full
   accuracy); the ECE sampling floor (the v0 yes/no "failure" disappears at n = 1,000); exact permutation invariance
   including numerical noise; and the batch-shape noise analysis of the cache.

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

### Open reproductions, including several with images

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
- LitJev (zhengxuyu): serves Jev's own `/v1/systemone` request/response schema (choice, score, noul) from a local
  Hugging Face checkpoint, reading option logits instead of generating. README says "the full Qwen family is
  supported (Qwen3.x text and vision checkpoints, any size)"; its own benchmark section pairs MMLU-Pro direct-answer
  scoring with "Doom and chess played from screenshots." Vision: yes, when a vision checkpoint is loaded; no default
  checkpoint is pinned in the README. github.com/zhengxuyu/litjev
- mini-jev (r-ms): frozen Qwen3-4B, text-only. Turns each field into a lettered multiple-choice question and reads
  the chosen letter's logit at the answer position instead of generating; the text's KV cache is reused across a
  request's fields. Preregistered (`PREREG.md`, amendments v1.1-v1.3, "every number below is recomputed from the
  stored run records"). CLINC150 intent classification, 6,750 paired observations: JSON 0.909 accuracy vs.
  letter-reading 0.907 (-0.22 pp), 4x faster than JSON on 32-token text via the shared prefix, 1.4-2.4x on
  2048-token text. Vision: no. github.com/r-ms/mini-jev
- open-alternative-jev (ikermoel): typed, calibrated decisions from any open-weights LLM (HF or vLLM backend),
  text-only, one forward pass. Two packing modes: "packed" puts every question for one state into one attention
  sequence (fastest); "separate" gives each question its own sequence over the same state ("no interference between
  questions"). The README reports packing changes 6-9% of answers relative to padding-only noise (2.7%), and that
  reordering the questions changes 8% of MMLU answers and 2.4% of RACE-H answers; a fitted temperature brings MMLU
  ECE from 5.4% to 2.1%. Vision: no. github.com/ikermoel/open-alternative-jev

Added 2026-09-20 evening (each fetched and read that day, after two outside reviews named them):

- YOFO, "You Only Forward Once: An Efficient Compositional Judging Paradigm" (Zhang et al., arXiv:2511.16600, v3
  2026-02-02). TRAINS Qwen2-VL-2B-Instruct and Qwen3-VL-2B-Instruct with LoRA (vision encoder frozen, 1.2M SA-1B images,
  cross-entropy on the answer positions) so that a template of N requirements is judged in ONE forward pass: the
  yes/no logits are read at the position before each requirement's answer slot. Binary requirements only; evaluated on
  fashion reranking (LAION-RVS-Fashion); no calibration; the paper names no released checkpoint (a reviewer's
  "YOFO-Qwen3-VL-2B-Instruct" download could not be confirmed). Vision: yes, trained.
- Laya Vision (independent fork, "not affiliated with Convai Innovations"): SmolVLM-256M-Instruct replaces Laya's text
  encoder; "the vision tower is frozen, and the language model and decision head are trained, about 150M parameters";
  about 72k examples (A-OKVQA, ScienceQA images, VQAv2 yes/no) with Laya's RLCD objective ("a policy gradient on
  strictly proper scoring rules"); decisions come from a dedicated head, not vocabulary tokens; per-type temperatures
  (choice 3.15, noul 1.56 on the card as fetched); `score` is untrained ("their outputs are meaningless"); own
  validation: 75.9% accuracy, ECE 0.035 (VQAv2 yes/no 73.2%). Code Apache-2.0, WEIGHTS CC BY-NC-SA 4.0, so it cannot be
  run under this project's license rule. github.com/r33drichards/laya-vision, huggingface.co/thaitea/laya-vision-smolvlm-256m
- OpenJev on DiffusionGemma (razorback16): a server that "speaks the same wire API as TypeSafe's Jev", running
  DiffusionGemma 26B-A4B with one read-only diffusion pass whose distribution over the answer tokens is the answer;
  accepts up to 8 images per request as an extension; depends on an unmerged vLLM pull request (vllm-project/vllm#57250).
  Apache-2.0. Not runnable on this project's laptop. A different project from AlexWortega's OpenJev above.
  github.com/razorback16/openjev. LocalJev (kexi) is wire-compatible too but asks the model to WRITE its probabilities
  as JSON, which is self-report, not a logit read. github.com/kexi/localjev

Two more third-party entries are not Qwen reproductions at all, but small encoders trained from scratch for typed
decisions, text-only, no image input: Laya (Convai Innovations), 421M parameters, Apache 2.0, non-autoregressive,
ModernBERT-large (395M) backbone plus a small decision head for choice/score/noul with calibrated probabilities (a
322M multilingual variant uses mmBERT-base), huggingface.co/convaiinnovations/laya; and openJev-verdict-2.0
(Heman10x-NGU), a 149.6M-parameter non-autoregressive ModernBERT-base decision engine, about 20-25 ms per decision,
reporting 77.10% accuracy / 0.0636 Brier / 1.44% ECE (confidence head) on a `LocalLLaMA/typed-decisions` set of
2,000 held-out decisions against Laya's 76.60% / 0.0660 and a claimed Jev 72.70%, github.com/Heman10x-NGU/openJev-verdict-2.0.

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

### Where Glance sits: positioning statements (2026-09-20 evening, owner's zero-shot framing, STATUS D44)

| | Glance | jev-visual, LitJev (nearest neighbours) | Simple Jev | YOFO | Laya Vision | TypeSafe Jev |
| --- | --- | --- | --- | --- | --- | --- |
| What it is | an ask-layer (library, CLI, server) around an open VLM | ask-layers around an open VLM | an ask-layer around any open LM | fine-tuned Qwen-VL 2B checkpoints | a trained 256M decision model | a hosted proprietary model |
| Trains weights? | no | no | no | yes (LoRA) | yes (LM + decision head) | yes |
| Images | yes, first | yes | no ("text only") | yes | yes | no (docs: text only) |
| Question types | yes/no, pick-one, ratings | choice, score, noul | choice, score, noul | yes/no per requirement only | choice, noul (`score` untrained) | choice, score, noul |
| How the answer is taken | logits of the allowed answer tokens at a forced answer position | the same | the same | yes/no logits at N positions of one packed template | a dedicated decision head | not disclosed |
| Calibration | commands: `glance fit --unlabeled`, `glance fit`; generic calibration for yes/no and pick-one | none reported | "not calibrated probabilities of correctness" | none | temperature after proper-scoring-rule training | claimed by the vendor |
| Several questions per image | one image prefill, independent branches (invariant by construction) | shared vision prefill | shared text prefix | one packed sequence | one image encode per call | n/a |
| Backbone | yours (Qwen3-VL-4B measured; 2B / 8B and SmolVLM2 in progress) | yours | yours | fixed | fixed | fixed |
| Credit line | "Qwen3-VL-4B + Glance" | - | "<model> + Simple Jev" | "YOFO" | "laya-vision-smolvlm-256m" | "Jev" |

Statements we stand behind, in this order:
1. **Same class as the training-free ask-layers.** Simple Jev (text), jev-visual and LitJev (images) and Glance all wrap
   a frozen generative model, force it to the answer position, read the allowed tokens' logits and reuse the shared
   prefix. For yes/no and pick-one, Glance's forward pass is NOT new and this repository says so. An outside review's
   phrase is fair: Glance is that readout pointed at photos, with calibration commands and a frontier-VLM scoreboard.
2. **Not the same class as YOFO, Laya Vision, OpenJev v2 or OpenJev-Vision.** Those train something. The trade is real
   in both directions: a trained head can be calibrated by construction and cannot emit prose, but it is tied to its
   backbone (a 256M specialist cannot borrow a 4B general model's perception); Glance's ceiling is whatever open VLM you
   attach, and it moves when a better open VLM ships, with no retraining.
3. **What is Glance's own, and only as far as it is measured:** (a) vision-first evidence: photos taken after every
   model's release with labels nobody here made, the same items sent to three frontier VLMs, the same VLM writing its
   answer against reading it, every experiment registered before it ran; (b) ratings: the elicitation (digits forward
   and reversed, with and without a magnified crop) and calibration as verbs (unlabeled self-calibration, labeled fit),
   printed with the caveat that Gemini 3.1 Pro leads by 8 points on zero-shot exact levels; (c) dollars and
   milliseconds against hosted VLMs on photo tasks; (d) the credit line; (e) portability across backbones: a plan until
   the second family (E3) and the size ladder (E15) are in the table.
4. **Lead with the ask.** "Glance is how you ask an open vision-language model for a typed decision, plus `fit`."
   "Decision engine" describes the open model being read, never Glance; led with as a product noun it files Glance
   under the trained models above. "Jev for vision" stays banned: the Jev-shaped servers with image input already
   exist (jev-visual, LitJev, OpenJev on DiffusionGemma), and Glance's difference from them is measurement and `fit`,
   not the interface.

Corrections to the relayed reviews, so their errors do not enter this repository: Simple Jev has no vision path (its
README: "images, audio, video, and tool calls are unsupported"), so "Simple Jev (vision)" is not a system; the
training-free image neighbours are jev-visual and LitJev. Jev is a typed decision API (choice, score, noul), not an
"action and control space" paradigm; the DiffusionGemma and patched-vLLM details belong to razorback16's OpenJev, not to
Jev. Glance's fits are matrix scaling on member logits and z-scoring over unlabeled images, not Platt or isotonic
scaling. A released YOFO checkpoint could not be confirmed.

### How glance relates

glance reads the same three question types (yes/no probability, choice, ordered score) but works on images
natively. The base VLM is frozen; unlike the Jev ecosystem's trained vision entries (OpenJev v2, OpenJev-Vision,
which fine-tune or train a scorer on top of vision features) and unlike Jev itself (whose docs state text only),
glance trains nothing beyond a small per-rubric readout fit on labeled examples. glance's calibration is post hoc,
fit per task on a few dozen to a few hundred labels; a fit on one rating dimension does not transfer to another,
about 32 labels per scale are enough, and a single temperature is not enough for ordered scales. On five synthetic
4-level image-quality rating scales the shipped v0 readout reached mean accuracy 0.500, the same readout with a
fitted bias and temperature reached 0.810, and a 4-pass ensemble of digit readouts with matrix scaling (`ens4d`)
reached 0.867, mean ECE about 0.03. On an Apple-silicon laptop one such rating of a fresh image takes 1.09 s (the v0
readout: 0.44 s); when five ratings share the image prefill it is 0.58 s per rating, and 0.34 s with 25
(`docs/paper/RESULTS_LAB.md`, section 9).

The multi-question-per-prefill idea that jev-visual, OpenJev-Vision and mini-jev all measure is what glance's prefix
cache does, and it is not a glance contribution (see vLLM's own docs above, which do this for any request shape).
The saving is smaller here than in their settings because a 196-token image is short next to a 100-token rating
prompt that has to be read once per readout. What glance measured that they did not is exact prediction invariance:
packing changed 0 of 100 checked predictions in glance's lab, because each packed rating question is sent as its own
sequence that reuses the image prefix's KV cache: the questions never see each other, or each other's answers, in
the attention pattern. That is a different mechanism from open-alternative-jev's "packed" mode, which puts several
questions inside one shared attention sequence and reports that this changes 6-9% of answers versus answering them
separately. Independent branches off one cached prefix and several questions packed into one sequence are not the
same claim, and only the first is invariant by construction.

## Naming

A reviewer flagged that `pip install glance` already installs something else. Confirmed by reading
https://pypi.org/project/glance/ today: PyPI's `glance` (v32.0.0, released Apr 1, 2026) is OpenStack's Glance Image
Service, "an OpenStack project that provides services and associated libraries to store, browse, share, distribute
and manage bootable disk images." Unrelated to this project. `glance-vlm` is unclaimed on PyPI (404 as of today).
This project does not publish to PyPI yet; a different package name will be needed before it does.

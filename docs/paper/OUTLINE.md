# Paper outline

This is a skeleton for a paper written directly from this repository's own records. Every claim in the draft
abstract and contributions list is traceable to a number in `docs/paper/RESULTS_V0.md` or `STATUS.md`. The
frontier baseline (`anthropic/claude-opus-5`) has completed, so the abstract and contributions below use its
real numbers rather than a placeholder. Nothing here should be read as a finished argument; it is a scaffold
to fill in once the related-work TODOs are researched.

## Working title options

1. "Logit Readout and Post-Hoc Calibration for Typed Visual Question Answering: A v0 Evaluation"
2. "Is It a Calibration Problem or a Data Problem? Probing an Open VLM with Yes/No Statement Logits"
3. "`glance`: A Generation-Free Harness for Typed Image Decisions"

## Draft abstract (150-200 words; word count below)

> We ask whether reading yes/no statement logits from single forward passes of an Apache-2.0 vision-language
> model (Qwen3-VL-4B-Instruct), with post-hoc calibration (Platt scaling for binary questions, temperature
> scaling otherwise), gets close enough to a frontier model that further work is a calibration problem and not
> a data problem. Our harness scores each candidate answer in one forward pass, with no generation, for three
> question formats (binary, multiple-choice, ordinal scale), evaluated on five public suites against a frontier
> baseline (Claude Opus 5). Scoring each option independently is on par with a lettered multiple-choice prompt
> on the same options (0.912 vs. 0.904 and 0.973 vs. 0.977) while being exactly invariant to option order (0.0
> vs. up to 0.97 maximum probability shift). The open model trails the frontier baseline by 3.1 accuracy points
> macro-averaged over suites, and its confidence is usable for gating: selective accuracy at 80% coverage
> (0.841) exceeds the baseline's full-coverage accuracy (0.818). Calibration cuts expected calibration error by
> 34-54% on binary and multiple-choice suites, yet three of five suites still exceed a 0.05 ECE gate at our
> sample sizes. A single temperature cannot move misplaced ordinal boundaries, a weakness the frontier baseline
> shares (0.536 on the same task). We release the harness, manifests, calibration artifacts and reports.

Word count: approximately 200. Every number is copied from
`docs/paper/RESULTS_V0.md`'s M5 section, sourced to `results/v0/m5_full_eval/metrics.json` and `STATUS.md`'s
"Update 2026-09-20: frontier baseline added to the full evaluation" section.

## Candidate contributions, each linked to its evidence

1. **A 4B open, Apache-2.0 model read out via yes/no logits comes within 3.1 accuracy points of a frontier
   model (Claude Opus 5), macro-averaged over five suites, and its calibrated confidence is already usable for
   gating** (selective accuracy at 80% coverage, 0.841, exceeds the frontier baseline's own full-coverage
   accuracy, 0.818). The dual encoder (SigLIP2, 375M parameters) separately beats the frontier baseline
   outright on fine-grained pet-breed classification (0.956 vs. 0.932). Evidence: `docs/paper/RESULTS_V0.md`
   M5 "Frontier baseline addition" and "Per-suite comparison against the frontier baseline" sections;
   `results/v0/m5_full_eval/metrics.json: baseline, go_no_go`.
2. **A generation-free scoring primitive** (one forward pass per candidate statement, logits read at a fixed
   token position, no decoding) that assembles three typed answer formats from the same readout. Evidence:
   `docs/paper/METHODS.md` sections 1-2; implementation `glance/backends/vlm_hf.py`, `glance/scorer.py`.
3. **An exactly permutation-invariant multiple-choice method** (`independent`), verified at max `|Δp| =
   0.0e+00` across every `independent` suite/backend unit in the full run, versus up to `9.7e-01` for a
   per-prompt multiple-choice baseline (`letter`) even with 4 cyclic rotations. Evidence:
   `docs/paper/RESULTS_V0.md` M5 "`independent` vs `letter`" section; `STATUS.md` Final summary; mechanism in
   `docs/paper/METHODS.md` section 5 (canonical statement ordering, D12).
4. **Measured effect of post-hoc calibration by question type**: cuts ECE by 34-54% for binary and
   multi-choice questions (e.g. `pope` 0.114 → 0.066, `caltech101` 0.052 → 0.034) but cannot repair ordinal
   `score` questions (`blur_ladder` stays at ECE 0.149 after a single temperature) — and the ordinal task
   itself may be intrinsically hard, since the frontier baseline scores only 0.536 on it. Evidence:
   `docs/paper/RESULTS_V0.md` M5 "Calibration gain" and M4 calibration table; calibration artifacts
   `calibration/*.json`.
5. **An explicit accounting for small-sample ECE bias** (the "ECE sampling floor", D23): equal-mass ECE with
   15 bins is upward-biased at a few hundred test items by an amount comparable to the 0.05 gate itself;
   reporting the floor next to each ECE changes which suites can be judged to have passed or failed. Evidence:
   `docs/paper/METHODS.md` section 10; `docs/paper/RESULTS_V0.md` M5 go/no-go table (floors 0.040-0.097
   against a 0.05 gate).
6. **A working prefix-cache design for statement-batched VLM inference** (shared-prefix KV cache reuse with
   explicit mRoPE position continuation) that gives a measured 3.75x eval-time speedup, together with a
   negative result: its accuracy-preservation acceptance criterion was not met on this hardware, and the
   residual drift is smaller than the reference path's own batch-shape noise. Evidence:
   `docs/paper/METHODS.md` section 7; `docs/paper/RESULTS_V0.md` "Prefix cache acceptance check" and
   "Negative results" table; `results/v0/prefix_cache_acceptance.json`.
7. **Three targeted stretch findings**: (a) rendered-text prompt injection flips 5.2% of the POPE
   negatives the model had answered No on (5 of 97) and shifts logits by a mean of +6.4 even when the decision holds; (b) an image-token budget of
   128 (vs. 768) costs about 1.5-2 accuracy points and cuts the p50 latency of the 1 image + 5 questions benchmark from
   8,604 ms to 1,920 ms on the reference path; (c) an `other` catch-all
   option fails as a candidate statement for open-set detection (AUROC 0.655) while the same signal is
   recoverable from the raw per-option logits without training (AUROC 0.977 for `-max z` over listed
   options). Evidence: `docs/paper/RESULTS_V0.md` "Stretch experiments" section.

## Section outline with figures/tables

1. **Introduction.** Motivation (frontier-model cost/latency/data-control vs. open-model calibration
   feasibility), the three question types, the go/no-go framing from `HANDOFF.md` section 1.
2. **Method** (maps to `docs/paper/METHODS.md`).
   - 2.1 Statement primitive and logit readout (float32 head, `off_mass`).
   - 2.2 Assembly into `noul`/`choice`/`score` answers (sigmoid, softmax with temperature, expected score,
     entropy confidence).
   - 2.3 `independent` vs. `letter`, canonical ordering, exact permutation invariance.
     Figure candidate: a schematic of the canonical-ordering mechanism (not a data plot; would need to be
     drawn, no source image exists yet).
   - 2.4 Dual-encoder baseline (SigLIP2).
   - 2.5 Prefix cache and mRoPE position continuation.
   - 2.6 Post-hoc calibration (Platt / temperature, pooled vs. per-suite).
3. **Evaluation protocol** (maps to `docs/paper/METHODS.md` section 9, `docs/paper/REPRODUCE.md`). Suites,
   licensing constraints, seeded split, metrics, ECE sampling floor.
4. **Results** (maps to `docs/paper/RESULTS_V0.md`).
   - 4.1 Milestone checks M0-M4 (uncalibrated and calibrated). Table: per-suite accuracy/ECE at M3 (n=200/50)
     and M4 (n=80). Figure candidates: reliability and risk-coverage plots, one pair per suite/backend/method,
     e.g. `results/v0/m4_calibrated_n80/plots/pope__vlm__statement__reliability.png` and
     `..._risk_coverage.png` (10 suite/backend/method combinations available in that directory).
   - 4.2 Full evaluation (M5): go/no-go table, per-suite accuracy/ECE/NLL/Brier, `independent` vs `letter`,
     latency (reference vs. prefix-cached), and the completed frontier-baseline comparison (accuracy gap +3.1
     points macro-averaged, selective accuracy 0.841 vs. baseline full-coverage accuracy 0.818, per-suite table
     in `docs/paper/RESULTS_V0.md`). Figure candidates: the 11 reliability/risk-coverage plot pairs (22 files)
     listed in `results/v0/m5_full_eval/report.md`'s "Plots" section, e.g.
     `results/v0/m5_full_eval/plots/pope__vlm__statement__reliability.png`.
   - 4.3 Prefix cache acceptance and its negative result.
   - 4.4 Offline per-level-bias analysis on `blur_ladder` (bridges to the score-lab follow-up work).
   - 4.5 Stretch experiments: injection, token-budget sweep, open set. Figure candidates: token-budget sweep
     could become a line plot of accuracy and p50 latency vs. budget (source numbers in
     `docs/paper/RESULTS_V0.md` "Image-token sweep" table; no plot file currently exists for this — would need
     to be newly drawn from the table). Reliability/risk-coverage plots exist per stretch run under
     `results/v0/stretch/*/plots/`.
5. **Negative results and design decisions that did not survive contact** (maps to `docs/paper/RESULTS_V0.md`
   "Negative results" table and the D-numbered decisions in `STATUS.md`).
6. **Discussion.** Per-question-type verdict (`choice` works, `noul` partial, `score` does not); what this
   implies for a v1 system's training-data priorities (`STATUS.md` "recommended v1 data").
7. **Threats to validity** (below).
8. **Related work** (`TODO(related work)`, below).
9. **Conclusion.**

## Threats to validity

- **Public benchmarks are likely present in training data.** POPE, GQA, Oxford-IIIT Pet, and Caltech-101 are
  widely used pretraining/fine-tuning sources; measured accuracy on them may not reflect performance on truly
  novel images. `DATASETS.md` and `STATUS.md` both flag this and identify `human_gold` (private, locally
  labeled) as the only uncontaminated check — and `human_gold` was empty in every v0 run
  (`STATUS.md` M3 section; `docs/paper/RESULTS_V0.md` M5 deviation list, item 5).
- **Small n relative to the ECE sampling floor.** At M4's n=80 (40 test items/suite) and even at M5's full
  scale, the equal-mass 15-bin ECE estimator has a measurable small-sample upward bias (0.04-0.10 depending on
  suite and confidence distribution) that is comparable to the 0.05 go/no-go gate itself
  (`docs/paper/METHODS.md` section 10, D23). Any ECE-based claim in this paper must be read against its
  reported floor, not in isolation.
- **Single model.** All local-backend numbers are from one VLM (`Qwen/Qwen3-VL-4B-Instruct`) and one dual
  encoder (`google/siglip2-base-patch16-256`) at one pinned revision each. Results may not generalize to
  other model sizes in the same family (`Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct` were pinned
  but never run, `MODELS.md`) or to newer releases in the family noted but not used
  (`Qwen/Qwen3.5-2B/4B/9B`, February 2026, `STATUS.md` "What the result means").
- **Single machine.** All measured latency, throughput, and the prefix-cache acceptance numbers are from one
  Apple M5 / 32 GB / MPS machine (`docs/paper/METHODS.md` section 12). Float16 batch-shape noise on MPS is
  itself hardware- and backend-specific (`docs/paper/REPRODUCE.md` section 6); CUDA behavior is untested in
  this repository.
- **Synthetic score suite.** The only `score`-type suite, `blur_ladder`, is a synthetic Gaussian-blur ladder
  derived from Caltech-101 images, not a naturally occurring ordinal-quality dataset. Its systematic
  one-step-too-blurry boundary error (`docs/paper/RESULTS_V0.md` M5 "Weakest suite") may be specific to this
  synthetic construction rather than representative of ordinal visual-quality judgments generally.
- **No `human_gold` data.** The one suite designed to be free of training-data contamination and free of
  public-benchmark artifacts was never populated in any v0 run (`STATUS.md`, "Open inputs from you", item 2).
  Every accuracy and calibration number in this paper is therefore on data the evaluated models may have seen
  during their own training.

## `TODO(related work)`

Topics to research and cite properly before this section is written (no citations are invented here):

- Calibration of large language/vision-language models (temperature scaling, Platt scaling, and their known
  failure modes on modern high-capacity models).
- Multiple-choice question answering via log-likelihood/logit readout of individual options vs. joint
  multi-option prompting, and known order-sensitivity effects in the latter.
- Expected calibration error estimators and their small-sample bias; alternatives to equal-mass binning.
- Selective prediction / risk-coverage curves and selective classification with a reject option.
- Prompt injection and adversarial text rendered into images for vision-language models.
- Zero-shot open-set recognition without a trained "unknown" class or threshold.
- Dual-encoder (CLIP/SigLIP-family) vs. decoder VLM approaches to visual classification.
- Ordinal regression / ordinal classification methods, and their adaptation to prompt-based zero-shot
  settings.
- KV-cache reuse and prefix caching for batched LLM/VLM inference, and multimodal rotary position embeddings
  (mRoPE) as used in the Qwen-VL model family.
- POPE, GQA, Oxford-IIIT Pet, and Caltech-101 as benchmark datasets: original papers and known limitations.
- Data contamination in vision-language model pretraining corpora relative to common evaluation benchmarks.

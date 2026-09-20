# Methods

This document describes the `glance` v0 harness in enough detail to re-implement it. Every claim below is
sourced from the code or spec files in this repository; file paths are given so each statement can be checked
against its source. Formulas are copied from the implementation, not paraphrased from memory.

## 1. Problem statement and the three question types

`glance` answers typed questions about one or more images by reading raw logits from a single forward pass of
a vision-capable model. It does not generate text and does not train anything (`HANDOFF.md` sections 1-2). The
question this v0 build was built to answer: does logit readout plus post-hoc calibration on an off-the-shelf
open VLM get close enough to a frontier VLM that a v1 system is a calibration problem rather than a data
problem (`HANDOFF.md` section 1).

Three question types (`HANDOFF.md` section 5, implemented in `glance/schema.py:50-93`):

| Type | Meaning | `criteria` shape |
| --- | --- | --- |
| `noul` | A single yes/no judgment, returned as a probability | optional `{"true": text, "false": text}` |
| `choice` | Pick one of 2-128 named options | required `{key: description_or_null, ...}` |
| `score` | Pick a level on an ordered scale of 2-10 levels | required ordered list, low to high |

Limits are enforced in `glance/schema.py:9-12`: `MAX_IMAGES = 4`, `MAX_OPTIONS = 128`,
`MIN_SCORE_LEVELS = 2`, `MAX_SCORE_LEVELS = 10`. Image limits (`configs/default.yaml` `limits:` block, read by
`glance/images.py`): 20 MB per image, formats JPEG/PNG/WEBP, resized so the longest side is at most 2048 px
(`glance/images.py:103-104`).

## 2. The statement primitive and logit readout

Every answer is assembled from one primitive: **the logit that a statement is true of the image**
(`HANDOFF.md` section 6). One statement is one forward pass, read at a single token position. Nothing is
decoded.

### VLM readout (`glance/backends/vlm_hf.py`)

- User content order: image(s), then `context` JSON, then the statement block. The assistant turn is opened
  and left empty (`render_prompt`, `glance/backends/vlm_hf.py:115-129`). The code asserts the rendered prompt
  ends with `<|im_start|>assistant\n` and contains no `<think>`/`</think>` tags, raising `BackendError`
  otherwise (`vlm_hf.py:125-128`).
- Next-token logits are read at the final position. Four token-string variants are tried for "yes" and "no"
  (`glance/prompts.py:35-36`):
  `YES_VARIANTS = ["Yes", " Yes", "yes", " yes"]`, `NO_VARIANTS = ["No", " No", "no", " no"]`. Only variants
  that tokenize to exactly one token for the loaded model are kept (`vlm_hf.py:105-113`,
  `_single_token_ids`).
- `z_yes = logsumexp(logit_i for i in yes-variant-token-ids)`, `z_no` likewise for the no-variant ids
  (`vlm_hf.py:339-340`, using `scipy.special.logsumexp`). The statement logit is `z = z_yes - z_no`
  (`vlm_hf.py:343`).
- `off_mass = 1 - P(yes variants) - P(no variants) = 1 - exp(z_yes - log_norm) - exp(z_no - log_norm)`,
  clipped to `[0, 1]` (`vlm_hf.py:341,343`), where `log_norm` is the logsumexp of the full-vocabulary next-
  token distribution (`_collect`, `vlm_hf.py:162-180`). A response warning is added when the worst statement's
  `off_mass` exceeds `off_mass_warn` (default `0.1`, `configs/default.yaml`; warning built in
  `glance/scorer.py:258-261`).
- **Float32 head.** The selected (yes/no, or letter-label) logits are computed from the final hidden state
  through a float32 copy of the model's output-embedding head, in both the reference and cached forward paths
  (`vlm_hf.py:162-180`, `_collect`). The code comment states the reason directly: "In float16 a logit near 20
  is quantized to steps of 1/64, which alone can move `z = z_yes - z_no` by several hundredths between two
  batch layouts." This was added as an M4 deviation and is described as "kept: it is strictly better"
  (`STATUS.md`, M4 section, item 1 of the two attempted prefix-cache fixes).
- The normalizer (`log_norm`) is computed as `logaddexp(logsumexp(all other vocab logits), logsumexp(selected
  logits))`, so `off_mass` cannot go negative purely from the float16/float32 mixed-precision computation
  (comment in `vlm_hf.py:175-179`).

### Prompt templates (verbatim, `glance/prompts.py`, `PROMPT_VERSION = "p1"`)

Any edit to these templates bumps `PROMPT_VERSION` (`glance/prompts.py:1,13`). The shared prefix
(`glance/prompts.py:16-17`):

```text
IMAGE_LABEL    = "Image `{id}`:"
CONTEXT_BLOCK  = "Context: {context_json}\n\n"
```

`noul` statement (`glance/prompts.py:19-21`):

```text
Question: {instructions}
{criteria_true_false_if_given}
Answer Yes or No.
```

where each criteria line, if given, is `"Yes means: {text}\n"` / `"No means: {text}\n"`
(`NOUL_TRUE_LINE`, `NOUL_FALSE_LINE`).

Candidate statement, used for one `choice` option or one `score` level (`glance/prompts.py:24-28`):

```text
Question: {instructions}
Candidate answer: {candidate}
Is this candidate the correct answer? Answer Yes or No.
```

`letter` comparison template (`glance/prompts.py:31-33`):

```text
Question: {instructions}
Options:
{options}
Answer with the letter of the correct option.
```

with each option line rendered as `"{label}. {candidate}"` and `LETTER_LABELS = list(string.ascii_uppercase)`
(26 letters, the hard cap for this method).

Frontier baseline enumeration template (`glance/prompts.py:41-46`, eval-only):

```text
system: You answer questions about images. Reply only with JSON that matches the schema.
        Each field must be exactly one of its allowed answers.
user:   Question: {instructions}
        Allowed answers:
        - {key}: {text}
        ...
```

`candidate_text(key, description)` (`glance/prompts.py:59-61`): the option description if given, otherwise
the option key with underscores replaced by spaces.

### Assembly in `scorer` (`glance/scorer.py`)

| Type | Statements | Output formula |
| --- | --- | --- |
| `noul` | 1 | `noul = sigmoid(z)` |
| `choice`, method `independent` | 1 per option | `probabilities = softmax(z / T)` over options; `choice = argmax` |
| `score` | 1 per level (numbers and neighbours never shown) | `probabilities = softmax(z / T)` over levels; `score = sum(k * p_k)` (expected value) |

Exact formulas (`glance/scorer.py:39-70`):

```python
def sigmoid(z):      return expit(z)                                   # scipy.special.expit
def softmax(z, T=1.0):
    scaled = z / T
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)

def confidence(p):   # 1 - H(p) / ln(K); 1 when all mass on one answer, 0 when uniform
    entropy = -(p[p>0] * log(p[p>0])).sum()
    return clip(1 - entropy / ln(len(p)), 0, 1)

def margin(p):        return sorted(p)[-1] - sorted(p)[-2]              # p_top1 - p_top2
def expected_score(p): return sum(k * p[k] for k in range(len(p)))
def noul_confidence(p): return 2 * abs(p - 0.5)                        # ranking signal for noul selective accuracy
```

`T = 1.0` (raw) until calibration fits a value. `noul` answers carry no `confidence`/`margin`; the probability
itself is the signal (`glance/scorer.py:321-324`, `NoulAnswer` has no such fields in
`glance/schema.py:99-102`).

Independent scoring is the default because it is permutation-invariant by construction and scales past 26
options (`HANDOFF.md` section 6). The comparison method is `letter`: all options appear in one prompt labeled
A, B, C, ..., logits are read over the label tokens, and results are averaged over cyclic rotations of the
option order (`glance/scorer.py:233-256`).

## 3. `independent` vs `letter`: rotations and the 26-option subset protocol (D14)

`letter` renders every option in one prompt and reads the logit at each label token (`A`, `B`, ... plus a
leading-space variant, `vlm_hf.py:99,357-359`). To reduce order sensitivity, it is evaluated over up to
`letter_rotations` (default 4, `configs/default.yaml` `vlm.letter_rotations`) cyclic rotations of the option
list, spread evenly around the cycle:

```python
# glance/scorer.py:127-130
def letter_shifts(n_options, max_rotations):
    rotations = max(1, min(max_rotations, n_options))
    return dedupe(round(r * n_options / rotations) % n_options for r in range(rotations))
```

For each rotation, label `j`'s logit is assigned back to option `(shift + j) % n_options`
(`glance/scorer.py:246-248`), and the final per-option logit is the mean across rotations.

`letter` is VLM-only and capped at 26 options (`glance/prompts.py:33`,
`glance/scorer.py:192-197`). D14 (`STATUS.md`, Decisions): because `pets37` (37 options) and `caltech101`
(101 options) exceed 26, `letter` on those suites sees the true label plus 25 seeded random distractors, kept
in their original relative order (`letter_subset`, `glance/evals/run.py:80-89`, seeded by
`f"{seed}:{item.suite}:{item.item_id}"`). Because `independent` still scores every option, the report also
restricts its logits to `letter`'s same option subset ("independent, same options") so the two methods are
compared like for like at no extra model cost (`STATUS.md` D14; implemented in
`glance/evals/report.py:129-161`, `_letter_vs_independent`).

## 4. The dual-encoder backend (`glance/backends/siglip.py`)

`google/siglip2-base-patch16-256`, run in float32 on every device ("375M params: full precision everywhere
keeps logits stable", `siglip.py:45`). One image at a time: the backend raises
`UnsupportedQuestionError` if more than one image is given (`siglip.py:90-94`). `context` is not used, because
a dual encoder embeds the image and each candidate text separately (`siglip.py:95`, D6 in `STATUS.md`).

- Candidate text: `candidate_text(key, description)` (option description, or the key with underscores as
  spaces), optionally prefixed with `"a photo of {text}"` when `models.siglip.photo_prefix` is true (off by
  default, `configs/default.yaml`), then lowercased (`siglip.py:60-62`, "SigLIP2 was trained on lowercased
  text").
- `z_k` is the model's own image-text logit: `text_embedding @ image_embedding * logit_scale.exp() +
  logit_bias` (`siglip.py:107-109`), i.e. the model's learned scale and bias, not a re-derived cosine
  similarity.
- Text is tokenized with `padding="max_length", max_length=64` to match SigLIP2's training setup
  (`siglip.py:17,71-73`).
- `noul` requires `criteria.true`; when `criteria.false` is also present, `z = z_true - z_false`
  (`glance/scorer.py:172-186,227`); without `criteria.true` the request is rejected as
  `unsupported_question_for_backend` (D6, `STATUS.md`).

## 5. Canonical statement ordering and exact permutation invariance (D12)

Batch composition and padding perturb half-precision logits slightly on this hardware (see section 10 below).
To make `independent` exactly, not approximately, permutation-invariant, the VLM backend scores every batch of
statements in a canonical (lexicographically sorted-by-text) order and un-sorts the result before returning it
to the scorer:

```python
# glance/backends/vlm_hf.py:302-319 (_read)
order = sorted(range(len(texts)), key=lambda i: texts[i])
ids = tokenize([texts[i] for i in order])
selected, log_norm = forward_pass(ids)         # reference or cached path
inverse = argsort(order)
selected, log_norm = selected[inverse], log_norm[inverse]
```

Because a statement's logit no longer depends on where the caller listed it (option order or question order),
`independent` is permutation-invariant exactly rather than approximately (`STATUS.md` D12). This was verified
at max `|Δp| = 0.0e+00` on 75 reordered `pets37` requests (M3 check, `STATUS.md`) and on 180 reordered VLM
requests in the full M5 run (`STATUS.md`, Final summary, Permutation invariance row).

## 6. Image token budget

The image token budget is per request: with `k` images, each gets `budget // k` tokens (D11, `STATUS.md`).
Pixels-per-token is derived from the processor's own patch and merge sizes, never hard-coded:

```python
# glance/backends/vlm_hf.py:90-92
pixels_per_token = (image_processor.patch_size * image_processor.merge_size) ** 2
```

The processor's `min_pixels`/`max_pixels`-equivalent `size` dict is set per image so image tokens stay at or
under the per-image budget: `shortest_edge = min(MIN_IMAGE_TOKENS, per_image) * pixels_per_token`,
`longest_edge = per_image * pixels_per_token`, with `MIN_IMAGE_TOKENS = 64` as the processor's own floor
(`vlm_hf.py:31,131-145`). If the realized token count exceeds the per-image budget the backend raises
`BackendError` (`vlm_hf.py:141-142`). The actual `image_tokens` is logged on every call
(`glance/pipeline.py:139`, `glance/logging_utils`).

Default budgets by tier (`configs/default.yaml`, `HANDOFF.md` section 3): 768 for `cuda_24gb`, `cuda_12gb`,
`apple_32gb`; 384 for `apple_8gb`.

## 7. Prefix cache design (M4, `glance/backends/vlm_hf.py:229-300`)

Two forward paths:

- **Reference path** (M2, default): every statement is a full prompt, batched with left padding
  (`_reference`, `vlm_hf.py:182-207`). This is the correctness oracle and stays available as
  `--no-prefix-cache` / `vlm.prefix_cache: false` (the shipped default).
- **Cached path** (M4, opt-in via `--prefix-cache`): the shared prefix (template head, image tokens, context)
  is run once with `use_cache=True`; its KV cache is then batch-expanded (as views, never overwritten) across
  every statement in the group, and only the statement suffixes run their own forward pass (`_cached`,
  `vlm_hf.py:241-300`, `_expanded_cache`, `vlm_hf.py:229-239`).

**mRoPE position continuation.** Qwen-VL uses multimodal RoPE (rope deltas tracked per the vision content).
The prefix's `rope_deltas` value is captured when the prefix is run (`_run_prefix`, `vlm_hf.py:213-227`,
`entry.rope_delta = int(model.model.rope_deltas.reshape(-1)[0])`). For the suffix forward pass, four rows of
position ids are built: row 0 is the plain text position (`length + arange(width)`), and rows 1-3 (the t, h, w
multimodal RoPE axes) continue from `text_pos + entry.rope_delta` (`vlm_hf.py:284-286`). The code comment
states plainly: "Qwen-VL uses multimodal RoPE. Suffix position ids must continue from the prefix's rope
deltas, or logits drift without an error" (module docstring, `vlm_hf.py:1-10`).

**Cache reuse policy (D13, `STATUS.md`).** The last `PREFIX_CACHE_ENTRIES = 2` image prefixes are kept, keyed
by a SHA-256 of `"{image.sha256}:{tokens_per_image}"` per image, as an `OrderedDict` LRU (`vlm_hf.py:32,100,
209-211,254-269`). This is what the call log's "cache hit or miss" field reports; the reference path always
reports no cache (`vlm_hf.py:317`, `hit = None`).

**Acceptance criterion** (`HANDOFF.md` section 6): on 100 items the cached path must match the reference
argmax on all items with max `|Δz| ≤ 0.05`. Measured result and the two attempted fixes are documented in
`docs/paper/RESULTS_V0.md` (source: `results/v0/prefix_cache_acceptance.json`, `STATUS.md` M4 section).

## 8. Post-hoc calibration (`glance/calibration.py`)

Calibration is post-hoc, fit on a calibration split, reported only on a disjoint test split
(`HANDOFF.md` section 7).

| Question type | Method | Parameters | Fit objective |
| --- | --- | --- | --- |
| `noul` | Platt scaling: `p = sigmoid(a·z + b)` | `a`, `b` | Minimize mean binary cross-entropy via `scipy.optimize.minimize(method="L-BFGS-B")`, closed-form gradient, starting at `(a=1, b=0)` (`fit_platt`, `calibration.py:116-130`) |
| `choice` | Temperature: `p = softmax(z / T)` | `T` | Minimize multiclass NLL via `scipy.optimize.minimize_scalar` over `log T`, bounds `(-5, 7)`, `method="bounded"` (`fit_temperature`, `calibration.py:133-138`) |
| `score` | Temperature: `p = softmax(z / T)` | `T` | Same as `choice` |

Isotonic regression is an optional flag for `noul` (`sklearn.isotonic.IsotonicRegression`,
`calibration.py:148-154`), enabled only when the calibration split has at least
`calibration.isotonic_min_n` (default 1000) `noul` examples (`glance/evals/report.py:53`,
`configs/default.yaml`).

**Configuration key** (`CalibrationKey`, `calibration.py:24-43`): `(backend, model id @ revision,
prompt_version, choice_method, image_token_budget)`. Its hash is the first 12 hex characters of a SHA-256 of
the canonical JSON of these fields; the human-readable version string is `f"cal_{hash[:6]}"`. Params are
saved to `calibration/<key-hash>.json` (`params_path`, `calibration.py:210-211`) with: the key, `version`,
`fit_date`, `source_run`, `types` (pooled fit per question type), and `per_suite` (a per-suite fit reported
for comparison only, never applied automatically — `calibration.py:61-69`).

**Pooled vs per-suite fits.** `fit_rows` (`calibration.py:172-195`) pools every suite of the same question type
for the type-level fit that the API actually applies, and separately fits one set of parameters per suite for
comparison. `glance/evals/report.py:129-161` and the "Calibration" section of every run report show both, so
the report can state how much a domain-specific fit would gain over the pooled fit that ships.

**Loading is strict.** `load_params` (`calibration.py:221-235`) raises `CalibrationMismatchError` (HTTP 409,
`calibration_mismatch`) if no file exists for the active key, or if the file's own key does not exactly equal
the active key — never a silent fallback or a warning (`HANDOFF.md` section 7; `STATUS.md`: "Loading params
whose key differs from the active configuration is an error, never a warning").

`glance calibrate --run <run_id>` fits from a finished eval run's calibration split without re-running the
model (`glance/cli.py:92-109`, `glance/evals/report.py:36-55`, `fit_run_calibration`). Every eval run also
fits its own calibration into `runs/<run_id>/calibration/` purely for that run's own report (D20,
`STATUS.md`); only the explicit `glance calibrate --run` command writes the committed top-level
`calibration/` directory used by the server and CLI at inference time.

## 9. Evaluation protocol

### Suites and how each dataset was recast

Six suites plus one hand-labeled slot (`HANDOFF.md` section 8, `DATASETS.md`, `glance/evals/suites/*.py`):

| Suite | Type | Recast | Backends | Loader |
| --- | --- | --- | --- | --- |
| `pope` | noul | "Is there {object} in `img0`?"; `criteria` given so the dual encoder can answer too (D17) | all three | `glance/evals/suites/pope.py` |
| `gqa_yesno` | noul | GQA yes/no question text verbatim as `instructions`; no `criteria` (no caption form exists), so the dual encoder cannot answer it (D17) | `vlm`, `frontier` only | `glance/evals/suites/gqa_yesno.py` |
| `pets37` | choice, 37 options | "Which breed is the animal in `img0`?" | all three | `glance/evals/suites/pets37.py` |
| `caltech101` | choice, 101 options | "What is the main subject of `img0`?"; background-clutter class dropped | all three | `glance/evals/suites/caltech101.py` |
| `blur_ladder` | score, 4 levels | Synthetic Gaussian blur (frozen radii 0, 1.6, 4.0, 10.0 px at 384 px longest side) on Caltech-101 images `caltech101` never uses; levels described as situations, never as numbers | all three | `glance/evals/suites/blur_ladder.py` |
| `doctype16` | choice, 16 options | **Skipped**: RVL-CDIP license is `other` on its Hub card, unclear (`DATASETS.md`) | n/a | `glance/evals/suites/doctype16.py` |
| `human_gold` | any | Local hand-labeled JSONL, private, never leaves the machine | all (whatever types are present) | `glance/evals/suites/human_gold.py` |

`pets37` and `caltech101` share a seeded file order (`caltech101.ordered_files`); `blur_ladder` draws from the
images `caltech101` reserves but never uses (`glance/evals/suites/blur_ladder.py:44-46`,
`glance/evals/suites/caltech101.py:94-96`).

### Seeded order, split, manifests

`glance/evals/suites/base.py` (`materialize`, lines 70-113): every suite lists candidate items in a
deterministic order, then `random.Random(cfg.eval.seed).shuffle(order)` (seed 7,
`configs/default.yaml` `eval.seed`). The first `max(n, manifest_n)` (manifest_n = 1000) items are exported and
checked against the committed manifest `glance/evals/manifests/<suite>.jsonl` (item id, image sha256, split);
a mismatch against the committed manifest raises `RuntimeError` rather than silently re-deriving a different
split.

**Alternating split (D15).** `split_for(index) = "calibration" if index % 2 == 0 else "test"`
(`base.py:66-67`) alternates down the seeded order, instead of cutting the shuffled list in half, so that the
first `n` items after a `--max-hours` trim are still exactly 50/50 calibration/test.

### Per-suite time-cap trimming (D22)

`--max-hours` (default 4, `configs/default.yaml` `eval.max_hours`) trims `n` **per suite**, not uniformly:
every suite gets the same time budget, so cheap suites (few forward passes per item) keep their full `n` and
only expensive ones (many options → many forward passes per item) lose items. The trim always keeps the first
items of the seeded order, so it stays 50/50 (D22, `STATUS.md`). Implementation:
`glance/evals/run.py:302-353` (`perm_items`, `suite_seconds`, `n_under_cap`) runs a timed warmup
(`cfg.eval.warmup_items = 10` items per unit, model-load time excluded), estimates each suite's total
remaining cost including its permutation pass, and binary-searches (40 iterations) for the largest per-suite
`n` whose combined ETA fits the time budget.

### Failure accounting

A failed item (any `GlanceError`) is logged to `errors.jsonl` and counted; the run continues
(`glance/evals/run.py:276-289`). A unit is marked invalid in the report if its failure rate exceeds
`eval.max_failure_rate` (default 0.02) (`glance/evals/report.py:98-101`, `HANDOFF.md` section 9).

## 10. Metrics (`glance/evals/metrics.py`, `glance/calibration.py`)

- **Accuracy**: fraction where `argmax(p) == label` (`choice`)/`(p >= 0.5) == label` (`noul`, treated as a
  binary argmax) (`_views`, `metrics.py:65-81`).
- **Macro-F1** (`choice`): `sklearn.metrics.f1_score(y, pred, average="macro", zero_division=0)`
  (`metrics.py:59-62`).
- **AUROC** (`noul`): `sklearn.metrics.roc_auc_score`; `None` if the test slice has only one label value
  present (`metrics.py:50-56`).
- **MAE in levels** (`score`): `mean(|expected_score(p) - label_index|)` (`metrics.py:114-115`).
- **NLL**: `noul` is mean binary cross-entropy, `-(y·log p + (1-y)·log(1-p))` with `p` clipped to
  `[1e-12, 1-1e-12]`; `choice`/`score` is `-mean(log p[label_index])` (`metrics.py:102-109`, `EPS = 1e-12` from
  `calibration.py:23`).
- **Brier**: `noul`: `mean((p - y)^2)`; `choice`/`score`: `mean(sum((p - onehot(y))^2))` (`metrics.py:107,110`).
- **Top-label ECE, 15 equal-mass bins** (`ece_equal_mass`, `calibration.py:75-86`): items are sorted by
  top-label confidence (`max(p)` for `choice`/`score`, `max(p, 1-p)` for `noul`) and split into
  `min(n_bins, n)` roughly-equal-count bins via `numpy.array_split` on the sorted order; each bin contributes
  `(bin size / n) · |mean(confidence in bin) - mean(correctness in bin)|`, summed over bins. `ece_bins = 15`
  (`configs/default.yaml`, `HANDOFF.md` section 8).
- **Selective accuracy** at coverage `{50, 80, 90, 100}%`: rank items by a ranking-confidence score — for
  `choice`/`score` this is the entropy confidence `1 - H(p)/ln K`; for `noul` it is `2·|p - 0.5|`
  (`noul_confidence`, `scorer.py:68-70`) — then report accuracy on the most-confident `k = round(coverage ·
  n)` items (`selective_accuracy`, `metrics.py:20-28`).
- **Permutation sensitivity**: for every `independent`/`letter` unit, up to `cfg.eval.permutation_items` test
  items (default 100; M3 checks used 75; the M5 full run used 30 via `--permutation-items`, see
  `STATUS.md`) are each re-requested under `cfg.eval.permutation_orders` (default 3) random option
  orderings; the metric is the max and mean absolute shift in raw per-option probability versus the
  unpermuted baseline, plus a count of `choice` flips (`permutation_sensitivity`, `glance/evals/run.py:382-
  417`).
- **ECE sampling floor (D23)**, `ece_noise_floor` (`glance/evals/metrics.py:84-92`): equal-mass ECE with 15
  bins is biased upward at small sample sizes, because each bin's accuracy is a noisy average of only a few
  outcomes. The floor is computed by taking the test set's own observed top-label confidences, then for
  `draws = 200` repetitions drawing synthetic correctness `~ Bernoulli(confidence)` per item (a hypothetical
  *perfectly calibrated* predictor with the same confidence values) and recomputing `ece_equal_mass` on that
  synthetic outcome; the floor is the mean of those 200 simulated ECEs. `seed = 7`
  (`numpy.random.default_rng(seed)`, matching the eval seed). D23's rationale from `STATUS.md`: "at a few
  hundred test items that bias is the same size as the 0.05 gate. The gate itself is unchanged" — the floor is
  reported next to every measured ECE, it does not relax the go/no-go threshold.

## 11. Frontier baseline protocol

`glance/backends/frontier.py` calls a vision model through LiteLLM with a JSON schema that enumerates each
question's allowed answers (`answer_schema`, `frontier.py:72-79`: one string field `q{i}` per question,
`enum` = that question's allowed keys, `additionalProperties: false`), at `temperature=0`
(`_complete`, `frontier.py:90-110`). D26 (`STATUS.md`): several current frontier models reject the
`temperature` parameter outright; the adapter retries once without it on a matching `BadRequestError` and
appends a response warning reporting the switch, which is also captured in the call log. `MAX_OUTPUT_TOKENS =
4096` to leave room for models that reason before answering (`frontier.py:28`). The model id is read from
`FRONTIER_MODEL` (or passed explicitly to `glance baseline`) and logged exactly (`frontier.py:49`).

**Picks are never stored.** `run_item` (`glance/evals/run.py:121-125`) keeps only `row["correct"] = (picked
index == label_index)` for a frontier row; the actual pick string is discarded before it reaches
`predictions.jsonl`. The call log redacts the answer with the literal string
`"<redacted: frontier outputs are evaluation-only>"` (`FRONTIER_REDACTED`, `glance/pipeline.py:30,159`). This
is a guardrail from `HANDOFF.md` section 11 ("Frontier model outputs are evaluation-only. Never write them to
any file that could serve as a training label; several providers bar training competing models on their
outputs").

**Gating.** `Engine.allow_frontier` must be explicitly set true; the eval runner only does this after
`--confirm-spend` and a live `FRONTIER_MODEL` + key are confirmed (`_frontier_ready`,
`glance/evals/run.py:181-195`); the server never sets it, so `model: frontier` is refused with
`unsupported_question_for_backend` from any HTTP request (`glance/pipeline.py:116-120`). The baseline runs on
the **test split only**, capped by `--baseline-n` (default 300 per suite, `configs/default.yaml`
`eval.baseline_n`), and is refused on `human_gold` without `--allow-upload-gold`
(`glance/evals/run.py:162-164`). A rough cost estimate (`estimate_frontier_cost`,
`glance/evals/run.py:198-214`, deliberately on the high side) is printed before any paid call.

**`glance baseline` (D25, added 2026-09-19 at the user's request, not in `HANDOFF.md`).** Adds the frontier
baseline to an already-finished eval run without editing `.env`
(`glance/evals/baseline.py`). It asks for provider, model, and API key interactively in the terminal
(`ask_model_and_key`, `baseline.py:55-78`, hidden input via `getpass.getpass`). The key lives only in the
`FrontierBackend._api_key` attribute for the lifetime of that one process; it is never written to disk, a log,
or shell history (`frontier.py:52-54`). `scrub()` (`frontier.py:32-36`) removes the exact key string and
anything shaped like a key (regex `(sk|key|AIza)[-_A-Za-z0-9*.]{8,}`) from any provider error text before it
is logged or printed. One cheap smoke-test call (`SMOKE_QUESTION`/`SMOKE_IMAGE`, `baseline.py:37-38`) validates
the key and that structured output works before the cost estimate is shown; nothing else is sent until an
interactive `y` (or `--confirm-spend`).

## 12. Hardware and software versions

All measured v0 numbers in this repository were produced on one machine (`results/v0/*/env.json`, all
snapshots agree):

| Field | Value | Source |
| --- | --- | --- |
| Chip | Apple M5 | `results/v0/m3_pope_n200/env.json: doctor.chip` |
| RAM | 32.0 GB | `env.json: doctor.ram_gb` |
| `vram_gb` (Metal working-set ceiling, `torch.mps.recommended_max_memory()`) | 25.0 GB | `env.json: doctor.vram_gb`; formula in `glance/doctor.py:75` |
| Device / dtype | `mps` / `float16` (VLM); SigLIP2 runs `float32` on every device | `env.json: doctor.device, doctor.dtype`; `glance/backends/siglip.py:45` |
| OS | Darwin 26.6.2 | `env.json: doctor.os` |
| Selected tier | `apple_32gb` | `env.json: doctor.selected_tier`; also `STATUS.md` M0 |
| torch | 2.14.0 | `env.json: uv_pip_freeze`; also `pyproject.toml`/`uv.lock`, `STATUS.md` M0 ("torch 2.14.0, transformers 5.17.0") |
| transformers | 5.17.0 | `env.json: uv_pip_freeze` |
| torchvision | 0.29.0 | `env.json: uv_pip_freeze` |
| accelerate | 1.15.0 | `env.json: uv_pip_freeze` |
| litellm | 1.101.0 | `env.json: uv_pip_freeze` |
| numpy / scipy / scikit-learn | 2.4.6 / 1.17.1 / 1.9.1 | `env.json: uv_pip_freeze` |
| pydantic / flask / pillow / datasets | 2.13.5 / 3.1.3 / 12.3.0 / 5.0.1 | `env.json: uv_pip_freeze` |
| Python | 3.11 (`requires-python = ">=3.11,<3.12"`) | `pyproject.toml` |

`harness_version` (semver in `pyproject.toml`, bumped on any change to scoring, prompts, or calibration math
per `HANDOFF.md` section 9): `0.1.0` through the M3 check (`results/v0/m3_pope_n200/report.md`: "Harness
0.1.0 at git `3f116790f2`"); `0.2.0` from M3's float32-head deviation through the M4 check
(`results/v0/m4_calibrated_n80/report.md`: "Harness 0.2.0 at git `58bb3b59b4`") and through the M5 full
evaluation run itself (`results/v0/m5_full_eval/report.md`: "Harness 0.2.0 at git `55177cc51c`" — the run
executed against the M4 commit, before the M5 milestone's own code, including the `off_mass` fix, was
committed); `0.2.1` starting with the M5 milestone commit and covering the stretch experiments
(`results/v0/stretch/injection/report.md`: "Harness 0.2.1 at git `d0bb23bd0c`"). The `off_mass` artifact in
the M5 run's call log (see `docs/paper/RESULTS_V0.md` "Negative results") is exactly this boundary: the run
that exhibits the artifact is on 0.2.0, and the fix ships in 0.2.1. Current `pyproject.toml` version: `0.2.1`.

Model ids and pinned revisions (`MODELS.md`, `configs/default.yaml`): dual encoder
`google/siglip2-base-patch16-256@3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` (Apache-2.0) on every device; VLM
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` (Apache-2.0) for the `apple_32gb` and
`cuda_12gb` tiers (this machine's tier is `apple_32gb`); `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd
4f463576704203` for `apple_8gb` and `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` for
`cuda_24gb` were pinned but never downloaded (not this machine's tier, `MODELS.md`).

## 13. Licensing constraints that shaped dataset choice

`HANDOFF.md` section 11 requires Apache-2.0 or MIT model weights only, and a dataset-card license check
before any download, recorded in `DATASETS.md`; an unclear license means the suite is skipped rather than
used. This directly shaped v0's dataset choices (`DATASETS.md`):

- `doctype16` (RVL-CDIP) was **skipped**: both Hub mirrors checked (`aharley/rvl_cdip`, `chainyo/rvl-cdip`)
  list license `other`, and the upstream IIT-CDIP collection has no clear reuse terms.
  `glance/evals/suites/doctype16.py:20-21` raises `SuiteSkipped` unconditionally with this reason.
- `pope` questions were rebuilt from the official MIT-licensed GitHub repository (pinned commit
  `08d957b917e5a378a2f99d35b6293c536a66298b`) rather than the Hub mirror `lmms-lab/POPE`, which carries no
  license tag (`DATASETS.md` notes).
- `caltech101` was sourced from the official CaltechDATA record (CC BY 4.0, doi:10.22002/D1.20086) rather than
  the Hub mirror `flwrlabs/caltech101`, whose license is listed as `unknown` (`DATASETS.md` notes).
- ImageNet and other research-only sets are excluded per `HANDOFF.md` and never used anywhere in this repo.
- `human_gold` stays private by construction: `gold/` is gitignored, and the frontier baseline refuses to run
  on it without the explicit `--allow-upload-gold` flag (`DATASETS.md`, `glance/evals/run.py:162-164`).
- Because the public suites are very likely present in every frontier and open model's training data,
  `human_gold` is called out as "the only uncontaminated check" (`DATASETS.md`, `STATUS.md` Final summary,
  "Open inputs from you").

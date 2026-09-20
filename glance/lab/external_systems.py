"""Adapters for two outside image-rating systems run on this project's own lab rating scales.

Registered design: `lab/NOTES.md` entry 37 and `docs/paper/COMPARABLE_SYSTEMS.md` ("Four systems", items 1 and 2).
This module implements items 1 and 2 only (`AlexWortega/openjev` v2 and `zhangzicheng/q-sit-mini`); items 3 and 4
of that doc are not registered for entry 37 and are out of scope here.

Both adapters share one small interface (`ExternalSystem`) so `tools/external_collect.py` can drive either one and
write rows in `glance.lab.collect`'s own schema, and `tools/external_report.py` can calibrate them with
`glance.rating.fit_matrix`/`apply_matrix` -- the same matrix calibration every readout in this lab uses, reused
here rather than reimplemented.

--- OpenJevV2 (AlexWortega/openjev, v2 4B checkpoint, subfolder `qwen3.5-4b-nli-v2/`) ---------------------------

An NLI cross-encoder: `Qwen3_5ForSequenceClassification`, 3 labels in dleemiller order (0=contradiction,
1=entailment, 2=neutral), last-token pooling. Quoted from the model's own README ("Use it" section, and "openjev-4B
v2: text, images and agents") and from `modeling_openjev.py` / `code/train.py` / `code/eval_image_nli.py` /
`code/data_mix.py` (all fetched from huggingface.co/AlexWortega/openjev on 2026-09-20, commit
`4395b29714015162db6112de91c35688e6e42717`):

    "All checkpoints: `Qwen3_5ForSequenceClassification`, 3 labels `contradiction`, `entailment`, `neutral`,
    last-token pooling, trained with plain cross-entropy over the three classes."                  (README.md)

    "Images go inside the premise as `<|vision_start|><|image_pad|>...<|vision_end|>` with `pixel_values` /
    `image_grid_thw` from the Qwen3.5 image processor; see `code/doom_vision.py` and `code/eval_image_nli.py`."
                                                                                                     (README.md)

    CON, ENT, NEU = 0, 1, 2                                                          (modeling_openjev.py)
    DEFAULT_TEMPLATE = "Premise: {premise}\\nHypothesis: {hypothesis}"                (modeling_openjev.py)
    def rerank(self, question, options, hyp_fmt: str = "The correct answer is: {}") -> int:
        # docstring: 'Zero-shot multiple choice: option with the highest P(entailment) given the question as premise.'
                                                                                       (modeling_openjev.py)

    TEMPLATE = "Premise: {premise}\\nHypothesis: {hypothesis}"                        (code/train.py)
    IMG_MARK = "<<IMG>>"  # data_mix.py puts this where the image-token block must go (code/train.py)

    text = template.format(premise=r["premise"].replace(IMG_MARK, block).strip(), hypothesis=r["hypothesis"].strip())
    n_img = int(ip(images=[Image.new("RGB", (320, 240))], return_tensors="pt")["image_grid_thw"].prod()) \\
        // ip.merge_size ** 2
    block = "<|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>"           (code/eval_image_nli.py)

    LEAD_INS = ["A photograph:", "A photo:", "A picture:", "An image:", "A photograph of a scene:"]
    ...
    p.add(f"{lead} {IMG}", stmt, OURS["entailment"], "vqa_disagree"/..., rel)         (code/data_mix.py)

Per-level claim (see `OpenJevV2.claim_premise`/`claim_hypotheses`): the premise reuses `data_mix.py`'s own recipe
for putting an image into a premise (one `LEAD_INS` phrase + the image marker, here followed by the ladder's own
question with `` `img0` `` replaced by "this image"); the hypothesis reuses `rerank()`'s own recipe for turning a
question and a candidate answer into a hypothesis ("The correct answer is: {level text}"), one hypothesis per
rubric level. Both recipes are copied unchanged from the model's own code -- nothing here invents a new claim style.

P(true) per level: the model's own entailment probability (label index ENT=1) from the 3-way softmax, reported as
a log-odds "logit of true" -- entailment vs. everything else -- the 3-way generalization of how
`glance.lab.generic_vlm.GenericVlm.score_statements` reports `z = z_yes - z_no` (log-odds of yes vs. no) for every
VLM backend in this lab:

    logit_true = logit_entailment - logsumexp([logit_contradiction, logit_neutral])

Label ids: `OpenJevCrossEncoder` (the wrapper the README shows first) has NO image handling at all -- its
`_encode` only tokenizes `(premise, hypothesis)` text pairs. Image claims are only supported through the lower
lower-level path `code/eval_image_nli.py` actually uses (`AutoModelForSequenceClassification` + a separately
loaded image processor + manual `pixel_values`/`image_grid_thw`), which is what this adapter mirrors instead of
the wrapper class.

HARD RULE for the lab machine this adapter was written on (`lab/NOTES.md` entry 37; two GPU jobs already running,
memory limited): never load this 4B checkpoint's weights, on any device. `OpenJevV2.load()` is left fully
implemented, following the README's own recipe, so a machine with headroom can run
`tools/external_collect.py --system openjev` later -- but it was never called while writing or testing this file;
only `build_inputs()` (tokenizer + image processor, no model weights) was exercised.

--- QSitMini (zhangzicheng/q-sit-mini, LLaVA-OneVision architecture) -------------------------------------------

A small LMM trained to answer image-quality questions with one of five fixed words. Quoted from the model's own
README ("Image Quality Scoring" section, fetched from huggingface.co/zhangzicheng/q-sit-mini on 2026-09-20, commit
`198f645ecbebd113041aa66fb973a1055d6a8c0d`):

    "Ensure that you use the Transformers package version 4.45.0 (`pip install transformers==4.45.0`)."

    toks = ["Excellent", "Good", "Fair", "Poor", "Bad"]
    ids_ = [id_[0] for id_ in tokenizer(toks)["input_ids"]]
    print("Rating token IDs:", ids_)
    ...
    conversation = [{"role": "user", "content": [{"type": "text", "text":
        "Assume you are an image quality evaluator. \\nYour rating should be chosen from the following five "
        "categories: Excellent, Good, Fair, Poor, and Bad (from high to low). \\nHow would you rate the quality "
        "of this image?"}, {"type": "image"}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    ...
    prefix_text = "The quality of this image is "
    prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(0)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    output = model.generate(**inputs, max_new_tokens=1, output_logits=True, return_dict_in_generate=True)
    last_logits = output.logits[-1][0]

`ids_[0]` is the FIRST sub-token id of each rating word -- the card's own defensive rule for a tokenizer that might
split one of the five words into more than one piece. Checked empirically in this lab (CPU smoke test, 2026-09-20):
for this tokenizer all five of "Excellent", "Good", "Fair", "Poor", "Bad" happen to be single tokens
(`QSitMini.level_token_counts == [1, 1, 1, 1, 1]`), so the `[0]` never actually discards anything here -- but this
adapter still applies the card's own rule rather than assuming single tokens, since a different tokenizer version
could split differently, and `level_token_counts` records the real count per word so that assumption is checkable.

Generating one token with `output_logits=True` is one forward pass, identical to reading `model(**inputs).logits`
directly (`generate` with `max_new_tokens=1` does exactly one forward call and returns that call's logits); this
adapter reads the logits straight from that single forward pass instead of calling `.generate()`, matching this
lab's "no generation" design (`glance/rating.py` module docstring: "single forward passes ... no generation")
without changing what is computed.

Direction: q-sit-mini's own scale runs Excellent (best) -> Bad (worst); the card's own `wa5()` weighting gives
Excellent the highest number (1.0) and Bad the lowest (0.0), i.e. HIGH = GOOD. This project's ladders
(`glance/lab/ladders.py`, `LADDERS[ladder]["levels"]`) run level 0 = best/least-distorted -> level K-1 =
worst/most-distorted for every one of the five lab scales (checked directly, e.g. blur: level 0 "Sharp: edges are
crisp ..." -> level 3 "Heavily blurred: vague blobs of color ..."), i.e. HIGH LEVEL = BAD. The two scales run in
the same direction along the distortion axis (further along the list = worse in both), which means q-sit-mini's
own high=good weighting runs OPPOSITE to this lab's high=bad `level` field, so `QSitMini.expected_degradation`
flips the sign of the card's own weighted average before comparing it with `level`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ExternalSystem:
    """Shared interface. `tools/external_collect.py` only calls `load` once and then `features` per item."""

    name: str
    model_id: str
    revision: str
    default_dtype: str = "float32"

    def load(self, device: str, dtype: str) -> None:
        raise NotImplementedError

    def features(self, image_path: str | Path, ladder_meta: dict[str, Any]) -> list[float]:
        """The per-level numbers the registered design asks for, one forward pass (or one batched request) per
        item. `ladder_meta` is one entry of `glance.lab.ladders.LADDERS` (or another bench's `SCALES`): a dict
        with `instructions` (str) and `levels` (list[str])."""
        raise NotImplementedError


# ============================================================================================================
# 1. AlexWortega/openjev (v2, 4B, multimodal) -- NEVER LOADED on this lab's machine (entry 37 hard rule).
# ============================================================================================================

CON, ENT, NEU = 0, 1, 2  # modeling_openjev.py: dleemiller order, 0=contradiction, 1=entailment, 2=neutral
OPENJEV_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"  # modeling_openjev.py DEFAULT_TEMPLATE == code/train.py TEMPLATE
OPENJEV_IMG_MARK = "<<IMG>>"  # code/train.py: IMG_MARK = "<<IMG>>"
OPENJEV_LEAD_IN = "A photograph:"  # code/data_mix.py: first of LEAD_INS = ["A photograph:", "A photo:", ...]
OPENJEV_HYP_FORMAT = "The correct answer is: {}"  # modeling_openjev.py: rerank()'s own hyp_fmt default


class OpenJevV2(ExternalSystem):
    """See the module docstring for the sourcing of every design choice below."""

    name = "openjev"
    model_id = "AlexWortega/openjev"
    revision = "4395b29714015162db6112de91c35688e6e42717"  # `sha` field of https://huggingface.co/api/models/AlexWortega/openjev, resolved 2026-09-20
    subfolder = "qwen3.5-4b-nli-v2"
    image_processor_id = "Qwen/Qwen3.5-4B"  # code/eval_image_nli.py: `--image-processor` default (Apache-2.0, only its small preprocessor_config.json is fetched)
    default_dtype = "bfloat16"  # README / code/eval_image_nli.py both load the checkpoint in bfloat16

    def __init__(self) -> None:
        self.tokenizer = None
        self.image_processor = None
        self.model = None
        self.device: str | None = None
        self.dtype: str | None = None

    # --- safe to call on this machine: no model weights, only a tokenizer and an image *processor* -----------

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, subfolder=self.subfolder)
        self.tokenizer.padding_side = "right"  # modeling_openjev.py: "the head pools the last non-pad token"
        return self.tokenizer

    def load_image_processor(self):
        from transformers import AutoImageProcessor

        self.image_processor = AutoImageProcessor.from_pretrained(self.image_processor_id)
        return self.image_processor

    def label_token_ids(self) -> dict[str, int]:
        """NOT vocabulary token ids. openjev's NLI head is a 3-way SEQUENCE CLASSIFICATION head on a pooled
        hidden state (`Qwen3_5ForSequenceClassification`), not a next-token logit readout, so there is no
        "yes"/"no" text token to look up here the way `QSitMini.level_token_ids` looks up word pieces. These are
        the classifier's own fixed label ids (dleemiller order), confirmed against `qwen3.5-4b-nli-v2/config.json`
        (`"label2id": {"contradiction": 0, "entailment": 1, "neutral": 2}`, fetched 2026-09-20)."""
        return {"contradiction": CON, "entailment": ENT, "neutral": NEU}

    def claim_premise(self, ladder_meta: dict[str, Any]) -> str:
        """`OPENJEV_LEAD_IN` + the image marker, exactly `code/data_mix.py`'s `build_images()` pattern
        (`f"{lead} {IMG}"`), followed by the ladder's own question with the `` `img0` `` markdown reference turned
        into plain prose so the premise reads as one sentence about "this image"."""
        question = ladder_meta["instructions"].replace("`img0`", "this image")
        return f"{OPENJEV_LEAD_IN} {OPENJEV_IMG_MARK} {question}"

    def claim_hypotheses(self, ladder_meta: dict[str, Any]) -> list[str]:
        """One hypothesis per level, `rerank()`'s own `hyp_fmt` ("The correct answer is: {}") applied to each
        level's description text -- unchanged from how openjev's own code turns a candidate answer into a claim."""
        return [OPENJEV_HYP_FORMAT.format(level) for level in ladder_meta["levels"]]

    def vision_block(self, n_img_tokens: int) -> str:
        """code/eval_image_nli.py: block = "<|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>" """
        return "<|vision_start|>" + "<|image_pad|>" * n_img_tokens + "<|vision_end|>"

    def build_inputs(self, image_path: str | Path, ladder_meta: dict[str, Any]) -> dict[str, Any]:
        """Everything up to (not including) the forward pass: a real tokenizer and a real image *processor*
        (feature extraction only, no model weights). This is the boundary this lab's openjev smoke check stays
        inside of (`tools/external_collect.py`'s `--system openjev` verification never goes past this method)."""
        from PIL import Image

        if self.tokenizer is None:
            self.load_tokenizer()
        if self.image_processor is None:
            self.load_image_processor()
        image = Image.open(image_path).convert("RGB")
        hypotheses = self.claim_hypotheses(ladder_meta)
        k = len(hypotheses)
        # code/eval_image_nli.py's own recipe for the per-image token count:
        single_vis = self.image_processor(images=[image], return_tensors="pt")
        n_img = int(single_vis["image_grid_thw"].prod()) // self.image_processor.merge_size ** 2
        block = self.vision_block(n_img)
        premise = self.claim_premise(ladder_meta).replace(OPENJEV_IMG_MARK, block)
        texts = [OPENJEV_TEMPLATE.format(premise=premise, hypothesis=h) for h in hypotheses]
        enc = self.tokenizer(texts, truncation=True, max_length=4096, padding=True, return_tensors="pt")
        # code/train.py DataCollatorNLIMM processes one image per row that carries one; every row here shares the
        # same image, so the processor is called on k identical copies (mirrors the collator's own per-row cost).
        batch_vis = self.image_processor(images=[image] * k, return_tensors="pt")
        return {
            "texts": texts, "hypotheses": hypotheses, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "pixel_values": batch_vis["pixel_values"], "image_grid_thw": batch_vis["image_grid_thw"], "n_image_tokens": n_img,
        }

    # --- HARD RULE (lab/NOTES.md entry 37): never call this on this lab's machine ----------------------------

    def load(self, device: str, dtype: str | None = None) -> None:
        """The full load a real collection run needs (README "Use it", plain-`transformers` form:
        `AutoModelForSequenceClassification.from_pretrained(..., subfolder="qwen3.5-4b-nli-v2")`). Left fully
        implemented so a machine with headroom can run `tools/external_collect.py --system openjev` once this
        lab's GPU jobs are done -- but this method was never invoked while writing or verifying this adapter."""
        import torch
        from transformers import AutoModelForSequenceClassification

        dtype = dtype or self.default_dtype
        self.load_tokenizer()
        self.load_image_processor()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, subfolder=self.subfolder, dtype=getattr(torch, dtype)
        )
        text_config = self.model.config.get_text_config()
        if text_config.pad_token_id is None:
            text_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device).eval()
        self.device, self.dtype = device, dtype

    def features(self, image_path: str | Path, ladder_meta: dict[str, Any]) -> list[float]:
        if self.model is None:
            raise RuntimeError("OpenJevV2.load() was never called (and must not be, on this lab's machine; entry 37 hard rule)")
        import numpy as np
        import torch
        from scipy.special import logsumexp

        built = self.build_inputs(image_path, ladder_meta)
        # code/train.py's DataCollatorNLIMM also computes `mm_token_type_ids` (input_ids == image_token_id) for
        # training, but never showed it being passed to forward() outside the collator; Qwen-style multimodal
        # models generally locate image tokens from `config.image_token_id` internally, so it is not passed here.
        # UNVERIFIED (this lab's hard rule means the model was never actually loaded to check its forward signature):
        # if a real run ever raises a missing-argument error, that is the first thing to add back.
        with torch.no_grad():
            out = self.model(
                input_ids=built["input_ids"].to(self.device),
                attention_mask=built["attention_mask"].to(self.device),
                pixel_values=built["pixel_values"].to(self.device),
                image_grid_thw=built["image_grid_thw"].to(self.device),
            )
        logits = out.logits.float().cpu().numpy()  # (K, 3): [contradiction, entailment, neutral]
        true_logit = logits[:, ENT] - logsumexp(logits[:, [CON, NEU]], axis=1)  # log-odds of entailment vs. not
        return [float(v) for v in np.asarray(true_logit)]


# ============================================================================================================
# 2. zhangzicheng/q-sit-mini (0.9B) -- CPU-only smoke test allowed on this lab's machine (entry 37 hard rule).
# ============================================================================================================

QSIT_LEVEL_WORDS = ["Excellent", "Good", "Fair", "Poor", "Bad"]  # README "Image Quality Scoring": toks = [...]
QSIT_QUESTION = (
    "Assume you are an image quality evaluator. \n"
    "Your rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad "
    "(from high to low). \n"
    "How would you rate the quality of this image?"
)  # quoted verbatim from the README's "Image Quality Scoring" example
QSIT_ANSWER_PREFIX = "The quality of this image is "  # README: manually appended assistant prefix before reading logits
QSIT_QUALITY_WEIGHTS = (4.0, 3.0, 2.0, 1.0, 0.0)  # Excellent..Bad, "4..0" (this lab's own choice; the card's own wa5() uses 1, 0.75, ..., 0, same order)


class QSitMini(ExternalSystem):
    """See the module docstring for the sourcing of every design choice below."""

    name = "qsit"
    model_id = "zhangzicheng/q-sit-mini"
    revision = "198f645ecbebd113041aa66fb973a1055d6a8c0d"  # `sha` field of https://huggingface.co/api/models/zhangzicheng/q-sit-mini, resolved 2026-09-20
    default_dtype = "float32"  # this lab's machine only ever runs this model on CPU (entry 37 hard rule); float16 (the card's own choice) is not supported on CPU

    def __init__(self) -> None:
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.device: str | None = None
        self.dtype: str | None = None
        self.level_ids: list[int] = []
        self.level_token_counts: list[int] = []

    def load(self, device: str, dtype: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration

        dtype = dtype or self.default_dtype
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
        )
        self.model.to(device).eval()
        self.device, self.dtype = device, dtype
        self.level_ids, self.level_token_counts = self.level_token_ids()

    def level_token_ids(self) -> tuple[list[int], list[int]]:
        """README: `ids_ = [id_[0] for id_ in tokenizer(toks)["input_ids"]]` -- the FIRST sub-token id of each of
        the five rating words. Also returns how many sub-tokens each word actually took, so a caller can see
        whether (and which) words needed this fallback rather than being a single token."""
        if self.tokenizer is None:
            raise RuntimeError("load a tokenizer first (QSitMini.load(), or set .tokenizer directly for tests)")
        encoded = self.tokenizer(QSIT_LEVEL_WORDS)["input_ids"]
        ids = [seq[0] for seq in encoded]
        counts = [len(seq) for seq in encoded]
        return ids, counts

    def build_prompt(self, image) -> dict[str, Any]:
        conversation = [{"role": "user", "content": [{"type": "text", "text": QSIT_QUESTION}, {"type": "image"}]}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        prefix_ids = self.tokenizer(QSIT_ANSWER_PREFIX, return_tensors="pt")["input_ids"]
        import torch

        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        return inputs

    def features(self, image_path: str | Path, ladder_meta: dict[str, Any]) -> list[float]:
        # `ladder_meta` is intentionally unused: q-sit-mini reads its OWN fixed prompt and vocabulary, not this
        # lab's per-rubric wording (entry 37: "its own prompt and its own five level words").
        del ladder_meta
        if self.model is None:
            raise RuntimeError("QSitMini.load() was never called")
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.build_prompt(image)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)  # one forward pass; see the module docstring for why this replaces `.generate(max_new_tokens=1)`
        last = out.logits[0, -1].float()
        return [float(last[i]) for i in self.level_ids]

    def expected_degradation(self, logits: list[float]) -> float:
        """softmax over the five logits, weights 4..0 for Excellent..Bad, then sign-flipped: see the module
        docstring ("Direction: ...") for why this lab's `level` field needs the flip."""
        import numpy as np
        from scipy.special import softmax

        p = softmax(np.asarray(logits, dtype=np.float64))
        quality = float(np.dot(p, np.asarray(QSIT_QUALITY_WEIGHTS)))  # high = good, the card's own direction
        return -quality  # flipped: high = bad, matching this lab's `level` direction (see module docstring)


SYSTEMS: dict[str, type[ExternalSystem]] = {"openjev": OpenJevV2, "qsit": QSitMini}
METHOD_NAMES: dict[str, str] = {"openjev": "openjev_claims", "qsit": "qsit_levels"}

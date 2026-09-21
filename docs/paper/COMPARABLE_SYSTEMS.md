# Comparable systems (written 2026-09-20)

Every row below was checked today against its primary source (Hugging Face model card, GitHub repo, or paper) with
WebFetch or WebSearch. Cells marked "not verified" mean the source could not be opened or did not state the fact;
they are not filled from memory. glance's own setup, for reference: Qwen3-VL-4B-Instruct (Apache-2.0), revision
`89644892e4d85e24eaac8bacfd4f463576704203`, float16 on Apple M5 MPS, PyTorch, no CUDA. Hard rule: glance may only
RUN model weights licensed Apache-2.0 or MIT. Benchmarks: five synthetic 4-level degradation ladders (blur, noise,
JPEG, underexposure, resolution; 448 px Oxford-IIIT Pet photos, 1,000 items/scale) and a 5-level, 25-distortion
KADID-10k-based semantic ladder set; manifests are JSONL with `path`, `level`, `split` (`lab/manifests/*.jsonl`,
`lab/manifests_kadid/*.jsonl`). glance's own shipped numbers on the 4-level ladders: v0 readout mean accuracy 0.500,
same readout with fitted bias+temperature 0.810, `ens4d` (4-pass ensemble + matrix scaling) 0.867, mean ECE about
0.03 (`docs/paper/RESULTS_LAB.md`).

## A. Vision-capable Jev-style reproductions

All four are read today; three are also described in `docs/paper/RELATED_WORK.md` (GitHub side). No other
vision-capable entry was found in `github.com/OmniJev/awesome-jev` beyond these four (checked today; entries citing
"robots" or "screenshots" route pixels through a separate OCR/vision step and feed only text to their Jev component).
None of the four publishes any IQA benchmark number (their own reported numbers are NLI/ANLI accuracy, Pets/CLEVR-4
classification, or Doom kill counts) so the "published KADID-10k or similar" column is "none found" for all four and
is omitted below; see the effort column instead.

| Name + URL | What it is | Code license | Weights license (base model) | Passes rule? |
| --- | --- | --- | --- | --- |
| hr98w/jev-visual. github.com/hr98w/jev-visual | Zero-shot typed-decision reader over one image; shares one vision prefill across up to 64 question suffixes by forking the KV cache (MLX). | MIT (repo LICENSE) | Qwen3.5-0.8B, 4-bit, ~596 MiB; base license not separately confirmed today, but Qwen3.5-4B-Base (same family) is Apache-2.0 (checked at huggingface.co/Qwen/Qwen3.5-4B-Base) | Likely yes, base family Apache-2.0; exact 0.8B card not opened today, so "likely" not "yes" |
| IamBusy/OpenJev-Vision. huggingface.co/IamBusy/OpenJev-Vision-Research-v0.1 | Two trained heads for vision: a small CNN with a learned prior, and frozen DINOv2-small features, evaluated on synthetic scenes and Pets/CLEVR-4 subsets. | Not stated separately from weights license | Mixed: Apache-2.0 for the synthetic-scene CNN heads, CC BY-SA 4.0 for the Pets heads, CC BY 4.0 for the CLEVR-4 heads; frozen backbone facebook/dinov2-small is Apache-2.0 (checked today) | Partial/unclear: the CNN-only checkpoints pass, the Pets and CLEVR-4 checkpoints do not (CC licenses are not Apache/MIT) |
| AlexWortega/openjev (v2). huggingface.co/AlexWortega/openjev | Qwen3.5 finetuned as an NLI cross-encoder; v2 (4B) is multimodal, image claims scored via `<|vision_start|>...<|vision_end|>` tokens. | MIT (model card) | Base Qwen/Qwen3.5-4B-Base, Apache-2.0 (checked today) | Yes, on license grounds. Exact vision-tower mechanism (native Qwen3.5 multimodality vs. an added encoder) not independently confirmed today |
| zhengxuyu/litjev (LitJev). github.com/zhengxuyu/litjev | Serves Jev's own request/response schema from a local HF checkpoint, reading option logits; README claims "the full Qwen family is supported," any size. | Apache-2.0 (repo LICENSE; adapted examples keep their own MIT) | No vision checkpoint ships or is pinned; README's own default is `Qwen/Qwen3.8-27B` (text), which needs one H100 80GB | No usable weights to rate: nothing vision-capable is actually distributed here |

| Name | Params / size | IQA-trained or general? | Arbitrary K-level rubric? | Fair-comparison method | MPS/CPU plain PyTorch, or other | Integration effort |
| --- | --- | --- | --- | --- | --- | --- |
| hr98w/jev-visual | 0.8B, ~596 MiB (4-bit) | General typed-decision reader, zero-shot, not IQA-trained | Plausible: its candidate-statement scaffolding is built for exactly this pattern (one statement per level), just not demonstrated on IQA | Zero-label: port our rubric text into its candidate-statement format; apply glance's own calibration on top of its raw scores to compare fairly, or report both raw and calibrated | Needs MLX, not plain PyTorch: a real framework mismatch with the rest of glance's PyTorch/MPS stack | M: port to `transformers`/PyTorch or run MLX as a side process; 0.8B is far below our 4B, likely an easy strawman rather than a serious baseline |
| IamBusy/OpenJev-Vision | DINOv2-small (~22M) + small CNN, likely well under 500 MB total | General typed-decision engine on the text side, but its vision heads are task-specific trained classifiers, not IQA | No as shipped: each head is trained for one fixed task (Pets breed, CLEVR-4 property); a new rubric needs a new trained head, not a prompt change | Not zero-label: would need to train a new head per rubric on our calibration split, a heavier ask than glance's linear calibration fit | Plain PyTorch (DINOv2 via `transformers` + a small CNN), MPS/CPU fine | L: training a new head per rubric, plus the CC-licensed checkpoints must be avoided, leaving only the Apache-2.0 synthetic-CNN track as license-clean |
| AlexWortega/openjev (v2) | 4B (Qwen3.5-4B based); ~8 GB fp16 typical for a 4B checkpoint, not independently confirmed | General NLI cross-encoder, zero-shot, not IQA-trained | Plausible via the same per-level true/false pattern glance's `independent` method uses; not demonstrated on ordered rating scales in its own benchmarks | Zero-label: state each rubric level as a claim, read P(true) per level the way glance's `independent` readout does, then fit glance's own calibration on our calibration split for a like-for-like comparison | Should be plain `transformers`/PyTorch if a standard HF checkpoint is published (not independently confirmed today), likely MPS-compatible like our own Qwen3-VL-4B | M: confirm the vision tokenizer path loads on `transformers`, then reuse glance's own prompt-per-level and calibration code almost unchanged |
| zhengxuyu/litjev | No vision weights ship; text default is 27B | N/A, no runnable vision model | N/A | N/A | Needs CUDA H100 for its only tested (text) checkpoint | Not integrable as shipped: no vision checkpoint to point it at |

## D. Hosted APIs (context only, not runnable under the license rule)

| Name | What it is | Input $/1M tok | Output $/1M tok | Source (fetched today) |
| --- | --- | --- | --- | --- |
| TypeSafe Jev (1.13) | Typed-decision API (Noul/Choice/Score); docs state text only, images/audio/video "not supported yet." Confirmed again today: `docs.typesafe.ai/pricing` 404s, no primary price page found; figure below is the one `RELATED_WORK.md` already sourced. | $0.042 | $0 (free) | theregister.com (16 Sept 2026), cross-checked today via search against mindstudio.ai and juliangoldie.com summaries, no primary source located |
| Anthropic Claude Opus 5 | Frontier multimodal API. | $5.00 | $25.00 | claude.com/pricing (fetched today) |
| Anthropic Claude Sonnet 5 | Frontier multimodal API. | $2.00 | $10.00 | claude.com/pricing (fetched today) |
| OpenAI GPT-6 Astra | Frontier multimodal API. | $10.00 | $50.00 | developers.openai.com/api/docs/pricing (fetched today) |
| OpenAI GPT-5.6 Terra | Mid-tier multimodal API. | $2.00 | $12.00 | developers.openai.com/api/docs/pricing (fetched today) |
| Google Gemini 3.1 Pro (<=200k tok) | Frontier multimodal API. | $2.00 | $12.00 | ai.google.dev/gemini-api/docs/pricing (fetched today) |
| Google Gemini 3.8 Flash (through 2026-12-31) | Mid-tier multimodal API. | $0.75 | $3.75 | ai.google.dev/gemini-api/docs/pricing (fetched today) |

## B. Trained image-quality VLMs / scorers

Checked today: HF model cards and `?blobs=true` file listings, GitHub repos and LICENSE files, and arXiv HTML for
Q-Align/OneAlign, DeQA-Score, Q-Instruct, Co-Instruct, Compare2Score, VisualQuality-R1, Q-Insight, Q-SiT-mini, and
Qwen2.5-VL-3B/7B/72B-Instruct license tags. Cross-cutting finding: Q-Align, DeQA-Score, Compare2Score, Co-Instruct
and the released Q-Instruct checkpoints all self-tag "MIT" or "apache-2.0" on Hugging Face, but their real base model
is mPLUG-Owl2 or LLaVA-v1.5, both LLaMA-2 derivatives; a downstream repo's own tag does not override the LLaMA 2
Community License on the weights it was built from. Qwen2.5-VL-7B-Instruct is Apache-2.0; Qwen2.5-VL-3B-Instruct is
`qwen-research` and Qwen2.5-VL-72B-Instruct is the `qwen` custom license, both confirmed non-Apache/MIT today.

| Name + URL | What it is | Code license | Weights license (base model) | Passes rule? |
| --- | --- | --- | --- | --- |
| Q-Align / OneAlign. huggingface.co/q-future/one-align, ICML 2024, arxiv.org/abs/2312.17090 | Fine-tunes an LMM on discrete text-defined levels (excellent/good/fair/poor/bad), reads level-token probabilities and their weighted average as score. | S-Lab License 1.0, explicitly non-commercial | HF tag says MIT; real base is `MAGAer13/mplug-owl2-llama2-7b` -> LLaMA 2 Community License | No |
| DeQA-Score. github.com/zhiyuanyou/DeQA-Score, CVPR 2025, arxiv.org/abs/2501.11561 | Trains with soft labels over level tokens so the predicted distribution matches the human score distribution. | MIT | HF tag says MIT; base is the same mPLUG-Owl2-LLaMA2-7B -> LLaMA 2 Community License | No |
| Q-Instruct. github.com/Q-Future/Q-Instruct, CVPR 2024 | Low-level-vision instruction-tuning data + fine-tuned mPLUG-Owl2 / LLaVA-v1.5 / InternLM-XC checkpoints; a Q&A model, not itself a rubric scorer. | LICENSE + S-Lab-LICENSE (non-commercial); text also requires emailing the team for commercial use | mPLUG-Owl2 and LLaVA-v1.5 checkpoints both resolve to LLaMA-2/Vicuna-LLaMA | No |
| Co-Instruct. huggingface.co/q-future/co-instruct, arxiv.org/abs/2402.16641 | Open-ended visual quality comparison across one or more images. | Not stated | No license tag at all in cardData; base is mplug_owl2 -> LLaMA 2 Community License regardless | No |
| Compare2Score. huggingface.co/q-future/Compare2Score, NeurIPS 2024, arxiv.org/abs/2405.19298 | Trains on pairwise comparisons against anchor images, MAP-estimates a continuous score at inference. | MIT | HF tag says MIT; architecture is `MPLUGOwl2LlamaForCausalLM` -> LLaMA 2 Community License | No |
| VisualQuality-R1. huggingface.co/TianheWu/VisualQuality-R1-7B, NeurIPS 2025, arxiv.org/abs/2505.14460 | RL-to-rank (GRPO + Thurstone model) IQA model that rates and optionally reasons about quality. | Apache-2.0 | MIT-tagged weights on `Qwen/Qwen2.5-VL-7B-Instruct`, confirmed Apache-2.0 today | Yes |
| Q-Insight. huggingface.co/ByteDance/Q-Insight, arxiv.org/abs/2503.22679 | Multitask GRPO-RL model: score regression, degradation-type perception, pairwise comparison reasoning. | Apache-2.0 | Apache-2.0, base is `Qwen/Qwen2.5-VL-7B-Instruct` | Yes |
| Q-SiT-mini. huggingface.co/zhangzicheng/q-sit-mini, arxiv.org/abs/2503.09197 | Small LMM taught to score AND explain quality on a fixed 5-level scale. | MIT | MIT, base is LLaVA-OneVision-Qwen2, which is Apache-2.0 (confirmed today) | Likely yes |
| Q-Ponder. github.com/vivoCameraResearch/Q-Ponder, arxiv.org/abs/2506.05384 | Qwen2.5-VL distilled from a 72B teacher for quality reasoning. | Not verified | Weights marked "coming soon", not released | Cannot evaluate; not released |

| Name | Params / size | IQA-trained or general? | Arbitrary K-level rubric? | MPS/CPU or CUDA-only | Published KADID-10k (or similar) numbers | Integration effort |
| --- | --- | --- | --- | --- | --- | --- |
| Q-Align / OneAlign | ~8B-class (mPLUG-Owl2 7B), ~16.4 GB | IQA/IAA/VQA specific | No: fixed 5 levels (excellent...bad) baked in | Needs `flash_attn` (CUDA-only) for training; base inference path not documented for MPS | KADID-10k SRCC 0.684-0.934/PLCC 0.674-0.935 depending on training mix (arxiv.org/html/2312.17090 Tables 3-4) | Fails license rule; cite only |
| DeQA-Score | Same base, ~16.4 GB | IQA specific | No: soft labels over the same kind of fixed levels | Same CUDA-only training deps; inference path unconfirmed for MPS | KADID-10k SRCC 0.687-0.953/PLCC 0.694-0.955 depending on training mix (arxiv.org/html/2501.11561) | Fails license rule; cite only |
| Q-Instruct checkpoints | 7B-class per checkpoint | General low-level-vision Q&A, quality-adjacent | Closer to arbitrary (free-form Q&A) but no logit-readout interface confirmed | Repo notes "only single GPU inference is supported" for mPLUG-Owl2 | Not verified (not found in fetched content) | Fails license rule |
| Co-Instruct | ~8B-class, ~16 GB | IQA comparison/description | Open-ended text, not a K-level logit readout | Same mPLUG-Owl2 stack, MPS unconfirmed | Not found in fetched content | Fails license rule |
| Compare2Score | 7B-class, ~32.8 GB (4 shards, larger than Q-Align's) | IQA specific | No: fixed "quality score of this image is {}" output, needs anchor images | `flash_attn` (CUDA-only) for training; inference unconfirmed for MPS | KADID-10k SRCC 0.952/PLCC 0.939, median over 10 LIQE-protocol splits (arxiv.org/html/2405.19298) | Fails license rule; cite only |
| VisualQuality-R1 | ~7-8B, ~16.6 GB (4 safetensors shards, confirmed via HF file listing) | IQA specific, RL-trained | No: fixed 1-5 numeric rating plus optional reasoning text | vLLM/flash-attn are optional accelerators per its docs; plain `transformers` load plausible on MPS, not vendor-confirmed | KonIQ-10k SRCC 0.830-0.855/PLCC 0.840-0.870, SPAQ SRCC 0.875-0.913/PLCC 0.878-0.917 depending on training mix; no KADID-10k test number found, though KADID-10k is one of its training sets (arxiv.org/html/2505.14460 Table 2) | S-M: passes rule, plain HF load, but reads a generated number, not a logit, so it departs from glance's no-generation design |
| Q-Insight | 7B-class, ~19.6 GB per checkpoint variant | IQA specific, RL-trained | Not verified: score regression, degradation typing, comparison reasoning, no confirmed K-level rubric interface | No CUDA-only kernel found in fetched content beyond standard training deps; Qwen2.5-VL base suggests MPS is plausible | Not verified: abstract claims SOTA but exact SRCC/PLCC table was not extracted this session | M: passes rule, but the actual inference interface and benchmark numbers need a follow-up read before committing to it |
| Q-SiT-mini | 0.9B, cheapest of this group | IQA specific | No: fixed 5-level scale (Excellent/Good/Fair/Poor/Bad), same design as Q-Align | Requires `transformers==4.45.0`; no CUDA-only kernel found in the fetched excerpt | Not found in fetched content | S: likely passes rule, tiny, but benchmark numbers and exact MPS behavior need a follow-up check |
| Q-Ponder | Not released | IQA specific (once released) | Not verified | Not verified | Not verified | Cannot integrate: weights not public |

## C. Frozen-encoder zero-label scorers

Checked today: arXiv/paper pages, `github.com/IceClear/CLIP-IQA`, `github.com/chaofengc/IQA-PyTorch` (pyiqa, the implementation almost
everyone actually uses) and its LICENSE and `clip_model.py`/`clipiqa_arch.py`/`default_model_configs.py`, `github.com/zwx8981/LIQE`
and its LICENSE and code, `huggingface.co/google/siglip-base-patch16-224` and `siglip2-base-patch16-224`,
`huggingface.co/somepago/AestheticSigLIP`, and arXiv 2509.17374 (WACV 2026 SigLIP2-IQA probe).

| Name + URL | What it is | Code license | Weights license (base model) | Passes rule? |
| --- | --- | --- | --- | --- |
| CLIP-IQA (official). arxiv.org/abs/2207.12396, github.com/IceClear/CLIP-IQA | Frozen CLIP, antonym prompt pairs ("Good photo."/"Bad photo."), softmax over cosine similarities, zero IQA labels. | S-Lab License 1.0, explicitly non-commercial | OpenAI CLIP RN50; no OSI license, model card states "any deployed use case... is currently out of scope" | No |
| CLIP-IQA / CLIP-IQA+ (pyiqa / IQA-PyTorch, the common implementation). github.com/chaofengc/IQA-PyTorch | Same idea; CLIP-IQA+ adds a small CoOp-style learned prompt context (16 context vectors) trained on KonIQ-10k, CLIP backbone stays frozen. | PolyForm Noncommercial 1.0.0 + S-Lab License | Same OpenAI CLIP checkpoints (downloaded from `openaipublic.azureedge.net`); CLIP-IQA+'s extra learned-prompt weights are hosted CC-BY-NC-SA-4.0 | No |
| LIQE. github.com/zwx8981/LIQE (CVPR 2023, arXiv 2303.14968) | Multitask BIQA: jointly predicts quality, scene, distortion type via CLIP image-text correspondence. | MIT | Fully fine-tuned OpenAI CLIP ViT-B/32; no separate weight license published, so it inherits CLIP's research-only stance | No / unclear |
| SigLIP2 base encoder, no packaged IQA scorer exists. huggingface.co/google/siglip2-base-patch16-224 | Google's sigmoid-loss CLIP variant. A CLIP-IQA-style antonym-prompt scorer would have to be hand-built; none was found published. | Apache-2.0 (`transformers`) | Apache-2.0, confirmed on the HF card | Yes for the encoder; nothing runnable ships today |

| Name | Params / size | IQA-trained or general | Arbitrary K-level rubric? | Fair-comparison method | MPS/CPU or CUDA-only | Published KADID-10k (or similar) numbers | Integration effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP-IQA | RN50 ~102M (~350 MB); other backbones up to ViT-L/14 ~428M (~1.7 GB) | Zero-shot, no IQA training at all | No: fixed 2-way antonym softmax per attribute, not a K-level scale | Zero-label, but needs a hand-built map from the 2-way score to our K levels | Plain PyTorch, MPS/CPU fine | Not evaluated on KADID-10k in the paper (KonIQ-10k, LIVE-itW, SPAQ, TID2013, restoration sets only); no KADID-10k number found anywhere | S if the license didn't block it |
| CLIP-IQA+ (pyiqa) | Same backbones + a few thousand learned-prompt params | Small IQA-supervised prompt tuning on KonIQ-10k, backbone frozen | No: same fixed 2-way softmax | Same as CLIP-IQA, and it already saw KonIQ-10k human labels during prompt tuning | Plain PyTorch, MPS/CPU fine | None found for KADID-10k | S if the license didn't block it |
| LIQE | ViT-B/32-scale, ~151M (LIQE's own exact count not independently confirmed) | Fully supervised multitask fine-tune, not zero-shot | No: fixed quality-level + scene + distortion vocabulary baked in at training time | Not zero-label (already IQA-trained); would need a remap from its fixed vocabulary to our levels and a check for training overlap with any shared images | Plain PyTorch (`clip.load` + `nn.Module`), MPS/CPU fine | Only zero-shot cross-database SRCC on AIGC sets (AGIQA-3K 0.7212, AGIQA-1K 0.5785, SJTU-H3D 0.6716, AIGCIQA2023 0.7435) was found; no KADID-10k row seen in the README | M, license blocks it anyway |
| SigLIP2-base, hand-built CLIP-IQA-style scorer (does not exist yet) | google/siglip2-base-patch16-224, ~400M (~0.8-1.6 GB fp16/fp32) | Would be zero-shot if we write it with fixed antonym prompts, no IQA training | No, unless we extend the antonym-softmax to K prompt pairs ourselves | Genuinely zero-label and license-clean, but it is a paper-original baseline we would have to write, not a citation of someone else's numbers | Plain PyTorch (`SiglipModel`), MPS/CPU fine | None to cite, nothing published in this exact configuration | M: no published code to adapt, `transformers.SiglipModel` + our own antonym prompts per rubric |

Also checked and set aside: `somepago/AestheticSigLIP` (Apache-2.0, fine-tuned SigLIP2-So400M, but a fixed 1-10 ordinal aesthetic
scale, not zero-shot, not a K-level rubric reader; SRCC 0.671 / MAE 1.04 vs. Qwen3-VL-32B on its own 1,967-image eval set, not
KADID-10k); arXiv 2509.17374 (WACV 2026), which probes SigLIP2-SO400M for IQA with a trained 3-layer MLP head, i.e. supervised
probing, not zero-shot, and reports no KADID-10k numbers either.

## Recommended comparison set

Four systems, chosen to cover the four shapes of alternative: a zero-shot same-primitive Jev reproduction, a tiny
trained scorer whose fixed levels resemble our own rubric shape, a larger RL-trained scorer, and a zero-label frozen
dual-encoder baseline. All four pass the Apache-2.0/MIT rule and should load with plain `transformers` on MPS. None
of their exact commit revisions were pinned today; pin the HF `main` commit sha at integration time and record it.

**1. AlexWortega/openjev (v2).** Pin `AlexWortega/openjev` at whatever commit is current when integration starts (not
independently confirmed today). Download: 4B params, roughly 8 GB fp16 (typical for this size, not independently
measured). Adapter: (1) load with `transformers`, resize each manifest image to 448 px longest side, same as glance;
(2) for each rubric level, write the claim "this image shows {level description}" and read its NLI head's P(true),
never generating; (3) treat the K per-level probabilities the way glance's `independent` readout does; (4) fit
glance's own matrix-scaling calibration on the manifest's `calibration` split; (5) score the `test` split, write
`item_id`, predicted level and probabilities to a predictions file in glance's own schema. Embarrassing for us: if
raw or calibrated openjev claims match or beat `ens4d` (0.867 mean accuracy), it says the composition glance adds on
top of the shared P(true) primitive is not doing much. Supportive: if openjev lands near glance's v0-as-shipped
(0.500) until it gets the same kind of calibration, that shows the gain is in the composition, not in having any VLM
read logits at all.

**2. zhangzicheng/q-sit-mini.** Pin at current `main`. Download: 0.9B params (smallest of the four; exact GB not
independently confirmed). Adapter: (1) load via `transformers` (LLaVA-OneVision architecture), same 448 px resize;
(2) prompt with Q-SiT's own fixed "Excellent/Good/Fair/Poor/Bad" scale and read logits over those five level tokens,
mirroring glance's `digits` readout; (3) map its 5 fixed levels onto our K levels (identity for our 5-level KADID-
style ladders, a recorded monotonic collapse for our 4-level synthetic ladders); (4) optionally fit glance's own
calibration on top, to separate "their training" from "our calibration" as two axes; (5) score `test`, write
predictions in the same schema. Embarrassing for us: if a 0.9B fine-tuned model beats `ens4d` zero-shot, with no
calibration data of ours at all, "just fine-tune a small model" beats "keep it frozen and calibrate." Supportive: if
its fixed IQA-jargon levels map poorly onto our scale-specific wording and it trails glance's calibrated numbers,
that supports asking the rubric in its own words rather than reusing fixed quality vocabulary.

**3. TianheWu/VisualQuality-R1-7B.** Pin at current `main`. Download: ~16.6 GB (4 safetensors shards, confirmed via
the HF file listing today: 4.97+4.99+4.93+1.69 GB). Adapter: (1) load via `transformers` (Qwen2.5-VL-7B-Instruct
architecture), 448 px resize; (2) it has no documented single-token score head, so take its generated 1-5 rating at
face value per item, noting explicitly that this departs from glance's no-generation design; (3) fit a monotonic
linear map from its raw 1-5 output onto our K levels on the `calibration` split, the closest analog to glance's own
calibration; (4) if the tokenizer gives single-token digits, also try reading logits at the answer position and
report whether that was possible; (5) score `test`, write predictions, and time it separately since it is roughly
1.75x our own model's parameter count. Embarrassing for us: a wide win for a model trained on real IQA data (KADID-
10k is one of its own training sets) over reading logits off a frozen general VLM undercuts the training-free pitch.
Supportive: if its natural-photo training distribution transfers poorly to our specific 448 px degradation ladders
while glance's per-rubric calibration (fit on a few dozen of our own labels) adapts better, that is a direct
calibration-over-training result.

**4. A hand-built CLIP-IQA-style scorer on google/siglip2-base-patch16-224.** Pin at current `main`. Download: ~400M
params, roughly 0.8-1.6 GB. This does not exist as a published implementation; we would write it. Adapter: (1) load
`SiglipModel`, encode each manifest image once, frozen, no calibration data needed for the raw version; (2) write one
antonym-style prompt pair per rubric level, drawn from our own rubric wording, the way CLIP-IQA writes "good photo"/
"bad photo"; (3) report the raw softmax-over-cosine-similarity result as a genuine zero-label baseline, no fitted
parameters; (4) optionally fit glance's matrix-scaling calibration on top of the K similarities, to isolate how much
of glance's gain is "any calibration on any frozen scorer" versus the VLM-specific choice; (5) score `test`, write
predictions, keep the prompt set in the repo for reproducibility. Embarrassing for us: if the raw, zero-label
baseline lands anywhere near glance's 0.500-0.810 range, a five-minute frozen dual-encoder scorer gets most of the
way there without labels or VLM machinery. Supportive: if it is near chance (0.20-0.25 for 4-5 levels) and only
partially recovers with calibration, while still trailing glance's VLM numbers, that shows the specific choice of a
jointly-attending VLM reading a described rubric is doing real work, not just "calibration fixes any frozen scorer."

## Cite, do not run

These fail the license rule (LLaMA-2-derived base models, or OpenAI CLIP's research-only weight terms) or have no
released weights at all. Their numbers can be quoted with the caveats below; none was reproduced by us.

| System | Published number | Split / protocol | Source (fetched today) | Caveat |
| --- | --- | --- | --- | --- |
| Q-Align / OneAlign | KADID-10k SRCC 0.684-0.934, PLCC 0.674-0.935, depending on training mix | Trained on KonIQ, SPAQ, or a mixed set; tested on KADID-10k | arxiv.org/html/2312.17090, Tables 3-4 | Continuous MOS regression on human-annotated natural-image distortions, not our per-scale ordinal level classification with a held-out calibration split; its 5 levels are fixed at training time, not worded per rubric like ours; LLaMA-2 base fails our rule |
| DeQA-Score | KADID-10k SRCC 0.687-0.953, PLCC 0.694-0.955, depending on training mix | Single-dataset vs. KonIQ+SPAQ+KADID co-trained | arxiv.org/html/2501.11561 | Same LLaMA-2 base issue; distribution-regression, not classification; its own KADID number is partly in-domain when KADID is in the training mix |
| Compare2Score | KADID-10k SRCC 0.952, PLCC 0.939 | Median over ten 70/10/20 train/val/test splits (LIQE's protocol) | arxiv.org/html/2405.19298, Tables 1-2 | LLaMA-2 base fails our rule; needs anchor images at inference, a different mechanism than one forward pass; trained on KADID-10k itself, so in-domain |
| CLIP-IQA | No KADID-10k number published; paper evaluates KonIQ-10k, LIVE-itW, SPAQ, TID2013 and restoration sets instead (exact figures not extracted this session, not verified) | Zero-shot | arxiv.org/abs/2207.12396 | Zero-shot like glance in spirit, but a fixed 2-way antonym softmax, not a K-level rubric reader; OpenAI CLIP weights carry a research-only stance, not Apache/MIT |
| LIQE | No KADID-10k number found; only zero-shot cross-database SRCC on AIGC sets (AGIQA-3K 0.7212, AGIQA-1K 0.5785, SJTU-H3D 0.6716, AIGCIQA2023 0.7435) | Zero-shot cross-dataset | github.com/zwx8981/LIQE README | Fully fine-tuned CLIP with an unspecified weight license; fixed quality+scene+distortion vocabulary; the only numbers found are on AIGC images, not KADID-10k or photographic ladders like ours, so not usable as a KADID reference point |
| Laya Vision (`thaitea/laya-vision-smolvlm-256m`) | own validation: 75.9% accuracy, ECE 0.035 over 8,235 samples; VQAv2 yes/no 73.2%, A-OKVQA 63.1%, ScienceQA 89.0% | its own held-out split of A-OKVQA, ScienceQA images, VQAv2 yes/no | huggingface.co/thaitea/laya-vision-smolvlm-256m, github.com/r33drichards/laya-vision (fetched 2026-09-20) | Trained 256M typed-decision model (vision tower frozen, LM + decision head trained). WEIGHTS are CC BY-NC-SA 4.0 (ScienceQA), which fails our license rule, so it is cited, not run. `score` is untrained by its own card, so only yes/no and pick-one could ever be compared. |
| YOFO (arXiv:2511.16600) | ranking error rate and throughput on LAION-RVS-Fashion (figures not extracted) | fashion reranking: does this item satisfy each of N requirements | arxiv.org/abs/2511.16600 v3 (fetched 2026-09-20) | LoRA-trained Qwen2-VL-2B / Qwen3-VL-2B, binary requirements only, all requirements in one packed sequence. No released checkpoint is named in the paper, so there is nothing to run. Closest published relative of our yes/no path; different object (a trained judge). |
| OpenJev on DiffusionGemma (`razorback16/openjev`) | none on images found | - | github.com/razorback16/openjev (fetched 2026-09-20) | Jev-wire-compatible server, accepts up to 8 images per request, Apache-2.0, but runs DiffusionGemma 26B-A4B through an unmerged vLLM pull request (vllm-project/vllm#57250): not runnable on a 32 GB laptop. |

Also cite-only with no comparable published numbers found: Q-Instruct and Co-Instruct (both LLaMA-2-derived, no
benchmark figures located in what was fetched today); the CC-licensed heads of IamBusy/OpenJev-Vision (Pets/CLEVR-4
classification numbers exist but are not an IQA metric); zhengxuyu/litjev (ships no vision weights to cite numbers
for); Q-Ponder (weights not released). DistortBench (arXiv 2604.19966) is already covered in `RELATED_WORK.md` as
the nearest external benchmark to our scales and is not repeated here.

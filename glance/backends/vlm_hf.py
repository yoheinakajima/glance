"""Open VLM backend with logit readout (HANDOFF section 6). Nothing is decoded.

One statement is one forward pass read at a single token position: the next-token logits after an empty
assistant turn. Two ways to get there:

- reference path: every statement is a full prompt, batched with left padding. This is the correctness oracle.
- cached path: run the shared prefix (template head, image tokens, context) once with use_cache=True, expand
  the KV cache across the batch, then run only the statement suffixes. Qwen-VL uses multimodal RoPE, so suffix
  position ids continue from the prefix's rope deltas.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import prompts
from ..config import Config
from ..doctor import run_doctor
from ..images import LoadedImage
from ..logging_utils import install_mps_fallback_logger
from ..schema import BackendError
from .base import BackendUsage, LabelScores, Statement, StatementScores

ASSISTANT_HEADER = "<|im_start|>assistant\n"
MIN_IMAGE_TOKENS = 64  # the processor's own floor (65536 px); lowered when the per-image budget is smaller
PREFIX_CACHE_ENTRIES = 2


@dataclass
class _Readout:
    selected: np.ndarray  # [n, m] raw logits at the requested token ids, float64
    log_norm: np.ndarray  # [n] logsumexp over the whole vocabulary
    image_tokens: int
    text_tokens: int
    prefix_ms: float
    score_ms: float
    cache_hit: bool | None


@dataclass
class _PrefixEntry:
    ids: list[int]
    kv: list[tuple[Any, Any]]  # per layer (keys, values), batch 1
    rope_delta: int


class VlmBackend:
    name = "vlm"
    kind = "vlm"

    def __init__(self, cfg: Config, device: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        report = run_doctor(cfg)
        tier = report["selected_tier"]
        if tier not in cfg.models.vlm_tiers:
            raise BackendError(f"no VLM tier for this device (tier `{tier}`): use the dual encoder", {"doctor": report})
        spec = cfg.models.vlm_tiers[tier]
        self.tier = tier
        self.model_id = spec.id
        self.revision = spec.revision
        self.dtype = spec.dtype
        self.device = device or report["device"]
        self.image_token_budget = cfg.models.image_token_budget_override or spec.image_token_budget
        self.batch_size = cfg.vlm.batch_size
        self.suffix_batch_size = cfg.vlm.suffix_batch_size
        self.use_prefix_cache = cfg.vlm.prefix_cache
        install_mps_fallback_logger(cfg.path("logs"))

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.tokenizer = self.processor.tokenizer
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.model_id, revision=self.revision, dtype=getattr(torch, self.dtype)
            )
            .to(self.device)
            .eval()
        )
        self.load_ms = (time.perf_counter() - t0) * 1000

        ip = self.processor.image_processor
        self.pixels_per_token = (ip.patch_size * ip.merge_size) ** 2  # derived, never hard-coded
        self.merge_length = ip.merge_size**2
        self.image_token = self.processor.image_token
        self.image_token_id = self.processor.image_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self.yes_ids = self._single_token_ids(prompts.YES_VARIANTS)
        self.no_ids = self._single_token_ids(prompts.NO_VARIANTS)
        self._label_ids = {lab: self._single_token_ids([lab, " " + lab]) for lab in prompts.LETTER_LABELS}
        self._prefix_cache: OrderedDict[str, _PrefixEntry] = OrderedDict()

    # --- prompt building --------------------------------------------------------------------------

    def _single_token_ids(self, variants: list[str]) -> list[int]:
        ids = []
        for text in variants:
            enc = self.tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        if not ids:
            raise BackendError(f"none of {variants} is a single token for {self.model_id}")
        return ids

    def render_prompt(self, images: list[LoadedImage], context: dict[str, Any] | None, block: str) -> str:
        """User content order: image(s), context JSON, then the statement block. The assistant turn is left empty."""
        content: list[dict[str, Any]] = []
        for img in images:
            content.append({"type": "text", "text": prompts.image_label(img.id)})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompts.context_block(context) + block})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        if not text.endswith(ASSISTANT_HEADER):
            raise BackendError("rendered prompt does not end with the assistant header")
        if "<think>" in text or "</think>" in text:
            raise BackendError("rendered prompt contains thinking tags; use the Instruct checkpoint")
        return text

    def _encode_images(self, images: list[LoadedImage]):
        """Run the image processor once per request, holding each image to its share of the token budget."""
        per_image = max(1, self.image_token_budget // len(images))
        size = {
            "shortest_edge": min(MIN_IMAGE_TOKENS, per_image) * self.pixels_per_token,
            "longest_edge": per_image * self.pixels_per_token,
        }
        out = self.processor.image_processor(images=[img.image for img in images], size=size, return_tensors="pt")
        grid = out["image_grid_thw"]
        tokens = [int(g.prod()) // self.merge_length for g in grid]
        if any(t > per_image for t in tokens):
            raise BackendError(f"image tokens {tokens} exceed the per-image budget {per_image}")
        for img, n in zip(images, tokens):
            img.image_tokens = n
        return out["pixel_values"], grid, tokens

    def _tokenize(self, texts: list[str], tokens_per_image: list[int]) -> list[list[int]]:
        """Expand each image placeholder to its token count (as the processor does), then tokenize."""
        expanded = []
        for text in texts:
            parts = text.split(self.image_token)
            if len(parts) != len(tokens_per_image) + 1:
                raise BackendError("prompt has a different number of image placeholders than images")
            rebuilt = parts[0]
            for n, tail in zip(tokens_per_image, parts[1:]):
                rebuilt += self.image_token * n + tail
            expanded.append(rebuilt)
        return self.tokenizer(expanded, add_special_tokens=False)["input_ids"]

    # --- forward paths ----------------------------------------------------------------------------

    def _collect(self, logits, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        logits = logits.float()
        selected = logits[:, token_ids].cpu().numpy().astype(np.float64)
        log_norm = torch.logsumexp(logits, dim=-1).cpu().numpy().astype(np.float64)
        return selected, log_norm

    def _reference(self, ids: list[list[int]], pixel_values, grid, token_ids: list[int]):
        """Every statement is a full prompt, batched with left padding."""
        import torch

        selected, log_norm = [], []
        for start in range(0, len(ids), self.batch_size):
            chunk = ids[start : start + self.batch_size]
            width = max(len(row) for row in chunk)
            input_ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, row in enumerate(chunk):
                input_ids[i, width - len(row) :] = torch.tensor(row)
                attention[i, width - len(row) :] = 1
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention.to(self.device),
                    mm_token_type_ids=(input_ids == self.image_token_id).long(),
                    pixel_values=pixel_values.repeat(len(chunk), 1).to(self.device),
                    image_grid_thw=grid.repeat(len(chunk), 1).to(self.device),
                    logits_to_keep=1,
                )
            sel, norm = self._collect(out.logits[:, -1], token_ids)
            selected.append(sel)
            log_norm.append(norm)
        return np.concatenate(selected), np.concatenate(log_norm)

    def _prefix_key(self, images: list[LoadedImage], tokens_per_image: list[int]) -> str:
        raw = "|".join(f"{img.sha256}:{n}" for img, n in zip(images, tokens_per_image))
        return hashlib.sha256(raw.encode()).hexdigest()

    def _run_prefix(self, prefix_ids: list[int], pixel_values, grid) -> _PrefixEntry:
        import torch

        input_ids = torch.tensor([prefix_ids], device=self.device)
        with torch.no_grad():
            out = self.model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=(input_ids == self.image_token_id).long(),
                pixel_values=pixel_values.to(self.device),
                image_grid_thw=grid.to(self.device),
                use_cache=True,
            )
        kv = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
        return _PrefixEntry(ids=list(prefix_ids), kv=kv, rope_delta=int(self.model.model.rope_deltas.reshape(-1)[0]))

    def _expanded_cache(self, entry: _PrefixEntry, length: int, batch: int):
        """A fresh cache whose layers are batch-expanded views of the prefix. `update` concatenates, so the
        shared prefix tensors are never written to."""
        from transformers import DynamicCache

        cache = DynamicCache(config=self.model.config.get_text_config())
        for layer, (keys, values) in zip(cache.layers, entry.kv):
            layer.lazy_initialization(keys, values)
            layer.keys = keys[:, :, :length].expand(batch, -1, -1, -1)
            layer.values = values[:, :, :length].expand(batch, -1, -1, -1)
        return cache

    def _cached(self, ids: list[list[int]], images, pixel_values, grid, tokens_per_image, token_ids: list[int]):
        """Shared prefix once, then only the statement suffixes (right padded; logits read at each last token)."""
        import torch

        shortest = min(len(row) for row in ids)
        shared = 0
        while shared < shortest - 1 and all(row[shared] == ids[0][shared] for row in ids):
            shared += 1
        vision_end = max(i for i, tok in enumerate(ids[0]) if tok == self.image_token_id) + 1
        if shared < vision_end:
            raise BackendError("statements do not share the image prefix; cannot use the prefix cache")

        t0 = time.perf_counter()
        key = self._prefix_key(images, tokens_per_image)
        entry = self._prefix_cache.get(key)
        length = 0
        if entry is not None:
            limit = min(len(entry.ids), shared)
            while length < limit and entry.ids[length] == ids[0][length]:
                length += 1
        hit = entry is not None and length >= vision_end
        if hit:
            self._prefix_cache.move_to_end(key)
        else:
            entry = self._run_prefix(ids[0][:shared], pixel_values, grid)
            length = shared
            self._prefix_cache[key] = entry
            while len(self._prefix_cache) > PREFIX_CACHE_ENTRIES:
                self._prefix_cache.popitem(last=False)
        if self.device == "mps":
            torch.mps.synchronize()
        prefix_ms = (time.perf_counter() - t0) * 1000

        selected, log_norm = [], []
        for start in range(0, len(ids), self.suffix_batch_size):
            suffixes = [row[length:] for row in ids[start : start + self.suffix_batch_size]]
            batch, width = len(suffixes), max(len(s) for s in suffixes)
            input_ids = torch.full((batch, width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((batch, width), dtype=torch.long)
            for i, suffix in enumerate(suffixes):
                input_ids[i, : len(suffix)] = torch.tensor(suffix)
                mask[i, : len(suffix)] = 1
            attention = torch.cat([torch.ones((batch, length), dtype=torch.long), mask], dim=1).to(self.device)
            text_pos = (length + torch.arange(width)).expand(batch, -1)
            # Row 0 is the plain text position; rows 1-3 (t, h, w) continue from the prefix's rope delta.
            position_ids = torch.stack([text_pos] + [text_pos + entry.rope_delta] * 3).to(self.device)
            with torch.no_grad():
                out = self.model.model.language_model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention,
                    position_ids=position_ids,
                    past_key_values=self._expanded_cache(entry, length, batch),
                    use_cache=True,
                )
                last = torch.tensor([len(s) - 1 for s in suffixes], device=self.device)
                hidden = out.last_hidden_state[torch.arange(batch, device=self.device), last]
                logits = self.model.lm_head(hidden)
            sel, norm = self._collect(logits, token_ids)
            selected.append(sel)
            log_norm.append(norm)
        return np.concatenate(selected), np.concatenate(log_norm), prefix_ms, hit

    def _read(self, images: list[LoadedImage], texts: list[str], token_ids: list[int]) -> _Readout:
        import torch

        t0 = time.perf_counter()
        pixel_values, grid, tokens_per_image = self._encode_images(images)
        # Score in a canonical order. Batch composition and padding perturb half-precision logits slightly, so
        # sorting makes a statement's logit independent of where the caller listed it (option order, question order).
        order = sorted(range(len(texts)), key=lambda i: texts[i])
        ids = self._tokenize([texts[i] for i in order], tokens_per_image)
        image_tokens = sum(tokens_per_image)
        text_tokens = sum(len(row) for row in ids) - image_tokens * len(ids)
        if self.use_prefix_cache:
            selected, log_norm, prefix_ms, hit = self._cached(ids, images, pixel_values, grid, tokens_per_image, token_ids)
        else:
            selected, log_norm = self._reference(ids, pixel_values, grid, token_ids)
            prefix_ms, hit = 0.0, None
        inverse = np.argsort(order)
        selected, log_norm = selected[inverse], log_norm[inverse]
        if self.device == "mps":
            torch.mps.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000
        if not np.isfinite(selected).all():
            raise BackendError("non-finite logits from the VLM forward pass")
        return _Readout(selected, log_norm, image_tokens, text_tokens, prefix_ms, total_ms - prefix_ms, hit)

    # --- Backend protocol -------------------------------------------------------------------------

    def score_statements(
        self, images: list[LoadedImage], context: dict[str, Any] | None, statements: list[Statement]
    ) -> StatementScores:
        from scipy.special import logsumexp

        texts = [self.render_prompt(images, context, s.text) for s in statements]
        read = self._read(images, texts, self.yes_ids + self.no_ids)
        z_yes = logsumexp(read.selected[:, : len(self.yes_ids)], axis=1)
        z_no = logsumexp(read.selected[:, len(self.yes_ids) :], axis=1)
        off_mass = 1.0 - np.exp(z_yes - read.log_norm) - np.exp(z_no - read.log_norm)
        return StatementScores(
            z=z_yes - z_no, z_yes=z_yes, z_no=z_no, off_mass=np.clip(off_mass, 0.0, 1.0),
            prompt_hashes=[prompts.prompt_hash(t) for t in texts],
            usage=BackendUsage(image_tokens=read.image_tokens, text_tokens=read.text_tokens, forward_passes=len(texts)),
            timing_ms={"prefix": read.prefix_ms, "score": read.score_ms},
            cache_hit=read.cache_hit,
        )

    def score_labels(
        self, images: list[LoadedImage], context: dict[str, Any] | None, blocks: list[str], labels: list[str]
    ) -> LabelScores:
        """`letter` method: logits over the label tokens (A, B, C, ...) for each rendered option block."""
        from scipy.special import logsumexp

        texts = [self.render_prompt(images, context, block) for block in blocks]
        groups = [self._label_ids[label] for label in labels]
        flat = [tid for group in groups for tid in group]
        read = self._read(images, texts, flat)
        logits = np.zeros((len(texts), len(labels)))
        cursor = 0
        for j, group in enumerate(groups):
            logits[:, j] = logsumexp(read.selected[:, cursor : cursor + len(group)], axis=1)
            cursor += len(group)
        off_mass = 1.0 - np.exp(logsumexp(logits, axis=1) - read.log_norm)
        return LabelScores(
            logits=logits, off_mass=np.clip(off_mass, 0.0, 1.0),
            prompt_hashes=[prompts.prompt_hash(t) for t in texts],
            usage=BackendUsage(image_tokens=read.image_tokens, text_tokens=read.text_tokens, forward_passes=len(texts)),
            timing_ms={"prefix": read.prefix_ms, "score": read.score_ms},
            cache_hit=read.cache_hit,
        )

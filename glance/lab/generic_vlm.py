"""A second-model backend for the lab (E3 in lab/NOTES.md entry 20): any Hugging Face image-text-to-text model that has
a chat template, read exactly as Qwen3-VL is read in `glance/backends/vlm_hf.py`, but with nothing model-specific:

- same user content order (image label, image, ..., then the statement or rating block), the model's own chat template,
  empty assistant turn;
- logits at the last position over the same token variants, from the final hidden state through a float32 copy of the
  output head, with the same log-normalizer and off-mass definition;
- one prompt per forward pass (reference path, no prefix cache, no batching, so no padding or batch-composition noise).

Nothing about the prompts or the calibration is tuned for the second model. The one configuration choice is the image
size handed to the model's image processor, fixed in advance in the notebook.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from .. import prompts
from ..backends.base import BackendUsage, LabelScores, Statement, StatementScores
from ..config import Config
from ..images import LoadedImage


class GenericVlm:
    name = "vlm2"
    kind = "vlm"

    def __init__(self, cfg: Config, model_id: str, revision: str, longest_edge: int | None = None, dtype: str = "bfloat16",
                 device: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from ..doctor import run_doctor

        self.model_id, self.revision, self.dtype = model_id, revision, dtype
        self.device = device or run_doctor(cfg)["device"]
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.tokenizer = self.processor.tokenizer
        if longest_edge:
            self.processor.image_processor.size = {"longest_edge": int(longest_edge)}
        self.image_token_budget = longest_edge
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype))
        self.model.to(self.device).eval()
        self.yes_ids = self._single_token_ids(prompts.YES_VARIANTS)
        self.no_ids = self._single_token_ids(prompts.NO_VARIANTS)
        self._label_ids: dict[str, list[int]] = {}
        self.keep_hidden = False
        self.last_hidden: np.ndarray | None = None
        self._checked_head = False

    def _single_token_ids(self, variants: list[str]) -> list[int]:
        ids = []
        for text in variants:
            enc = self.tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        if not ids:
            raise ValueError(f"none of {variants} is a single token for {self.model_id}")
        return list(dict.fromkeys(ids))

    def _messages(self, images: list[LoadedImage], context: dict[str, Any] | None, block: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for img in images:
            content.append({"type": "text", "text": prompts.image_label(img.id)})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompts.context_block(context) + block})
        return [{"role": "user", "content": content}]

    def _forward(self, images: list[LoadedImage], context, block: str, token_ids: list[int]):
        """One prompt -> (selected logits [len(token_ids)], log-normalizer, image tokens, text tokens, prompt hash, hidden)."""
        import torch

        text = self.processor.apply_chat_template(self._messages(images, context, block), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[img.image for img in images], return_tensors="pt").to(self.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(getattr(torch, self.dtype))
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1][0, -1]
            head = self.model.get_output_embeddings()
            if not self._checked_head:  # the last hidden state must be what the head multiplies (final norm included)
                gap = (head(hidden).float() - out.logits[0, -1].float()).abs().max().item()
                if gap > 0.05:
                    raise RuntimeError(f"hidden_states[-1] is not the output head's input for {self.model_id} (max gap {gap:.3f})")
                self._checked_head = True
            weight = head.weight[token_ids].float()
            selected = hidden.float() @ weight.T
            if getattr(head, "bias", None) is not None:
                selected = selected + head.bias[token_ids].float()
            others = out.logits[0, -1].float().clone()
            others[token_ids] = float("-inf")
            log_norm = torch.logaddexp(torch.logsumexp(others, dim=-1), torch.logsumexp(selected, dim=-1))
        image_token_id = getattr(self.model.config, "image_token_id", None)
        n_image = int((inputs["input_ids"] == image_token_id).sum()) if image_token_id is not None else 0
        return (selected.cpu().numpy().astype(np.float64), float(log_norm), n_image, int(inputs["input_ids"].shape[1]) - n_image,
                hashlib.sha256(text.encode()).hexdigest()[:16], hidden.float().cpu().numpy().astype(np.float16))

    def score_statements(self, images: list[LoadedImage], context, statements: list[Statement]) -> StatementScores:
        from scipy.special import logsumexp

        t0 = time.perf_counter()
        ids = self.yes_ids + self.no_ids
        rows = [self._forward(images, context, s.text, ids) for s in statements]
        sel = np.stack([r[0] for r in rows])
        log_norm = np.array([r[1] for r in rows])
        z_yes, z_no = logsumexp(sel[:, : len(self.yes_ids)], axis=1), logsumexp(sel[:, len(self.yes_ids):], axis=1)
        self.last_hidden = np.stack([r[5] for r in rows]) if self.keep_hidden else None
        return StatementScores(
            z=z_yes - z_no, z_yes=z_yes, z_no=z_no,
            off_mass=np.clip(1.0 - np.exp(z_yes - log_norm) - np.exp(z_no - log_norm), 0.0, 1.0),
            prompt_hashes=[r[4] for r in rows],
            usage=BackendUsage(image_tokens=rows[0][2], text_tokens=sum(r[3] for r in rows), forward_passes=len(rows)),
            timing_ms={"prefix": 0.0, "score": (time.perf_counter() - t0) * 1000}, cache_hit=None,
        )

    def score_labels(self, images: list[LoadedImage], context, blocks: list[str], labels: list[str]) -> LabelScores:
        from scipy.special import logsumexp

        t0 = time.perf_counter()
        for label in labels:
            if label not in self._label_ids:
                self._label_ids[label] = self._single_token_ids([label, " " + label])
        groups = [self._label_ids[label] for label in labels]
        flat = [tid for group in groups for tid in group]
        rows = [self._forward(images, context, block, flat) for block in blocks]
        logits = np.zeros((len(blocks), len(labels)))
        for i, row in enumerate(rows):
            cursor = 0
            for j, group in enumerate(groups):
                logits[i, j] = logsumexp(row[0][cursor: cursor + len(group)])
                cursor += len(group)
        log_norm = np.array([r[1] for r in rows])
        self.last_hidden = np.stack([r[5] for r in rows]) if self.keep_hidden else None
        return LabelScores(
            logits=logits, off_mass=np.clip(1.0 - np.exp(logsumexp(logits, axis=1) - log_norm), 0.0, 1.0),
            prompt_hashes=[r[4] for r in rows],
            usage=BackendUsage(image_tokens=rows[0][2], text_tokens=sum(r[3] for r in rows), forward_passes=len(rows)),
            timing_ms={"prefix": 0.0, "score": (time.perf_counter() - t0) * 1000}, cache_hit=None,
        )

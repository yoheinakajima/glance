"""Dual encoder backend (SigLIP2). z_k is the model's image-text logit, learned scale and bias included."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import Config
from ..images import LoadedImage
from ..logging_utils import install_mps_fallback_logger
from ..prompts import DUAL_ENCODER_PHOTO_PREFIX, prompt_hash
from ..schema import UnsupportedQuestionError
from .base import BackendUsage, Statement, StatementScores

MAX_TEXT_TOKENS = 64  # SigLIP2 was trained with padding="max_length", max_length=64
TEXT_CACHE_SIZE = 4096


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SiglipBackend:
    name = "siglip"
    kind = "dual_encoder"
    image_token_budget = None

    def __init__(self, cfg: Config, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.cfg = cfg
        self.model_id = cfg.models.siglip.id
        self.revision = cfg.models.siglip.revision
        self.photo_prefix = cfg.models.siglip.photo_prefix
        self.device = device or pick_device()
        self.dtype = "float32"  # 375M params: full precision everywhere keeps logits stable
        install_mps_fallback_logger(cfg.path("logs"))

        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = (
            AutoModel.from_pretrained(self.model_id, revision=self.revision, dtype=torch.float32)
            .to(self.device)
            .eval()
        )
        self.load_ms = (time.perf_counter() - t0) * 1000
        vision = self.model.config.vision_config
        self.tokens_per_image = (vision.image_size // vision.patch_size) ** 2
        self._text_cache: dict[str, Any] = {}  # text -> (normalized embedding on device, n_tokens)

    def _prepare_text(self, candidate: str) -> str:
        text = DUAL_ENCODER_PHOTO_PREFIX.format(text=candidate) if self.photo_prefix else candidate
        return text.lower()  # SigLIP2 was trained on lowercased text

    def _embed_texts(self, texts: list[str]):
        import torch

        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            if len(self._text_cache) + len(missing) > TEXT_CACHE_SIZE:
                self._text_cache.clear()
            tok = self.processor.tokenizer(
                missing, padding="max_length", max_length=MAX_TEXT_TOKENS, truncation=True, return_tensors="pt"
            )
            pad_id = self.processor.tokenizer.pad_token_id
            n_tokens = (tok["input_ids"] != pad_id).sum(dim=1).tolist()
            with torch.no_grad():
                out = self.model.get_text_features(input_ids=tok["input_ids"].to(self.device))
            emb = out if isinstance(out, torch.Tensor) else out.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            for text, e, n in zip(missing, emb, n_tokens):
                self._text_cache[text] = (e, int(n))
        embs = torch.stack([self._text_cache[t][0] for t in texts])
        return embs, sum(self._text_cache[t][1] for t in texts)

    def score_statements(
        self, images: list[LoadedImage], context: dict[str, Any] | None, statements: list[Statement]
    ) -> StatementScores:
        import torch

        if len(images) != 1:
            raise UnsupportedQuestionError(
                "the dual encoder scores one image at a time; refer to exactly one image id in the instructions",
                {"images": [img.id for img in images]},
            )
        # `context` is not used: a dual encoder embeds the image and each candidate text separately.
        t0 = time.perf_counter()
        pixels = self.processor.image_processor(images=[images[0].image], return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            out = self.model.get_image_features(pixel_values=pixels.to(self.device))
        img_emb = out if isinstance(out, torch.Tensor) else out.pooler_output
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        images[0].image_tokens = self.tokens_per_image
        t1 = time.perf_counter()

        texts = [self._prepare_text(s.candidate) for s in statements]
        text_embs, text_tokens = self._embed_texts(texts)
        with torch.no_grad():
            logits = text_embs @ img_emb[0] * self.model.logit_scale.exp() + self.model.logit_bias
        z = logits.float().cpu().numpy().astype(np.float64)
        t2 = time.perf_counter()

        return StatementScores(
            z=z,
            z_yes=z.copy(),
            z_no=None,
            off_mass=None,
            prompt_hashes=[prompt_hash(t) for t in texts],
            usage=BackendUsage(
                image_tokens=self.tokens_per_image, text_tokens=text_tokens, forward_passes=len(statements)
            ),
            timing_ms={"prefix": (t1 - t0) * 1000, "score": (t2 - t1) * 1000},
            cache_hit=None,
        )

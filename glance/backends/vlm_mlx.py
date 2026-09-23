"""Experimental Apple-Silicon VLM backend using MLX.

The public Glance contract stays `model: "vlm"`. Selecting `--backend mlx`
changes only the local runtime. The implementation is deliberately narrow:
the pinned 8-bit Qwen3-VL-2B checkpoint measured in Speedlab E017, one shared
multimodal prefix, batched statement suffixes, and selected output-head rows.
Nothing is generated.
"""

from __future__ import annotations

import importlib.util
import platform
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .. import prompts
from ..config import Config
from ..images import LoadedImage
from ..schema import BackendError
from .base import BackendUsage, LabelScores, Statement, StatementScores

ASSISTANT_HEADER = "<|im_start|>assistant\n"
VALIDATED_MODEL = "mlx-community/Qwen3-VL-2B-Instruct-8bit"
VALIDATED_REVISION = "b0338e0e843d8e1befe873d144b81fefdc47efa6"
VALIDATED_MLX_VLM = "0.7.2"


@dataclass
class _Readout:
    selected: np.ndarray
    image_tokens: int
    text_tokens: int
    prefix_ms: float
    score_ms: float


class MlxVlmBackend:
    """Glance-compatible direct scorer for the validated MLX checkpoint."""

    name = "vlm"
    kind = "vlm"
    runtime = "mlx"

    def __init__(self, cfg: Config):
        machine = platform.machine().lower()
        if platform.system() != "Darwin" or machine not in ("arm64", "aarch64"):
            raise BackendError(
                "the experimental MLX backend requires Apple Silicon; use --backend torch on this machine",
                {"system": platform.system(), "machine": platform.machine()},
            )
        missing = [name for name in ("mlx", "mlx_vlm") if importlib.util.find_spec(name) is None]
        if missing:
            raise BackendError(
                'the experimental MLX backend is not installed; run `pip install "glance-vlm[mlx]"` '
                "or `uv sync --extra mlx`",
                {"missing": missing},
            )
        try:
            installed = version("mlx-vlm")
        except PackageNotFoundError as exc:  # defensive: find_spec succeeded but package metadata did not
            raise BackendError("mlx-vlm package metadata is unavailable; reinstall the `mlx` extra") from exc
        if installed != VALIDATED_MLX_VLM:
            raise BackendError(
                f"the experimental backend requires mlx-vlm {VALIDATED_MLX_VLM}, found {installed}",
                {"required": VALIDATED_MLX_VLM, "installed": installed},
            )

        spec = cfg.models.mlx
        if (spec.id, spec.revision) != (VALIDATED_MODEL, VALIDATED_REVISION):
            raise BackendError(
                "the experimental MLX backend currently accepts only its validated pinned checkpoint",
                {"required_model": VALIDATED_MODEL, "required_revision": VALIDATED_REVISION,
                 "configured_model": spec.id, "configured_revision": spec.revision},
            )

        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.models import cache as mlx_cache
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        self.mx = mx
        self.mlx_cache = mlx_cache
        self.prepare_inputs = prepare_inputs
        self.should_add_special_tokens = should_add_special_tokens
        self.cfg = cfg
        self.model_id = spec.id
        self.revision = spec.revision
        self.dtype = spec.dtype
        self.device = "mlx-metal"
        self.image_token_budget = cfg.models.image_token_budget_override or spec.image_token_budget
        self.suffix_batch_size = cfg.vlm.suffix_batch_size

        started = time.perf_counter()
        self.model, self.processor = load(self.model_id, revision=self.revision)
        self.load_ms = (time.perf_counter() - started) * 1000
        self.tokenizer = self.processor.tokenizer
        model_type = str(getattr(self.model.config, "model_type", ""))
        tied = bool(getattr(self.model.language_model.args, "tie_word_embeddings", False))
        if "qwen3_vl" not in model_type.lower() or not tied:
            raise BackendError(
                "the experimental MLX path requires Qwen3-VL with tied input/output embeddings",
                {"model_type": model_type, "tie_word_embeddings": tied},
            )

        image_processor = self.processor.image_processor
        self.pixels_per_token = int(image_processor.patch_size) ** 2 * int(image_processor.merge_size) ** 2
        self.merge_length = int(image_processor.merge_size) ** 2
        self.image_token = self.processor.image_token
        self.image_token_id = int(self.model.config.image_token_index)
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.yes_ids = self._single_token_ids(prompts.YES_VARIANTS)
        self.no_ids = self._single_token_ids(prompts.NO_VARIANTS)
        self._row_cache: dict[tuple[int, ...], Any] = {}

    def _single_token_ids(self, variants: list[str]) -> list[int]:
        ids = []
        for text in variants:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(int(encoded[0]))
        if not ids:
            raise BackendError(f"none of {variants} is a single token for {self.model_id}")
        return ids

    def render_prompt(self, images: list[LoadedImage], context: dict[str, Any] | None, block: str) -> str:
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "text", "text": prompts.image_label(image.id)})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompts.context_block(context) + block})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        if not text.endswith(ASSISTANT_HEADER):
            raise BackendError("rendered MLX prompt does not end with the assistant header")
        if "<think>" in text or "</think>" in text:
            raise BackendError("rendered MLX prompt contains thinking tags; use the pinned Instruct checkpoint")
        return text

    def _prepare(self, images: list[LoadedImage], text: str) -> tuple[dict[str, Any], list[int]]:
        if not images:
            raise BackendError("the MLX backend requires at least one image")
        per_image = max(1, self.image_token_budget // len(images))
        processor = self.processor.image_processor
        old_min, old_max = int(processor.min_pixels), int(processor.max_pixels)
        processor.max_pixels = per_image * self.pixels_per_token
        processor.min_pixels = min(old_min, processor.max_pixels)
        try:
            prepared = self.prepare_inputs(
                self.processor,
                images=[image.image for image in images],
                prompts=text,
                image_token_index=self.image_token_id,
                add_special_tokens=self.should_add_special_tokens(
                    self.model.config.model_type, self.processor
                ),
            )
        finally:
            processor.min_pixels, processor.max_pixels = old_min, old_max

        grid = np.asarray(prepared.get("image_grid_thw"))
        if grid.ndim != 2 or len(grid) != len(images):
            raise BackendError(
                "MLX processor returned an unexpected image grid",
                {"grid_shape": list(grid.shape), "images": len(images)},
            )
        tokens = [int(np.prod(row)) // self.merge_length for row in grid]
        if any(count > per_image for count in tokens):
            raise BackendError(f"image tokens {tokens} exceed the per-image budget {per_image}")
        for image, count in zip(images, tokens):
            image.image_tokens = count
        return prepared, tokens

    def _expanded_ids(self, text: str, tokens_per_image: list[int]) -> list[int]:
        raw = self.tokenizer(text, add_special_tokens=False).input_ids
        expanded: list[int] = []
        image_index = 0
        for token in raw:
            if int(token) == self.image_token_id:
                if image_index >= len(tokens_per_image):
                    raise BackendError("prompt has more image placeholders than request images")
                expanded.extend([int(token)] * tokens_per_image[image_index])
                image_index += 1
            else:
                expanded.append(int(token))
        if image_index != len(tokens_per_image):
            raise BackendError("prompt has fewer image placeholders than request images")
        return expanded

    def _output_rows(self, token_ids: list[int]):
        key = tuple(token_ids)
        rows = self._row_cache.get(key)
        if rows is None:
            rows = self.model.language_model.model.embed_tokens(self.mx.array(token_ids))
            self.mx.eval(rows)
            self._row_cache[key] = rows
        return rows

    def _read(self, images: list[LoadedImage], texts: list[str], token_ids: list[int]) -> _Readout:
        mx = self.mx
        order = sorted(range(len(texts)), key=lambda index: texts[index])
        ordered = [texts[index] for index in order]

        prepare_started = time.perf_counter()
        first, tokens_per_image = self._prepare(images, ordered[0])
        ids = [self._expanded_ids(text, tokens_per_image) for text in ordered]
        first_ids = [int(value) for value in first["input_ids"][0].tolist()]
        if ids[0] != first_ids:
            raise BackendError("MLX prompt expansion did not reproduce processor input ids")
        shortest = min(map(len, ids))
        shared = 0
        while shared < shortest - 1 and all(row[shared] == ids[0][shared] for row in ids):
            shared += 1
        vision_end = max(index for index, token in enumerate(ids[0]) if token == self.image_token_id) + 1
        if shared < vision_end:
            raise BackendError("statements do not share the complete image prefix")
        prepare_ms = (time.perf_counter() - prepare_started) * 1000

        prefix_started = time.perf_counter()
        prefix_ids = mx.array([ids[0][:shared]])
        prefix_mask = mx.ones(prefix_ids.shape, dtype=mx.int32)
        feature_kwargs = {
            key: value for key, value in first.items()
            if key not in ("input_ids", "pixel_values", "attention_mask")
        }
        features = self.model.get_input_embeddings(
            prefix_ids, first["pixel_values"], mask=prefix_mask, **feature_kwargs
        )
        prompt_cache = self.mlx_cache.make_prompt_cache(self.model.language_model)
        self.model.language_model.model(
            prefix_ids,
            inputs_embeds=features.inputs_embeds,
            cache=prompt_cache,
            position_ids=features.position_ids,
            visual_pos_masks=features.visual_pos_masks,
            deepstack_visual_embeds=features.deepstack_visual_embeds,
        )
        mx.eval([entry.state for entry in prompt_cache])
        prefix_ms = prepare_ms + (time.perf_counter() - prefix_started) * 1000

        suffixes = [row[shared:] for row in ids]
        grid = first.get("image_grid_thw")
        positions_by_row: list[np.ndarray] = []
        for row in ids:
            full_ids = mx.array([row])
            full_mask = mx.ones(full_ids.shape, dtype=mx.int32)
            positions, _ = self.model.language_model.get_rope_index(full_ids, grid, None, full_mask)
            positions_by_row.append(np.asarray(positions)[:, 0, shared:])

        output_rows = self._output_rows(token_ids)
        selected = np.empty((len(suffixes), len(token_ids)), dtype=np.float64)
        score_started = time.perf_counter()
        for start in range(0, len(suffixes), self.suffix_batch_size):
            indexes = list(range(start, min(start + self.suffix_batch_size, len(suffixes))))
            batch, width = len(indexes), max(len(suffixes[index]) for index in indexes)
            suffix_array = np.full((batch, width), self.pad_id, dtype=np.int64)
            position_array = np.ones((3, batch, width), dtype=np.int64)
            lengths: list[int] = []
            for batch_index, statement_index in enumerate(indexes):
                suffix = suffixes[statement_index]
                lengths.append(len(suffix))
                suffix_array[batch_index, : len(suffix)] = suffix
                position_array[:, batch_index, : len(suffix)] = positions_by_row[statement_index]

            batch_cache = [type(entry).merge([entry] * batch) for entry in prompt_cache]
            suffix_ids = mx.array(suffix_array)
            right_padding = mx.array([width - length for length in lengths])
            attention = self.mlx_cache.create_causal_mask(width, offset=shared, right_padding=right_padding)
            hidden = self.model.language_model.model(
                suffix_ids,
                inputs_embeds=self.model.language_model.model.embed_tokens(suffix_ids),
                mask=attention,
                cache=batch_cache,
                position_ids=mx.array(position_array),
            )
            last_hidden = hidden[mx.arange(batch), mx.array([length - 1 for length in lengths])]
            logits = last_hidden.astype(mx.float32) @ output_rows.astype(mx.float32).T
            mx.eval(logits)
            selected[indexes] = np.asarray(logits).astype(np.float64)

        score_ms = (time.perf_counter() - score_started) * 1000
        inverse = np.argsort(order)
        selected = selected[inverse]
        if not np.isfinite(selected).all():
            raise BackendError("non-finite logits from the MLX forward pass")
        image_tokens = sum(tokens_per_image)
        text_tokens = sum(len(row) for row in ids) - image_tokens * len(ids)
        return _Readout(selected, image_tokens, text_tokens, prefix_ms, score_ms)

    def score_statements(
        self, images: list[LoadedImage], context: dict[str, Any] | None, statements: list[Statement]
    ) -> StatementScores:
        from scipy.special import logsumexp

        texts = [self.render_prompt(images, context, statement.text) for statement in statements]
        read = self._read(images, texts, self.yes_ids + self.no_ids)
        z_yes = logsumexp(read.selected[:, : len(self.yes_ids)], axis=1)
        z_no = logsumexp(read.selected[:, len(self.yes_ids) :], axis=1)
        return StatementScores(
            z=z_yes - z_no,
            z_yes=z_yes,
            z_no=z_no,
            off_mass=None,
            prompt_hashes=[prompts.prompt_hash(text) for text in texts],
            usage=BackendUsage(
                image_tokens=read.image_tokens,
                text_tokens=read.text_tokens,
                forward_passes=len(texts),
            ),
            timing_ms={"prefix": read.prefix_ms, "score": read.score_ms},
            cache_hit=False,
        )

    def score_labels(
        self,
        images: list[LoadedImage],
        context: dict[str, Any] | None,
        blocks: list[str],
        labels: list[str],
        assistant_prefix: str = "",
    ) -> LabelScores:
        from scipy.special import logsumexp

        texts = [self.render_prompt(images, context, block) + assistant_prefix for block in blocks]
        groups = [self._single_token_ids([label, " " + label]) for label in labels]
        flat = [token_id for group in groups for token_id in group]
        read = self._read(images, texts, flat)
        logits = np.zeros((len(texts), len(labels)), dtype=np.float64)
        cursor = 0
        for column, group in enumerate(groups):
            logits[:, column] = logsumexp(read.selected[:, cursor : cursor + len(group)], axis=1)
            cursor += len(group)
        return LabelScores(
            logits=logits,
            off_mass=None,
            prompt_hashes=[prompts.prompt_hash(text) for text in texts],
            usage=BackendUsage(
                image_tokens=read.image_tokens,
                text_tokens=read.text_tokens,
                forward_passes=len(texts),
            ),
            timing_ms={"prefix": read.prefix_ms, "score": read.score_ms},
            cache_hit=False,
        )

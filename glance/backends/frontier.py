"""Frontier baseline adapter. Eval only.

Calls a vision model through LiteLLM with a JSON schema that enumerates the allowed answers, temperature 0.
The model id comes from FRONTIER_MODEL and is logged exactly. This is the only code in glance that sends an image
off the machine, and the Engine only lets it run when the caller has opted in (`glance eval --confirm-spend`).

Frontier outputs are evaluation-only: the eval runner keeps whether a pick was right, never the pick itself, and
the call log redacts it, so nothing written to disk could serve as a training label.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any

from .. import prompts
from ..config import Config
from ..images import LoadedImage
from ..schema import BackendError, UnsupportedQuestionError
from .base import BackendUsage, PickItem, PickResult, Statement, StatementScores

JPEG_QUALITY = 90
MAX_OUTPUT_TOKENS = 4096  # room for models that think before answering; the JSON itself is a few dozen tokens
KEY_LIKE = re.compile(r"(sk|key|AIza)[-_A-Za-z0-9*.]{8,}")


def scrub(text: str, secret: str | None = None) -> str:
    """Remove the API key, and anything shaped like one, from text that is about to be logged or shown."""
    if secret:
        text = text.replace(secret, "<api-key>")
    return KEY_LIKE.sub("<api-key>", text)


class FrontierBackend:
    name = "frontier"
    kind = "frontier"
    revision = None
    device = "remote"
    dtype = "n/a"
    image_token_budget = None

    def __init__(self, cfg: Config, model_id: str | None = None, api_key: str | None = None):
        self.cfg = cfg
        self.model_id = (model_id or os.environ.get("FRONTIER_MODEL", "")).strip()
        if not self.model_id:
            raise BackendError("FRONTIER_MODEL is not set; the frontier baseline cannot run")
        # A key typed at the `glance baseline` prompt lives here, in memory, for the life of the process. It is never
        # written to disk or to a log. When it is None, LiteLLM reads the provider's usual environment variable.
        self._api_key = api_key
        self._send_temperature = True

    def __repr__(self) -> str:
        return f"FrontierBackend(model_id={self.model_id!r})"

    def score_statements(
        self, images: list[LoadedImage], context: dict[str, Any] | None, statements: list[Statement]
    ) -> StatementScores:
        raise UnsupportedQuestionError("the frontier baseline returns hard picks, not logits")

    @staticmethod
    def _data_url(img: LoadedImage) -> str:
        buf = io.BytesIO()
        img.image.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def answer_schema(items: list[PickItem]) -> dict[str, Any]:
        """One string field per question, each restricted to that question's allowed answers."""
        return {
            "type": "object",
            "properties": {f"q{i}": {"type": "string", "enum": list(item.allowed)} for i, item in enumerate(items)},
            "required": [f"q{i}" for i in range(len(items))],
            "additionalProperties": False,
        }

    def build_messages(self, images: list[LoadedImage], context: dict[str, Any] | None, items: list[PickItem]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for img in images:
            content.append({"type": "text", "text": prompts.image_label(img.id)})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(img)}})
        questions = "\n\n".join(f"q{i}:\n{item.prompt}" for i, item in enumerate(items))
        content.append({"type": "text", "text": prompts.context_block(context) + questions})
        return [{"role": "system", "content": prompts.FRONTIER_SYSTEM}, {"role": "user", "content": content}]

    def _complete(self, messages: list[dict[str, Any]], schema: dict[str, Any], warnings: list[str]):
        import litellm

        litellm.suppress_debug_info = True  # no "Give Feedback / Get Help" banner on provider errors
        kwargs: dict[str, Any] = {
            "model": self.model_id, "messages": messages, "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answers", "schema": schema, "strict": True}},
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._send_temperature:
            try:
                return litellm.completion(temperature=0, **kwargs)
            except litellm.BadRequestError as exc:
                if "temperature" not in str(exc).lower():
                    raise
                # Several current frontier models reject sampling parameters outright. Never a silent change:
                # the switch is reported in the response warnings, which the call log records too.
                self._send_temperature = False
        warnings.append(f"{self.model_id} rejects `temperature`; sent without it (provider default sampling)")
        return litellm.completion(**kwargs)

    def pick(self, images: list[LoadedImage], context: dict[str, Any] | None, items: list[PickItem]) -> PickResult:
        t0 = time.perf_counter()
        warnings: list[str] = []
        try:
            response = self._complete(self.build_messages(images, context, items), self.answer_schema(items), warnings)
        except Exception as exc:  # provider errors can echo the key back; scrub before anything is logged
            raise BackendError(
                f"frontier call failed: {type(exc).__name__}: {scrub(str(exc), self._api_key)}"
            ) from None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(text)
            picks = [str(parsed[f"q{i}"]) for i in range(len(items))]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise BackendError(f"frontier model did not return the requested JSON: {exc}") from exc
        usage = getattr(response, "usage", None)
        return PickResult(
            picks=picks,
            usage=BackendUsage(image_tokens=0, text_tokens=int(getattr(usage, "prompt_tokens", 0) or 0), forward_passes=1),
            timing_ms={"score": elapsed_ms},
            warnings=warnings,
        )

import base64
import hashlib
import io
import os

import numpy as np
import pytest
from PIL import Image

from glance.backends.base import BackendUsage, LabelScores, StatementScores
from glance.config import load_config

needs_models = pytest.mark.skipif(
    os.environ.get("GLANCE_TEST_MODELS") != "1", reason="set GLANCE_TEST_MODELS=1 to run tests that load model weights"
)


def _hash_logit(text: str) -> float:
    """Deterministic pseudo-logit in [-4, 4) from the text alone, so it cannot depend on batch order."""
    h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return (h / 2**32) * 8.0 - 4.0


class FakeBackend:
    """Type-agnostic backend with text-determined logits. kind='vlm' so every question type is supported."""

    name = "vlm"
    kind = "vlm"
    model_id = "fake/model"
    revision = "0" * 40
    device = "cpu"
    dtype = "float32"
    image_token_budget = 64

    def __init__(self, fixed: dict[str, float] | None = None):
        self.fixed = fixed or {}
        self.calls = 0

    def score_statements(self, images, context, statements):
        self.calls += 1
        z = np.array([self.fixed.get(s.candidate, _hash_logit(s.text)) for s in statements], dtype=np.float64)
        for img in images:
            img.image_tokens = 64
        return StatementScores(
            z=z, z_yes=z / 2, z_no=-z / 2, off_mass=np.full(len(z), 0.01),
            prompt_hashes=[hashlib.sha256(s.text.encode()).hexdigest()[:16] for s in statements],
            usage=BackendUsage(image_tokens=64 * len(images), text_tokens=10 * len(z), forward_passes=len(z)),
            timing_ms={"prefix": 1.0, "score": 2.0}, cache_hit=True,
        )

    def score_labels(self, images, context, prompts, labels, assistant_prefix=""):
        # Logit of a label depends on the text of the option printed next to it, not on its position.
        logits = np.zeros((len(prompts), len(labels)))
        for r, prompt in enumerate(prompts):
            option_lines = [ln for ln in prompt.splitlines() if len(ln) > 2 and ln[1:3] == ". "]
            for j, line in enumerate(option_lines):
                logits[r, j] = self.fixed.get(line[3:], _hash_logit(line[3:]))
        return LabelScores(
            logits=logits, off_mass=np.full(len(prompts), 0.02),
            prompt_hashes=[hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts],
            usage=BackendUsage(image_tokens=64, text_tokens=30 * len(prompts), forward_passes=len(prompts)),
            timing_ms={"prefix": 1.0, "score": 2.0}, cache_hit=True,
        )


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"paths": {"logs": str(tmp_path / "logs"), "runs": str(tmp_path / "runs"),
                                            "calibration": str(tmp_path / "calibration")}})


@pytest.fixture
def png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def request_body(png_b64):
    return {
        "model": "vlm",
        "state": {"images": [{"id": "img0", "base64": png_b64}], "context": {"source": "test"}},
        "questions": {
            "is_red": {"type": "noul", "instructions": "Is `img0` mostly red?"},
            "color": {"type": "choice", "instructions": "What color is `img0`?",
                      "criteria": {"red": "A red image", "blue": "A blue image", "other": None}},
            "brightness": {"type": "score", "instructions": "How bright is `img0`?",
                           "criteria": ["Very dark", "Medium", "Very bright"]},
        },
        # The v0 tests exercise the v0 `score` method; Glance elicitation (the default since v0.3) is in test_rating.py.
        "options": {"score_method": "statements"},
    }

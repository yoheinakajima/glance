"""The Python face of Glance: ask a frozen VLM typed questions about an image, get distributions back.

    from glance import Glance

    g = Glance()                                   # local Qwen3-VL, loaded on first use
    g.score("photo.jpg", "How blurry is `img0`?", ["Sharp", "Slightly soft", "Blurry", "Very blurry"])
    g.noul("photo.jpg", "Is there a dog in `img0`?")
    g.choice("photo.jpg", "What is in `img0`?", {"dog": None, "cat": None, "other": None})
    g.ask("photo.jpg", {"blur": {...}, "noise": {...}})   # many questions, one image prefill per view
    g.fit("How blurry is `img0`?", [...levels...], "my_labels/")   # a calibration for YOUR rubric from a few dozen labels

Glance is how you ask a frozen VLM for a score. It is not a VLM: nothing here trains or ships weights. Every method is a
thin wrapper over the same `Engine.decide()` the CLI and the server use, so the request and response shapes are the
ones in HANDOFF.md section 5 (plus the additive `score_method` and `calibrated: "auto"` options).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config, load_config

ImageInput = Any  # a path, a {"path"|"url"|"base64": ...} dict, or a list of those


def _image_refs(images: ImageInput) -> list[dict[str, Any]]:
    items = images if isinstance(images, (list, tuple)) else [images]
    refs = []
    for i, item in enumerate(items):
        ref = dict(item) if isinstance(item, dict) else {"path": str(item)}
        ref.setdefault("id", f"img{i}")
        refs.append(ref)
    return refs


class Glance:
    def __init__(self, model: str = "vlm", config: Config | str | Path | None = None, prefix_cache: bool = True,
                 calibrated: bool | str = "auto", score_method: str = "auto", model_id: str | None = None, revision: str | None = None,
                 image_longest_edge: int | None = None, dtype: str = "auto", backend: str = "torch"):
        """`prefix_cache=True` shares the image prefill between questions (about 3 to 4 times faster on requests with
        several questions; changes no decision in our checks, see docs/paper/RESULTS_LAB.md section 9). The CLI and the
        server keep the hand-off's default (off) unless started with --prefix-cache."""
        from .pipeline import Engine

        if isinstance(config, Config):
            cfg = config
        else:
            overrides: dict[str, Any] = {"vlm": {"prefix_cache": prefix_cache, "backend": backend}}
            if model_id:  # any Hugging Face image-text-to-text model instead of the built-in tiers
                overrides["models"] = {"generic": {"id": model_id, "revision": revision, "dtype": dtype, "image_longest_edge": image_longest_edge}}
            cfg = load_config(config, overrides)
        self.model = model
        self.options = {"calibrated": calibrated, "score_method": score_method}
        self.engine = Engine(cfg, source="python")

    def ask(self, images: ImageInput, questions: dict[str, dict[str, Any]], context: dict[str, Any] | None = None,
            **options: Any) -> dict[str, Any]:
        """The full response (section 5 shape) for several questions about the same image(s). Raises GlanceError."""
        body = {"model": self.model, "state": {"images": _image_refs(images), "context": context},
                "questions": questions, "options": {**self.options, **options}}
        return self.engine.decide(body).response.model_dump()

    def _one(self, images: ImageInput, question: dict[str, Any], **options: Any) -> dict[str, Any]:
        response = self.ask(images, {"q": question}, **options)
        return {**response["answers"]["q"], "warnings": response["warnings"]}

    def score(self, images: ImageInput, instructions: str, criteria: list[str], **options: Any) -> dict[str, Any]:
        """Rate on an ordered rubric (lowest level first). `score` is the expected level; `probabilities` the distribution."""
        return self._one(images, {"type": "score", "instructions": instructions, "criteria": list(criteria)}, **options)

    def noul(self, images: ImageInput, instructions: str, **options: Any) -> dict[str, Any]:
        return self._one(images, {"type": "noul", "instructions": instructions}, **options)

    def choice(self, images: ImageInput, instructions: str, criteria: dict[str, str | None] | list[str], **options: Any) -> dict[str, Any]:
        options_map = criteria if isinstance(criteria, dict) else {key: None for key in criteria}
        return self._one(images, {"type": "choice", "instructions": instructions, "criteria": options_map}, **options)

    def fit(self, instructions: str, criteria: list[str], labels: Any, method: str = "ens4d", name: str | None = None,
            save: bool = True, unlabeled: bool = False):
        """Fit a calibration for this rubric. `labels` is a folder-per-level directory, a JSONL/CSV file, or a list of
        (image path, level index) pairs. Returns the calibration; it is saved under calibration/ratings/ and picked up
        by later calls with the same instructions and criteria."""
        from . import fit as fit_module

        if unlabeled:  # any folder or list of images from your domain, no levels needed
            examples = fit_module.read_unlabeled(labels) if isinstance(labels, (str, Path)) else [(str(i), -1) for i in labels]
        else:
            examples = fit_module.read_labels(labels) if isinstance(labels, (str, Path)) else list(labels)
        cal, _ = fit_module.fit_rubric(self.engine, instructions, list(criteria), examples, method=method,
                                       model=self.model, name=name, save=save, unlabeled=unlabeled)
        return cal

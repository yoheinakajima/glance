"""Versioned prompt templates (HANDOFF section 6). Any edit to a template bumps PROMPT_VERSION.

These are plain text renderers. They take strings, not question objects: only `scorer` knows about question types.
"""

from __future__ import annotations

import hashlib
import json
import string
from typing import Any

PROMPT_VERSION = "p1"

# Shared prefix, in user-content order: image(s), context JSON, then the statement block.
IMAGE_LABEL = "Image `{id}`:"
CONTEXT_BLOCK = "Context: {context_json}\n\n"

NOUL_TEMPLATE = "Question: {instructions}\n{criteria}Answer Yes or No."
NOUL_TRUE_LINE = "Yes means: {text}\n"
NOUL_FALSE_LINE = "No means: {text}\n"

# Used for choice options and score levels. Level numbers and neighbours are never shown.
CANDIDATE_TEMPLATE = (
    "Question: {instructions}\n"
    "Candidate answer: {candidate}\n"
    "Is this candidate the correct answer? Answer Yes or No."
)

# Comparison method `letter`: all options in one prompt, logits read over the label tokens.
LETTER_TEMPLATE = "Question: {instructions}\nOptions:\n{options}\nAnswer with the letter of the correct option."
LETTER_OPTION_LINE = "{label}. {candidate}"
LETTER_LABELS = list(string.ascii_uppercase)  # capped at 26 options

YES_VARIANTS = ["Yes", " Yes", "yes", " yes"]
NO_VARIANTS = ["No", " No", "no", " no"]

DUAL_ENCODER_PHOTO_PREFIX = "a photo of {text}"

# Frontier baseline (eval only): the model returns one enumerated answer per question via a JSON schema.
FRONTIER_SYSTEM = (
    "You answer questions about images. Reply only with JSON that matches the schema. "
    "Each field must be exactly one of its allowed answers."
)
FRONTIER_ENUM_TEMPLATE = "Question: {instructions}\nAllowed answers:\n{answers}"
FRONTIER_ENUM_LINE = "- {key}: {text}"


def image_label(image_id: str) -> str:
    return IMAGE_LABEL.format(id=image_id)


def context_block(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    return CONTEXT_BLOCK.format(context_json=json.dumps(context, sort_keys=True, ensure_ascii=False))


def candidate_text(key: str, description: str | None) -> str:
    """The option description, or the key with underscores as spaces when the description is null."""
    return description if description else key.replace("_", " ")


def render_noul(instructions: str, true_text: str | None = None, false_text: str | None = None) -> str:
    criteria = ""
    if true_text:
        criteria += NOUL_TRUE_LINE.format(text=true_text)
    if false_text:
        criteria += NOUL_FALSE_LINE.format(text=false_text)
    return NOUL_TEMPLATE.format(instructions=instructions, criteria=criteria)


def render_candidate(instructions: str, candidate: str) -> str:
    return CANDIDATE_TEMPLATE.format(instructions=instructions, candidate=candidate)


def render_letter(instructions: str, candidates: list[str]) -> str:
    if len(candidates) > len(LETTER_LABELS):
        raise ValueError(f"letter method is capped at {len(LETTER_LABELS)} options")
    lines = [LETTER_OPTION_LINE.format(label=LETTER_LABELS[i], candidate=c) for i, c in enumerate(candidates)]
    return LETTER_TEMPLATE.format(instructions=instructions, options="\n".join(lines))


def render_frontier_enum(instructions: str, answers: list[tuple[str, str]]) -> str:
    lines = [FRONTIER_ENUM_LINE.format(key=key, text=text) for key, text in answers]
    return FRONTIER_ENUM_TEMPLATE.format(instructions=instructions, answers="\n".join(lines))


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

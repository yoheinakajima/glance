"""Turns raw logits into noul / choice / score answers with confidence (HANDOFF section 6).

This is the only module that knows about question types. It builds statements from questions, asks the backend
for raw logits, and assembles answers. Calibration happens downstream and hands calibrated probabilities back to
`build_answer`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit

from . import prompts, rating
from .backends.base import Backend, BackendUsage, PickItem, Statement
from .images import LoadedImage
from .schema import (
    Answer,
    BackendError,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    UnsupportedQuestionError,
)

ROUND = 6

# --- math on fixed logits -----------------------------------------------------------------------


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    out = expit(np.asarray(z, dtype=np.float64))
    return float(out) if out.ndim == 0 else out


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(z, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def confidence(p: np.ndarray) -> float:
    """1 - H(p) / ln(K): 1 when all mass is on one answer, 0 when uniform."""
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return float(min(1.0, max(0.0, 1.0 - entropy / math.log(len(p)))))


def margin(p: np.ndarray) -> float:
    top2 = np.sort(np.asarray(p, dtype=np.float64))[-2:]
    return float(top2[1] - top2[0])


def expected_score(p: np.ndarray) -> float:
    return float((np.arange(len(p)) * np.asarray(p, dtype=np.float64)).sum())


def noul_confidence(p: float) -> float:
    """Ranking signal for selective accuracy on noul: 2 * |p - 0.5|."""
    return 2.0 * abs(p - 0.5)


# --- scoring ------------------------------------------------------------------------------------


@dataclass
class QuestionScore:
    qid: str
    qtype: str
    keys: list[str]  # noul: ["true"]; choice: option keys; score: "0".."K-1"
    method: str  # "statement" | "independent" | "letter" | "pick" | "digits" | "ens4d"
    z: np.ndarray | None = None  # noul: [1]; choice and score: [K]; None for a frontier pick
    pick: str | None = None  # frontier only
    statements: list[dict[str, Any]] = field(default_factory=list)  # per-statement log records
    features: np.ndarray | None = None  # rating methods: the members' level logits, concatenated, for the rubric calibration


@dataclass
class ScoringResult:
    scores: dict[str, QuestionScore]
    usage: BackendUsage
    timing_ms: dict[str, float]
    warnings: list[str]
    cache_hit: bool | None = None


def _candidates(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return [prompts.candidate_text(key, desc) for key, desc in question.criteria.items()]
    if isinstance(question, ScoreQuestion):
        return list(question.criteria)
    raise TypeError(type(question).__name__)


def _keys(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria.keys())
    if isinstance(question, ScoreQuestion):
        return [str(i) for i in range(len(question.criteria))]
    return ["true"]


def _referenced_image(question: Question, images: list[LoadedImage]) -> LoadedImage:
    """A dual encoder scores one image. Pick the one image the instructions name by backticked id."""
    ids = {img.id: img for img in images}
    named = [m for m in dict.fromkeys(re.findall(r"`([^`]+)`", question.instructions)) if m in ids]
    if len(named) == 1:
        return ids[named[0]]
    if not named and len(images) == 1:
        return images[0]
    raise UnsupportedQuestionError(
        "the dual encoder needs a question about exactly one image; "
        f"the instructions name {named or 'no image'} and the request has {len(images)} images",
        {"named_images": named},
    )


def letter_shifts(n_options: int, max_rotations: int) -> list[int]:
    """Up to `max_rotations` cyclic rotations of the option order, spread evenly around the cycle."""
    rotations = max(1, min(max_rotations, n_options))
    return list(dict.fromkeys(round(r * n_options / rotations) % n_options for r in range(rotations)))


def _statement_records(scores, lo: int, hi: int) -> list[dict[str, Any]]:
    records = []
    for i in range(lo, hi):
        records.append(
            {
                "prompt_hash": scores.prompt_hashes[i],
                "z_yes": float(scores.z_yes[i]),
                "z_no": None if scores.z_no is None else float(scores.z_no[i]),
                "z": float(scores.z[i]),
                "off_mass": None if scores.off_mass is None else float(scores.off_mass[i]),
            }
        )
    return records


def score_questions(
    backend: Backend,
    images: list[LoadedImage],
    context: dict[str, Any] | None,
    questions: dict[str, Question],
    choice_method: str = "independent",
    letter_rotations: int = 4,
    off_mass_warn: float = 0.1,
    score_method: str | dict[str, str] = "auto",
) -> ScoringResult:
    if backend.kind == "frontier":
        return _score_frontier(backend, images, context, questions)

    dual = backend.kind == "dual_encoder"
    # `score_method` is one method for every rating question, or {question id: method} (the engine resolves "auto" per rubric:
    # a fitted rubric keeps the method it was fit with, an unfitted one gets the zero-shot read)
    per_question = dict(score_method) if isinstance(score_method, dict) else {}
    explicit = score_method if isinstance(score_method, str) and score_method != "auto" else None
    default_method = explicit or ("statements" if dual else rating.ZERO_SHOT_METHOD)
    method_of = lambda qid: per_question.get(qid, default_method)  # noqa: E731
    usage = BackendUsage()
    timing: dict[str, float] = {}
    warnings: list[str] = []
    cache_hits: list[bool] = []
    scores: dict[str, QuestionScore] = {}

    # 1. Build every Yes/No statement in the request so one backend call can share the image prefix.
    plan: list[tuple[str, str, list[Statement], LoadedImage | None]] = []  # qid, method, statements, dual image
    letter_qids: list[str] = []
    rating_qids: list[str] = []
    for qid, q in questions.items():
        image = _referenced_image(q, images) if dual else None
        if isinstance(q, NoulQuestion):
            true_text = q.criteria.true if q.criteria else None
            false_text = q.criteria.false if q.criteria else None
            if dual:
                if not true_text:
                    raise UnsupportedQuestionError(
                        f"question `{qid}`: the dual encoder needs criteria.true to answer a noul question",
                        {"question": qid},
                    )
                texts = [true_text] + ([false_text] if false_text else [])
                statements = [Statement(text=t, candidate=t) for t in texts]
            else:
                text = prompts.render_noul(q.instructions, true_text, false_text)
                statements = [Statement(text=text, candidate=true_text or q.instructions)]
            plan.append((qid, "statement", statements, image))
        elif isinstance(q, ChoiceQuestion) and choice_method == "letter":
            if dual:
                raise UnsupportedQuestionError(
                    f"question `{qid}`: choice_method `letter` is VLM-only", {"question": qid}
                )
            if len(q.criteria) > len(prompts.LETTER_LABELS):
                raise UnsupportedQuestionError(
                    f"question `{qid}`: choice_method `letter` is capped at {len(prompts.LETTER_LABELS)} options "
                    f"(got {len(q.criteria)})",
                    {"question": qid},
                )
            letter_qids.append(qid)
        elif isinstance(q, ScoreQuestion) and method_of(qid) != "statements":
            if dual:
                raise UnsupportedQuestionError(
                    f"question `{qid}`: score_method `{method_of(qid)}` is VLM-only; use `statements`", {"question": qid}
                )
            rating_qids.append(qid)
        else:
            statements = [
                Statement(text=prompts.render_candidate(q.instructions, c), candidate=c) for c in _candidates(q)
            ]
            method = "independent" if isinstance(q, ChoiceQuestion) else "statement"
            plan.append((qid, method, statements, image))

    # 2. Score them. The VLM takes everything in one call; the dual encoder goes image by image.
    groups: dict[str, list[int]] = {}
    for idx, (_, _, _, image) in enumerate(plan):
        groups.setdefault(image.id if image else "*", []).append(idx)
    for group in groups.values():
        flat = [s for idx in group for s in plan[idx][2]]
        group_images = [plan[group[0]][3]] if dual else images
        result = backend.score_statements(group_images, context, flat)
        usage.add(result.usage)
        for key, ms in result.timing_ms.items():
            timing[key] = timing.get(key, 0.0) + ms
        if result.cache_hit is not None:
            cache_hits.append(result.cache_hit)
        cursor = 0
        for idx in group:
            qid, method, statements, _ = plan[idx]
            q = questions[qid]
            lo, hi = cursor, cursor + len(statements)
            cursor = hi
            z = np.asarray(result.z[lo:hi], dtype=np.float64)
            if isinstance(q, NoulQuestion):
                z = np.array([z[0] - z[1]]) if len(z) == 2 else z[:1]  # dual encoder: z_true - z_false
            scores[qid] = QuestionScore(
                qid=qid, qtype=q.type, keys=_keys(q), method=method, z=z,
                statements=_statement_records(result, lo, hi),
            )

    # 3. Letter method: all options in one prompt, logits over label tokens, averaged over cyclic rotations.
    for qid in letter_qids:
        q = questions[qid]
        candidates = _candidates(q)
        k = len(candidates)
        shifts = letter_shifts(k, letter_rotations)
        rendered = [prompts.render_letter(q.instructions, candidates[s:] + candidates[:s]) for s in shifts]
        result = backend.score_labels(images, context, rendered, prompts.LETTER_LABELS[:k])  # type: ignore[attr-defined]
        usage.add(result.usage)
        for key, ms in result.timing_ms.items():
            timing[key] = timing.get(key, 0.0) + ms
        if result.cache_hit is not None:
            cache_hits.append(result.cache_hit)
        per_option = np.stack(
            [[result.logits[r, (i - s) % k] for i in range(k)] for r, s in enumerate(shifts)]
        )  # label j of rotation s is option (s + j) % k
        scores[qid] = QuestionScore(
            qid=qid, qtype=q.type, keys=_keys(q), method="letter", z=per_option.mean(axis=0),
            statements=[
                {"prompt_hash": result.prompt_hashes[r], "shift": s, "label_logits": per_option[r].tolist(),
                 "off_mass": float(result.off_mass[r])}
                for r, s in enumerate(shifts)
            ],
        )

    # 4. Rating methods: digit readouts, packed so that every question shares the image prefill(s).
    for method in dict.fromkeys(method_of(qid) for qid in rating_qids):
        _score_ratings(backend, images, context, {qid: questions[qid] for qid in rating_qids if method_of(qid) == method}, method,
                       scores, usage, timing, cache_hits)

    for qid, qs in scores.items():
        worst = max((s["off_mass"] for s in qs.statements if s.get("off_mass") is not None), default=0.0)
        if worst > off_mass_warn:
            warnings.append(f"question `{qid}`: off_mass {worst:.3f} exceeds {off_mass_warn}")

    ordered = {qid: scores[qid] for qid in questions}
    return ScoringResult(
        scores=ordered, usage=usage, timing_ms=timing, warnings=warnings,
        cache_hit=all(cache_hits) if cache_hits else None,
    )


def _subject_id(question: Question, images: list[LoadedImage]) -> str | None:
    """The id of the one image a rating question is about (named by backticked id, or the only image), else None."""
    ids = {img.id for img in images}
    named = [m for m in dict.fromkeys(re.findall(r"`([^`]+)`", question.instructions)) if m in ids]
    if len(named) == 1:
        return named[0]
    return images[0].id if not named and len(images) == 1 else None


def _rated_image(qid: str, question: Question, images: list[LoadedImage]) -> LoadedImage:
    """The one image a rating question is about: named by backticked id, or the only image in the request."""
    ids = {img.id: img for img in images}
    named = [m for m in dict.fromkeys(re.findall(r"`([^`]+)`", question.instructions)) if m in ids]
    if len(named) == 1:
        return ids[named[0]]
    if not named and len(images) == 1:
        return images[0]
    raise UnsupportedQuestionError(
        f"question `{qid}`: score_method `ens4d` magnifies the one image the question rates, but the instructions name "
        f"{named or 'no image'} and the request has {len(images)} images; name exactly one image by backticked id, or "
        "use score_method `digits` or `statements`",
        {"question": qid, "named_images": named},
    )


def _zoom_image(target: LoadedImage) -> LoadedImage:
    import hashlib

    crop = rating.zoom_crop(target.image)
    return LoadedImage(id=rating.ZOOM_IMAGE_ID, image=crop, sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                       width=crop.width, height=crop.height, format="PNG", source=f"derived:zoom:{target.id}")


def _score_ratings(backend, images, context, questions: dict[str, ScoreQuestion], score_method: str,
                   scores: dict[str, QuestionScore], usage: BackendUsage, timing: dict[str, float],
                   cache_hits: list[bool]) -> None:
    """Glance elicitation (`glance/rating.py`). One `score_labels` call per (view, number of levels): the full view is
    the request's images; the zoom view adds a magnified crop of the rated image. Every question of a call sits behind
    the same image prefix, so with the prefix cache a request pays at most one prefill per view."""
    members = rating.MEMBERS[score_method]
    needs_zoom = any(m.startswith("zoom_") for m in members)
    if needs_zoom and any(img.id == rating.ZOOM_IMAGE_ID for img in images):
        raise UnsupportedQuestionError(f"image id `{rating.ZOOM_IMAGE_ID}` is reserved by score_method `{score_method}`")

    # (view key, K, forced start of the assistant turn) -> [(qid, member, reversed?, block)], with the image list of each view
    calls: dict[tuple[str, int, str], list[tuple[str, str, bool, str]]] = {}
    views: dict[str, list[LoadedImage]] = {"": images}
    labels_for: dict[int, list[str]] = {}
    for qid, q in questions.items():
        levels = list(q.criteria)
        target = _rated_image(qid, q, images) if needs_zoom else None
        for member in members:
            view = ""
            if member.startswith("zoom_"):
                view = target.id
                if view not in views:
                    views[view] = images + [_zoom_image(target)]
            image_id = target.id if target else _subject_id(q, images)
            block, labels, reverse = rating.member_prompt(member, q.instructions, levels, image_id if member == "jsondigits" else (image_id or "img0"))
            labels_for[len(levels)] = labels
            calls.setdefault((view, len(levels), rating.member_assistant_prefix(member)), []).append((qid, member, reverse, block))

    member_logits: dict[str, dict[str, np.ndarray]] = {qid: {} for qid in questions}
    records: dict[str, list[dict[str, Any]]] = {qid: [] for qid in questions}
    for (view, k, prefix), entries in calls.items():
        extra = {"assistant_prefix": prefix} if prefix else {}  # only passed when used, so backends without the parameter keep working
        result = backend.score_labels(views[view], context, [block for *_, block in entries], labels_for[k], **extra)
        usage.add(result.usage)
        for key, ms in result.timing_ms.items():
            timing[key] = timing.get(key, 0.0) + ms
        if result.cache_hit is not None:
            cache_hits.append(result.cache_hit)
        for row, (qid, member, reverse, _) in enumerate(entries):
            logits = np.asarray(result.logits[row], dtype=np.float64)
            member_logits[qid][member] = logits[::-1] if reverse else logits
            records[qid].append({"member": member, "prompt_hash": result.prompt_hashes[row],
                                 "label_logits": member_logits[qid][member].tolist(), "off_mass": float(result.off_mass[row])})
    for qid, q in questions.items():
        ordered = [member_logits[qid][m] for m in members]
        scores[qid] = QuestionScore(
            qid=qid, qtype=q.type, keys=_keys(q), method=score_method, z=rating.raw_level_logits(ordered),
            features=np.concatenate(ordered), statements=sorted(records[qid], key=lambda r: members.index(r["member"])),
        )


def _score_frontier(backend, images, context, questions: dict[str, Question]) -> ScoringResult:
    items: list[PickItem] = []
    for q in questions.values():
        if isinstance(q, NoulQuestion):
            true_text = q.criteria.true if q.criteria else None
            false_text = q.criteria.false if q.criteria else None
            items.append(PickItem(prompt=prompts.render_noul(q.instructions, true_text, false_text), allowed=["Yes", "No"]))
        elif isinstance(q, ChoiceQuestion):
            answers = [(key, prompts.candidate_text(key, desc)) for key, desc in q.criteria.items()]
            items.append(PickItem(prompt=prompts.render_frontier_enum(q.instructions, answers), allowed=_keys(q)))
        else:
            answers = [(str(i), text) for i, text in enumerate(q.criteria)]
            items.append(PickItem(prompt=prompts.render_frontier_enum(q.instructions, answers), allowed=_keys(q)))
    result = backend.pick(images, context, items)
    scores = {}
    for (qid, q), item, pick in zip(questions.items(), items, result.picks):
        if pick not in item.allowed:
            raise BackendError(f"question `{qid}`: frontier model returned {pick!r}, not one of the allowed answers")
        scores[qid] = QuestionScore(
            qid=qid, qtype=q.type, keys=_keys(q), method="pick", pick=pick,
            statements=[{"prompt_hash": prompts.prompt_hash(item.prompt)}],
        )
    return ScoringResult(scores=scores, usage=result.usage, timing_ms=result.timing_ms,
                         warnings=list(result.warnings), cache_hit=None)


# --- answers ------------------------------------------------------------------------------------


def raw_probabilities(qs: QuestionScore) -> float | np.ndarray | None:
    """Uncalibrated probabilities: sigmoid(z) for noul, softmax(z) with T = 1 otherwise."""
    if qs.z is None:
        return None
    if qs.qtype == "noul":
        return float(sigmoid(qs.z[0]))
    return softmax(qs.z)


def _as_dict(keys: list[str], p: np.ndarray) -> dict[str, float]:
    return {k: round(float(v), ROUND) for k, v in zip(keys, p)}


def build_answer(
    question: Question,
    qs: QuestionScore,
    raw: float | np.ndarray | None,
    calibrated: float | np.ndarray | None = None,
    calibration_version: str | None = None,
) -> Answer:
    """Assemble the typed answer. `calibrated` is None when calibration is off; the answer then uses `raw`."""
    final = raw if calibrated is None else calibrated

    if isinstance(question, NoulQuestion):
        if qs.pick is not None:
            return NoulAnswer(noul=1.0 if qs.pick == "Yes" else 0.0, raw=None)
        return NoulAnswer(noul=round(float(final), ROUND), raw=round(float(raw), ROUND))

    if isinstance(question, ChoiceQuestion):
        if qs.pick is not None:
            return ChoiceAnswer(choice=qs.pick, probabilities=None, confidence=None, margin=None, raw=None)
        return ChoiceAnswer(
            choice=qs.keys[int(np.argmax(final))],
            probabilities=_as_dict(qs.keys, final),
            confidence=round(confidence(final), ROUND),
            margin=round(margin(final), ROUND),
            raw=_as_dict(qs.keys, raw),
        )

    legend = {str(i): text for i, text in enumerate(question.criteria)}
    if qs.pick is not None:
        return ScoreAnswer(
            score=float(int(qs.pick)), legend=legend, probabilities=None, confidence=None, margin=None, raw=None
        )
    return ScoreAnswer(
        score=round(expected_score(final), ROUND),
        legend=legend,
        probabilities=_as_dict(qs.keys, final),
        confidence=round(confidence(final), ROUND),
        margin=round(margin(final), ROUND),
        raw=_as_dict(qs.keys, raw),
        method="statements" if qs.method == "statement" else qs.method,
        calibration=calibration_version if calibrated is not None else None,
    )

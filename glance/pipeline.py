"""The one decide() path shared by the server, the CLI and the eval runner.

Request JSON -> schema -> images -> backend (raw logits) -> scorer -> calibration -> Response JSON + call log.
"""

from __future__ import annotations

import threading
from typing import Any

from . import __version__, calibration, rating
from .backends import Backend, load_backend, model_string
from .config import Config
from .images import load_images
from .logging_utils import Timer, call_log_writer, error_record, git_sha, new_request_id, utc_now
from .prompts import PROMPT_VERSION
from .schema import (
    BackendError,
    CalibrationMismatchError,
    DecideRequest,
    DecideResponse,
    ErrorResponse,
    GlanceError,
    UnsupportedQuestionError,
    Usage,
    parse_request,
)
from .scorer import ScoringResult, build_answer, raw_probabilities, score_questions


FRONTIER_REDACTED = "<redacted: frontier outputs are evaluation-only>"


class DecisionTrace:
    """What the eval runner needs beyond the public response: logits and statement records per question."""

    def __init__(self, response: DecideResponse, scoring: ScoringResult):
        self.response = response
        self.scoring = scoring


class Engine:
    def __init__(self, cfg: Config, source: str = "cli", backends: dict[str, Backend] | None = None,
                 allow_frontier: bool = False):
        self.cfg = cfg
        self.source = source
        # The frontier baseline sends images off the machine and costs money. It is eval-only: the eval runner
        # turns it on after --confirm-spend; the server never does.
        self.allow_frontier = allow_frontier
        self._backends: dict[str, Backend] = dict(backends or {})
        self._calibration: dict[str, calibration.CalibrationParams] = {}
        self._lock = threading.Lock()  # one forward pass at a time; the models are not re-entrant

    # --- backends ---------------------------------------------------------------------------------

    def backend(self, name: str) -> Backend:
        if name not in self._backends:
            self._backends[name] = load_backend(name, self.cfg)
        return self._backends[name]

    def loaded_backends(self) -> dict[str, Backend]:
        return dict(self._backends)

    def calibration_key(self, backend: Backend, choice_method: str) -> calibration.CalibrationKey:
        return calibration.CalibrationKey(
            backend=backend.name,
            model=f"{backend.model_id}@{backend.revision}" if backend.revision else backend.model_id,
            prompt_version=PROMPT_VERSION,
            choice_method=choice_method,
            image_token_budget=backend.image_token_budget,
        )

    def _calibration_params(self, key: calibration.CalibrationKey) -> calibration.CalibrationParams:
        cache_key = key.hash()
        if cache_key not in self._calibration:
            self._calibration[cache_key] = calibration.load_params(self.cfg.path("calibration"), key)
        return self._calibration[cache_key]

    def rating_key(self, backend: Backend, method: str, question) -> rating.RatingKey:
        return rating.RatingKey(
            backend=backend.name,
            model=f"{backend.model_id}@{backend.revision}" if backend.revision else backend.model_id,
            prompt_version=rating.RATING_PROMPT_VERSION, score_method=method,
            image_token_budget=backend.image_token_budget,
            instructions=question.instructions, criteria=tuple(question.criteria),
        )

    def rating_calibration(self, key: rating.RatingKey) -> rating.RatingCalibration | None:
        """User fits (calibration/ratings/) shadow the calibrations shipped with the package."""
        return rating.load_calibration([self.cfg.path("calibration") / "ratings", rating.ASSETS_DIR], key)

    # --- decide -----------------------------------------------------------------------------------

    def decide(self, body: Any, source: str | None = None, request_id: str | None = None) -> DecisionTrace:
        """Run one request. Raises GlanceError on failure. Every call, good or bad, writes one call-log line."""
        request_id = request_id or new_request_id()
        timer = Timer()
        log: dict[str, Any] = {
            "request_id": request_id,
            "ts": utc_now(),
            "source": source or self.source,
            "harness_version": __version__,
            "git_sha": git_sha(),
            "prompt_version": PROMPT_VERSION,
            "error": None,
        }
        try:
            trace = self._decide(body, request_id, timer, log)
        except GlanceError as exc:
            self._log_error(log, timer, exc.code, exc)
            exc.request_id = request_id  # type: ignore[attr-defined]
            raise
        except Exception as exc:  # model or library failure: surface as backend_error, with the traceback logged
            self._log_error(log, timer, BackendError.code, exc)
            wrapped = BackendError(f"{type(exc).__name__}: {exc}", {"exception_type": type(exc).__name__})
            wrapped.request_id = request_id  # type: ignore[attr-defined]
            raise wrapped from exc
        return trace

    def _log_error(self, log: dict[str, Any], timer: Timer, code: str, exc: BaseException) -> None:
        log["error"] = error_record(code, exc)
        log["timing_ms"] = timer.result()
        call_log_writer(self.cfg.path("logs")).write(log)

    def _decide(self, body: Any, request_id: str, timer: Timer, log: dict[str, Any]) -> DecisionTrace:
        request = body if isinstance(body, DecideRequest) else parse_request(body)
        log["questions"] = {qid: q.model_dump() for qid, q in request.questions.items()}
        log["context"] = request.state.context
        log["choice_method"] = request.options.choice_method
        log["score_method"] = request.options.score_method
        if request.model == "frontier" and not self.allow_frontier:
            raise UnsupportedQuestionError(
                "the frontier baseline is eval-only: it sends images off this machine and spends money, so it runs "
                "only from `glance eval --confirm-spend` (or `glance decide --confirm-spend`), never from the server"
            )

        with timer.span("load"):
            images = load_images(request.state.images, self.cfg.limits)
            backend = self.backend(request.model)
        log.update(
            backend=backend.name, model_id=backend.model_id, model_revision=backend.revision,
            device=backend.device, dtype=backend.dtype, image_token_budget=backend.image_token_budget,
        )

        with self._lock:
            scoring = score_questions(
                backend, images, request.state.context, request.questions,
                choice_method=request.options.choice_method,
                letter_rotations=self.cfg.vlm.letter_rotations,
                off_mass_warn=self.cfg.vlm.off_mass_warn,
                score_method=request.options.score_method,
            )
        timer.add("prefix", scoring.timing_ms.get("prefix", 0.0))
        timer.add("score", scoring.timing_ms.get("score", 0.0))
        log["images"] = [img.log_record() for img in images]

        warnings = list(scoring.warnings)
        wanted = request.options.calibrated  # False | True | "auto"
        params = None
        needs_v0 = any(qs.method not in rating.MEMBERS for qs in scoring.scores.values())
        if wanted and backend.kind == "frontier":
            warnings.append("calibrated: true has no effect on frontier picks, which carry no probabilities")
        elif wanted and needs_v0:
            try:
                params = self._calibration_params(self.calibration_key(backend, request.options.choice_method))
            except CalibrationMismatchError:
                if wanted is True:
                    raise
                warnings.append("no fitted calibration matches the active configuration; noul, choice and "
                                "statement-scored answers are uncalibrated (run `glance calibrate`)")

        answers = {}
        question_logs = {}
        versions: list[str] = [params.version] if params is not None else []
        with timer.span("calibrate"):
            for qid, question in request.questions.items():
                qs = scoring.scores[qid]
                raw = raw_probabilities(qs)
                cal, cal_version = None, None
                if qs.method in rating.MEMBERS:
                    # Glance elicitation: the calibration belongs to this rubric; there is no pooled fallback.
                    fitted = self.rating_calibration(self.rating_key(backend, qs.method, question)) if wanted else None
                    if fitted is not None:
                        cal, cal_version = rating.apply_matrix(fitted, qs.features), fitted.version
                        versions.append(fitted.version)
                    elif wanted is True:
                        raise CalibrationMismatchError(
                            f"calibrated: true, but question `{qid}` has no calibration for this rubric and configuration; "
                            "fit one from a few dozen labeled images with `glance fit`, or send calibrated: \"auto\"",
                            {"question": qid, "rating_key": self.rating_key(backend, qs.method, question).model_dump()},
                        )
                    elif wanted == "auto":
                        warnings.append(f"question `{qid}`: no calibration for this rubric; probabilities are uncalibrated "
                                        "(fit one from a few dozen labeled images with `glance fit`)")
                elif params is not None:
                    cal = calibration.apply(qs, params)
                answers[qid] = build_answer(question, qs, raw, cal, cal_version)
                # Frontier outputs are evaluation-only and must never land in a file that could serve as a
                # training label, so the call log records that a pick was made but not what it was.
                logged_answer = FRONTIER_REDACTED if backend.kind == "frontier" else answers[qid].model_dump()
                question_logs[qid] = {
                    "type": qs.qtype, "method": qs.method, "keys": qs.keys,
                    "z": None if qs.z is None else qs.z.tolist(),
                    "features": None if qs.features is None else qs.features.tolist(),
                    "raw_probabilities": raw, "calibrated_probabilities": cal,
                    "answer": logged_answer, "statements": qs.statements,
                }

        response = DecideResponse(
            request_id=request_id,
            model=model_string(backend),
            prompt_version=PROMPT_VERSION,
            calibration_version="+".join(dict.fromkeys(versions)) or None,
            answers=answers,
            usage=Usage(
                image_tokens=scoring.usage.image_tokens, text_tokens=scoring.usage.text_tokens,
                forward_passes=scoring.usage.forward_passes, output_tokens=scoring.usage.output_tokens,
                cost_usd=scoring.usage.cost_usd,
            ),
            timing_ms=timer.result(),
            warnings=warnings,
        )
        log.update(
            calibration_version=response.calibration_version, output=question_logs, warnings=warnings,
            timing_ms=response.timing_ms, forward_passes=scoring.usage.forward_passes,
            usage=response.usage.model_dump(), cache=_cache_label(scoring.cache_hit),
        )
        call_log_writer(self.cfg.path("logs")).write(log)
        return DecisionTrace(response, scoring)

    def decide_json(self, body: Any, source: str | None = None) -> tuple[int, dict[str, Any]]:
        """HTTP-shaped wrapper: (status, JSON payload). Errors use the section 5 error shape."""
        try:
            trace = self.decide(body, source=source)
        except GlanceError as exc:
            payload = ErrorResponse(
                request_id=getattr(exc, "request_id", "") or new_request_id(),
                code=exc.code, message=exc.message, detail=exc.detail,
            )
            return exc.http_status, payload.model_dump()
        return 200, trace.response.model_dump()


def _cache_label(cache_hit: bool | None) -> str | None:
    if cache_hit is None:
        return None
    return "hit" if cache_hit else "miss"

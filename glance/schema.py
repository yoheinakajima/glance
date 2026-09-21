"""Typed request, question, answer and error models (HANDOFF section 5). Leftmost module: imports nothing from glance."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MAX_IMAGES = 4
MAX_OPTIONS = 128
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

QUESTION_TYPES = ("noul", "choice", "score")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --- request ------------------------------------------------------------------------------------


class ImageRef(_Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    path: str | None = None
    url: str | None = None
    base64: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ImageRef":
        given = [name for name in ("path", "url", "base64") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(f"give exactly one of path, url, base64 (got {given or 'none'})")
        return self


class State(_Strict):
    images: list[ImageRef] = Field(min_length=1, max_length=MAX_IMAGES)
    context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> "State":
        ids = [img.id for img in self.images]
        if len(set(ids)) != len(ids):
            raise ValueError("image ids must be unique")
        return self


class NoulCriteria(_Strict):
    true: str | None = None
    false: str | None = None


class NoulQuestion(_Strict):
    type: Literal["noul"]
    instructions: str = Field(min_length=1)
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2, max_length=MAX_OPTIONS)

    @model_validator(mode="after")
    def _non_empty_keys(self) -> "ChoiceQuestion":
        if any(not key.strip() for key in self.criteria):
            raise ValueError("option keys must be non-empty")
        return self


class ScoreQuestion(_Strict):
    type: Literal["score"]
    instructions: str = Field(min_length=1)
    criteria: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS
    )


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class Options(_Strict):
    choice_method: Literal["independent", "letter"] = "independent"
    # How `score` questions are read (v0.3, additive). "auto" = "ens4d" on the VLM, "statements" elsewhere.
    # "statements" is the v0 method (one yes/no statement per level); "digits" (1 pass), "fast2" (2) and "ens4d" (4) are Glance elicitation
    # (`glance/rating.py`), which is calibrated per rubric with `glance fit`.
    score_method: Literal["auto", "statements", "digits", "fast2", "ens4d", "jsondigits"] = "auto"
    # true: every answer must be calibrated, or the request fails with calibration_mismatch (v0 behaviour).
    # "auto": calibrate the answers that have fitted parameters, warn about the others.
    calibrated: bool | Literal["auto"] = False


class DecideRequest(_Strict):
    model: Literal["siglip", "vlm", "frontier"]
    state: State
    questions: dict[str, Question] = Field(min_length=1)
    options: Options = Options()


# --- response -----------------------------------------------------------------------------------


class NoulAnswer(_Strict):
    type: Literal["noul"] = "noul"
    noul: float
    raw: float | None


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] | None
    confidence: float | None
    margin: float | None
    raw: dict[str, float] | None


class ScoreAnswer(_Strict):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float] | None
    confidence: float | None
    margin: float | None
    raw: dict[str, float] | None
    method: str | None = None  # "statements" | "digits" | "ens4d"; None for a frontier pick
    calibration: str | None = None  # version of the rubric calibration applied to this answer, if any


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(_Strict):
    image_tokens: int = 0
    text_tokens: int = 0
    forward_passes: int = 0
    output_tokens: int = 0  # frontier baseline only
    cost_usd: float | None = None  # frontier baseline only: list-price cost of the call


class DecideResponse(_Strict):
    request_id: str
    model: str
    prompt_version: str
    calibration_version: str | None
    answers: dict[str, Answer]
    usage: Usage
    timing_ms: dict[str, int]
    warnings: list[str]


class ErrorResponse(_Strict):
    request_id: str
    code: str
    message: str
    detail: Any = None


# --- errors -------------------------------------------------------------------------------------


class GlanceError(Exception):
    """An error with a stable code and HTTP status from the section 5 error table."""

    code = "backend_error"
    http_status = 500

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class RequestValidationError(GlanceError):
    code = "validation_error"
    http_status = 422


class UnsupportedQuestionError(GlanceError):
    code = "unsupported_question_for_backend"
    http_status = 400


class ImageLoadError(GlanceError):
    code = "image_load_failed"
    http_status = 400


class CalibrationMismatchError(GlanceError):
    code = "calibration_mismatch"
    http_status = 409


class BackendError(GlanceError):
    code = "backend_error"
    http_status = 500


def _field_path(loc: tuple[Any, ...]) -> str:
    """Pydantic puts the union tag after the question id; drop it so the path matches the request body."""
    parts = [str(p) for p in loc]
    if len(parts) >= 3 and parts[0] == "questions" and parts[2] in QUESTION_TYPES:
        del parts[2]
    return ".".join(parts)


def parse_request(body: Any) -> DecideRequest:
    """Validate a request body. Raises RequestValidationError whose detail names each failing field path."""
    try:
        return DecideRequest.model_validate(body)
    except ValidationError as exc:
        detail = [{"path": _field_path(err["loc"]), "message": err["msg"]} for err in exc.errors()]
        paths = ", ".join(d["path"] or "<body>" for d in detail)
        raise RequestValidationError(f"request failed validation at: {paths}", detail) from exc

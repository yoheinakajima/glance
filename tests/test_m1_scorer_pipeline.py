import json
import math

import numpy as np
import pytest

from glance import scorer
from glance.logging_utils import read_jsonl
from glance.pipeline import Engine
from glance.schema import GlanceError, UnsupportedQuestionError, parse_request

from .conftest import FakeBackend, needs_models

# --- scorer math on fixed logits ------------------------------------------------------------------


def test_sigmoid():
    assert scorer.sigmoid(0.0) == 0.5
    assert scorer.sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert scorer.sigmoid(-800.0) == 0.0 and scorer.sigmoid(800.0) == 1.0  # no overflow


def test_softmax_and_temperature():
    z = np.array([2.0, 1.0, 0.0])
    p = scorer.softmax(z)
    expected = np.exp(z) / np.exp(z).sum()
    assert p == pytest.approx(expected)
    assert scorer.softmax(z + 1000.0) == pytest.approx(expected)  # shift invariant, no overflow
    hot = scorer.softmax(z, temperature=0.5)
    cool = scorer.softmax(z, temperature=4.0)
    assert hot[0] > p[0] > cool[0]
    assert scorer.softmax(z, 2.0) == pytest.approx(np.exp(z / 2) / np.exp(z / 2).sum())


def test_score_mean():
    assert scorer.expected_score(np.array([0.02, 0.16, 0.82])) == pytest.approx(1.80)
    assert scorer.expected_score(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0
    assert scorer.expected_score(np.array([0.0, 0.0, 0.0, 1.0])) == 3.0


def test_confidence():
    assert scorer.confidence(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert scorer.confidence(np.full(5, 0.2)) == pytest.approx(0.0, abs=1e-12)
    p = np.array([0.91, 0.07, 0.02])
    entropy = -(p * np.log(p)).sum()
    assert scorer.confidence(p) == pytest.approx(1 - entropy / math.log(3))
    assert round(scorer.confidence(p), 2) == 0.68  # the HANDOFF example response


def test_margin():
    assert scorer.margin(np.array([0.91, 0.07, 0.02])) == pytest.approx(0.84)
    assert scorer.margin(np.array([0.5, 0.5])) == 0.0


def test_noul_confidence():
    assert scorer.noul_confidence(0.5) == 0.0
    assert scorer.noul_confidence(0.97) == pytest.approx(0.94)
    assert scorer.noul_confidence(0.03) == pytest.approx(0.94)


def test_letter_shifts():
    assert scorer.letter_shifts(3, 4) == [0, 1, 2]
    assert scorer.letter_shifts(8, 4) == [0, 2, 4, 6]
    assert scorer.letter_shifts(26, 4) == [0, 6, 13, 20]
    assert scorer.letter_shifts(2, 4) == [0, 1]


# --- assembly -------------------------------------------------------------------------------------


def _score(request_body, backend, method="independent"):
    from glance.images import load_images
    from glance.config import load_config

    req = parse_request(request_body)
    images = load_images(req.state.images, load_config().limits)
    return req, scorer.score_questions(backend, images, req.state.context, req.questions, choice_method=method)


def test_assembly_from_fixed_logits(request_body):
    backend = FakeBackend(fixed={"A red image": 3.0, "A blue image": 0.0, "other": -1.0,
                                 "Very dark": -2.0, "Medium": 0.0, "Very bright": 2.0})
    req, result = _score(request_body, backend)
    assert backend.calls == 1  # every statement in the request shares one backend call
    assert result.usage.forward_passes == 1 + 3 + 3

    color = result.scores["color"]
    assert color.keys == ["red", "blue", "other"] and color.method == "independent"
    raw = scorer.raw_probabilities(color)
    answer = scorer.build_answer(req.questions["color"], color, raw)
    assert answer.choice == "red"
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert answer.probabilities["red"] == pytest.approx(math.exp(3) / (math.exp(3) + 1 + math.exp(-1)), abs=1e-6)
    assert answer.raw == answer.probabilities  # uncalibrated: both are the raw distribution

    brightness = result.scores["brightness"]
    b_answer = scorer.build_answer(req.questions["brightness"], brightness, scorer.raw_probabilities(brightness))
    p = scorer.softmax(np.array([-2.0, 0.0, 2.0]))
    assert b_answer.score == pytest.approx(p[1] + 2 * p[2], abs=1e-6)
    assert b_answer.legend == {"0": "Very dark", "1": "Medium", "2": "Very bright"}

    noul = result.scores["is_red"]
    n_answer = scorer.build_answer(req.questions["is_red"], noul, scorer.raw_probabilities(noul))
    assert n_answer.noul == pytest.approx(scorer.sigmoid(noul.z[0]), abs=1e-6)
    assert not hasattr(n_answer, "confidence")  # noul carries neither confidence nor margin


def test_calibrated_answer_keeps_raw(request_body):
    req, result = _score(request_body, FakeBackend(fixed={"A red image": 3.0, "A blue image": 0.0, "other": -1.0}))
    color = result.scores["color"]
    raw = scorer.raw_probabilities(color)
    cal = scorer.softmax(color.z, temperature=3.0)
    answer = scorer.build_answer(req.questions["color"], color, raw, cal)
    assert answer.probabilities["red"] < answer.raw["red"]
    assert answer.confidence == pytest.approx(scorer.confidence(cal), abs=1e-6)


def test_independent_is_permutation_invariant(request_body):
    backend = FakeBackend()
    _, a = _score(request_body, backend)
    crit = request_body["questions"]["color"]["criteria"]
    request_body["questions"]["color"]["criteria"] = dict(reversed(list(crit.items())))
    _, b = _score(request_body, backend)
    pa = dict(zip(a.scores["color"].keys, scorer.raw_probabilities(a.scores["color"])))
    pb = dict(zip(b.scores["color"].keys, scorer.raw_probabilities(b.scores["color"])))
    assert pa == pytest.approx(pb)


def test_letter_maps_rotations_back_to_options(request_body):
    fixed = {"A red image": 3.0, "A blue image": 0.5, "other": -1.0}
    _, result = _score(request_body, FakeBackend(fixed=fixed), method="letter")
    color = result.scores["color"]
    assert color.method == "letter" and len(color.statements) == 3  # min(4, K) rotations
    assert color.z == pytest.approx([3.0, 0.5, -1.0])  # every rotation maps back to the same options
    assert result.scores["brightness"].method == "statement"  # letter only affects choice questions


def test_letter_cap(request_body):
    request_body["questions"]["color"]["criteria"] = {f"opt{i}": None for i in range(27)}
    with pytest.raises(UnsupportedQuestionError, match="capped at 26"):
        _score(request_body, FakeBackend(), method="letter")


def test_dual_encoder_rules(request_body):
    backend = FakeBackend()
    backend.kind = "dual_encoder"
    with pytest.raises(UnsupportedQuestionError, match="criteria.true"):
        _score(request_body, backend)

    request_body["questions"]["is_red"]["criteria"] = {"true": "a red image", "false": "a blue image"}
    backend = FakeBackend(fixed={"a red image": 2.0, "a blue image": -1.5})
    backend.kind = "dual_encoder"
    _, result = _score(request_body, backend)
    assert result.scores["is_red"].z == pytest.approx([3.5])  # z_true - z_false

    with pytest.raises(UnsupportedQuestionError, match="VLM-only"):
        _score(request_body, backend, method="letter")


# --- pipeline -------------------------------------------------------------------------------------


def test_pipeline_response_shape_and_call_log(cfg, request_body, png_b64):
    engine = Engine(cfg, source="cli", backends={"vlm": FakeBackend()})
    status, payload = engine.decide_json(request_body)
    assert status == 200
    assert set(payload) == {"request_id", "model", "prompt_version", "calibration_version", "answers", "usage",
                            "timing_ms", "warnings"}
    assert payload["model"] == "vlm:fake/model@" + "0" * 40
    assert payload["prompt_version"] == "p1" and payload["calibration_version"] is None
    assert set(payload["timing_ms"]) == {"load", "prefix", "score", "calibrate", "total"}
    assert payload["usage"] == {"image_tokens": 64, "text_tokens": 70, "forward_passes": 7}
    assert set(payload["answers"]["color"]) == {"type", "choice", "probabilities", "confidence", "margin", "raw"}
    assert set(payload["answers"]["brightness"]) == {"type", "score", "legend", "probabilities", "confidence", "margin", "raw"}
    assert set(payload["answers"]["is_red"]) == {"type", "noul", "raw"}

    rows = [r for f in sorted((cfg.path("logs") / "calls").glob("*.jsonl")) for r in read_jsonl(f)]
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == payload["request_id"] and row["source"] == "cli" and row["error"] is None
    assert row["images"][0]["sha256"] and png_b64 not in json.dumps(row)  # never the image bytes
    assert row["output"]["color"]["statements"][0].keys() >= {"prompt_hash", "z_yes", "z_no", "z", "off_mass"}
    assert row["cache"] == "hit" and row["forward_passes"] == 7
    for key in ("harness_version", "git_sha", "backend", "model_id", "model_revision", "prompt_version",
                "calibration_version", "choice_method", "device", "dtype", "image_token_budget", "questions", "context"):
        assert key in row


def test_pipeline_errors_are_shaped_and_logged(cfg, request_body):
    engine = Engine(cfg, backends={"vlm": FakeBackend()})

    status, payload = engine.decide_json({"model": "vlm"})
    assert status == 422 and payload["code"] == "validation_error"
    assert {"path": "state", "message": "Field required"} in payload["detail"]

    request_body["state"]["images"] = [{"id": "img0", "path": "/no/such.jpg"}]
    status, payload = engine.decide_json(request_body)
    assert status == 400 and payload["code"] == "image_load_failed"

    class Exploding(FakeBackend):
        def score_statements(self, *a, **k):
            raise RuntimeError("kaboom")

    status, payload = Engine(cfg, backends={"vlm": Exploding()}).decide_json(
        {**request_body, "state": {"images": [{"id": "img0", "path": "samples/dog.jpg"}]}}
    )
    assert status == 500 and payload["code"] == "backend_error" and "RuntimeError" in payload["message"]

    rows = [r for f in sorted((cfg.path("logs") / "calls").glob("*.jsonl")) for r in read_jsonl(f)]
    assert [r["error"]["code"] for r in rows] == ["validation_error", "image_load_failed", "backend_error"]
    assert "kaboom" in rows[-1]["error"]["traceback"]
    assert all(set(p) == {"request_id", "code", "message", "detail"} for p in [payload])


def test_calibrated_without_params_is_409(cfg, request_body):
    request_body["options"] = {"calibrated": True}
    status, payload = Engine(cfg, backends={"vlm": FakeBackend()}).decide_json(request_body)
    assert status == 409 and payload["code"] == "calibration_mismatch"


def test_engine_decide_raises_glance_error(cfg):
    with pytest.raises(GlanceError) as err:
        Engine(cfg, backends={"vlm": FakeBackend()}).decide({})
    assert err.value.request_id.startswith("req_")


@needs_models
@pytest.mark.parametrize("sample", ["receipt", "invoice", "dog"])
def test_bundled_samples_on_siglip(cfg, sample):
    from glance.config import PROJECT_ROOT

    body = json.loads((PROJECT_ROOT / "samples" / f"{sample}.json").read_text())
    body["model"] = "siglip"
    status, payload = _siglip_engine(cfg).decide_json(body)
    assert status == 200, payload
    assert set(payload["answers"]) == {"is_receipt", "doc_type", "legibility"}
    assert 0.0 <= payload["answers"]["is_receipt"]["noul"] <= 1.0
    assert payload["usage"]["image_tokens"] == 256


_ENGINES = {}


def _siglip_engine(cfg):
    if "siglip" not in _ENGINES:
        _ENGINES["siglip"] = Engine(cfg)
    engine = _ENGINES["siglip"]
    engine.cfg = cfg
    return engine

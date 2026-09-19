import json

import numpy as np
import pytest
from scipy.special import expit

from glance import calibration as C
from glance.pipeline import Engine
from glance.schema import CalibrationMismatchError
from glance.scorer import QuestionScore, softmax

from .conftest import FakeBackend, needs_models

KEY = C.CalibrationKey(backend="vlm", model="fake/model@" + "0" * 40, prompt_version="p1",
                       choice_method="independent", image_token_budget=64)

# --- fitting --------------------------------------------------------------------------------------


def test_platt_recovers_known_parameters():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 6, 4000)  # overconfident raw logits
    y = rng.random(4000) < expit(0.25 * z - 0.4)
    a, b = C.fit_platt(z, y)
    assert a == pytest.approx(0.25, abs=0.03) and b == pytest.approx(-0.4, abs=0.1)
    fit = C.fit_noul(z, y, ["s"])
    assert fit.method == "platt" and fit.nll_after < fit.nll_before and fit.ece_after < fit.ece_before


def test_temperature_recovers_known_value_with_ragged_options():
    rng = np.random.default_rng(1)
    zs, labels = [], []
    for i in range(1500):
        k = 5 if i % 2 else 12  # suites with different option counts pool into one fit
        true_logits = rng.normal(0, 1.5, k)
        labels.append(int(rng.choice(k, p=softmax(true_logits))))
        zs.append(true_logits * 4.0)  # the model reports logits 4x too sharp
    t = C.fit_temperature(zs, labels)
    assert t == pytest.approx(4.0, rel=0.12)
    fit = C.fit_multiclass(zs, labels, ["a", "b"])
    assert fit.nll_after < fit.nll_before and fit.ece_after < fit.ece_before and fit.suite_ids == ["a", "b"]


def test_isotonic_is_monotone_and_serializable():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 4, 2000)
    y = rng.random(2000) < expit(0.5 * z)
    fit = C.fit_noul(z, y, ["s"], isotonic=True)
    assert fit.method == "isotonic" and {"a", "b", "x", "y"} <= set(fit.params)
    grid = np.linspace(-10, 10, 50)
    p = [C.apply_fit(fit, np.array([v])) for v in grid]
    assert all(b >= a - 1e-12 for a, b in zip(p, p[1:]))
    json.dumps(fit.model_dump())


def _rows():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(300):
        z = float(rng.normal(0, 8))
        rows.append({"suite": "n1" if i % 2 else "n2", "type": "noul", "z": [z], "label_index": int(rng.random() < expit(0.3 * z))})
    for i in range(300):
        logits = rng.normal(0, 1, 6)
        rows.append({"suite": "c1", "type": "choice", "z": (logits * 5).tolist(), "label_index": int(rng.choice(6, p=softmax(logits)))})
    for i in range(200):
        logits = rng.normal(0, 1, 4)
        rows.append({"suite": "s1", "type": "score", "z": (logits * 3).tolist(), "label_index": int(rng.choice(4, p=softmax(logits)))})
    return rows


def test_build_save_load_roundtrip(tmp_path):
    params = C.build_params(KEY, _rows(), source_run="run123")
    assert set(params.types) == {"noul", "choice", "score"} and set(params.per_suite) == {"n1", "n2", "c1", "s1"}
    assert params.version == KEY.version() and params.version.startswith("cal_")
    assert params.types["noul"].suite_ids == ["n1", "n2"] and params.types["noul"].n == 300
    path = C.save_params(tmp_path, params)
    assert path.name == f"{KEY.hash()}.json"
    saved = json.loads(path.read_text())
    for field in ("key", "fit_date", "types", "per_suite", "source_run"):
        assert field in saved
    for field in ("params", "n", "suite_ids", "nll_before", "nll_after", "ece_before", "ece_after"):
        assert field in saved["types"]["choice"]
    assert C.load_params(tmp_path, KEY) == params


def test_loading_a_different_key_is_an_error(tmp_path):
    params = C.build_params(KEY, _rows(), source_run=None)
    C.save_params(tmp_path, params)
    other = KEY.model_copy(update={"image_token_budget": 384})
    with pytest.raises(CalibrationMismatchError):
        C.load_params(tmp_path, other)  # no file for this key
    # A file that sits at the right path but was fit for another configuration is also rejected.
    C.params_path(tmp_path, other).write_text(params.model_dump_json())
    with pytest.raises(CalibrationMismatchError, match="different configuration"):
        C.load_params(tmp_path, other)


def test_apply_by_type():
    params = C.build_params(KEY, _rows(), source_run=None)
    noul = QuestionScore("q", "noul", ["true"], "statement", z=np.array([10.0]))
    a, b = params.types["noul"].params["a"], params.types["noul"].params["b"]
    assert C.apply(noul, params) == pytest.approx(expit(a * 10 + b))
    choice = QuestionScore("q", "choice", list("abcdef"), "independent", z=np.array([5.0, 0, 0, 0, 0, 0]))
    cal = C.apply(choice, params)
    assert cal.sum() == pytest.approx(1.0) and cal[0] < softmax(choice.z)[0]  # T > 1 softens
    del params.types["score"]
    with pytest.raises(CalibrationMismatchError):
        C.apply(QuestionScore("q", "score", ["0", "1"], "statement", z=np.array([0.0, 1.0])), params)


# --- pipeline and server ----------------------------------------------------------------------------


def test_pipeline_applies_calibration(cfg, request_body):
    backend = FakeBackend(fixed={"A red image": 6.0, "A blue image": 0.0, "other": -2.0})
    engine = Engine(cfg, backends={"vlm": backend})
    key = engine.calibration_key(backend, "independent")
    C.save_params(cfg.path("calibration"), C.build_params(key, _rows(), source_run="r1"))

    request_body["options"] = {"calibrated": True}
    status, payload = engine.decide_json(request_body)
    assert status == 200 and payload["calibration_version"] == key.version()
    color = payload["answers"]["color"]
    assert color["raw"]["red"] > color["probabilities"]["red"] > 1 / 3  # tempered, same argmax
    assert payload["answers"]["is_red"]["raw"] != payload["answers"]["is_red"]["noul"]

    request_body["options"] = {"calibrated": True, "choice_method": "letter"}  # a different key: never a warning
    status, payload = engine.decide_json(request_body)
    assert status == 409 and payload["code"] == "calibration_mismatch"


@pytest.fixture
def client(cfg):
    from glance.server import create_app

    app = create_app(cfg, engine=Engine(cfg, source="server", backends={"vlm": FakeBackend()}))
    return app.test_client()


def test_decide_round_trip(client, request_body):
    resp = client.post("/v1/decide", json=request_body)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["request_id"].startswith("req_") and set(body["answers"]) == {"is_red", "color", "brightness"}
    assert body["answers"]["color"]["choice"] in ("red", "blue", "other")


def test_server_error_shapes(client, request_body):
    resp = client.post("/v1/decide", json={"model": "vlm", "state": {"images": []}, "questions": {}})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["code"] == "validation_error" and {d["path"] for d in body["detail"]} >= {"state.images", "questions"}

    resp = client.post("/v1/decide", data="not json", content_type="application/json")
    assert resp.status_code == 422 and resp.get_json()["code"] == "validation_error"

    request_body["state"]["images"] = [{"id": "img0", "path": "/nope.jpg"}]
    resp = client.post("/v1/decide", json=request_body)
    assert resp.status_code == 400 and resp.get_json()["code"] == "image_load_failed"

    request_body["options"] = {"calibrated": True}
    request_body["state"]["images"] = [{"id": "img0", "path": "samples/dog.jpg"}]
    assert client.post("/v1/decide", json=request_body).status_code == 409


def test_models_and_healthz(client):
    models = client.get("/v1/models").get_json()
    assert models["loaded"][0]["name"] == "vlm" and models["loaded"][0]["revision"] == "0" * 40
    assert models["prompt_version"] == "p1"
    health = client.get("/healthz").get_json()
    assert health["status"] == "ok" and health["loaded"] == ["vlm"] and "device" in health


def test_server_binds_to_localhost(cfg):
    assert cfg.server.host == "127.0.0.1"


@needs_models
def test_real_server_round_trip_offline(cfg, monkeypatch):
    """POST /v1/decide against the real SigLIP2 weights with the Hub switched off."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from glance.config import PROJECT_ROOT
    from glance.server import create_app

    client = create_app(cfg, preload=["siglip"]).test_client()
    assert client.get("/healthz").get_json()["loaded"] == ["siglip"]
    body = json.loads((PROJECT_ROOT / "samples" / "dog.json").read_text())
    resp = client.post("/v1/decide", json=body)
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["answers"]["doc_type"]["choice"] == "other"

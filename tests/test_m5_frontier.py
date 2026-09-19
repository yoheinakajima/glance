import json
import types

import pytest

from glance.backends.base import PickItem
from glance.backends.frontier import FrontierBackend
from glance.pipeline import FRONTIER_REDACTED, Engine
from glance.logging_utils import read_jsonl


@pytest.fixture
def frontier(cfg, monkeypatch):
    monkeypatch.setenv("FRONTIER_MODEL", "fakeprovider/vision-large")
    return FrontierBackend(cfg)


def _fake_completion(answers):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(content=json.dumps(answers))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                     usage=types.SimpleNamespace(prompt_tokens=1234))

    return completion, calls


def test_requires_model_id(cfg, monkeypatch):
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    with pytest.raises(Exception, match="FRONTIER_MODEL"):
        FrontierBackend(cfg)


def test_schema_enumerates_allowed_answers(frontier):
    schema = frontier.answer_schema([PickItem("p0", ["Yes", "No"]), PickItem("p1", ["a", "b", "c"])])
    assert schema["properties"]["q0"]["enum"] == ["Yes", "No"] and schema["properties"]["q1"]["enum"] == ["a", "b", "c"]
    assert schema["required"] == ["q0", "q1"] and schema["additionalProperties"] is False


def test_frontier_answers_are_hard_picks_and_never_logged(cfg, frontier, request_body, monkeypatch):
    import litellm

    completion, calls = _fake_completion({"q0": "Yes", "q1": "blue", "q2": "2"})
    monkeypatch.setattr(litellm, "completion", completion)
    request_body["model"] = "frontier"
    engine = Engine(cfg, backends={"frontier": frontier}, allow_frontier=True)
    status, payload = engine.decide_json(request_body)
    assert status == 200, payload
    assert payload["model"] == "frontier:fakeprovider/vision-large"
    assert payload["answers"]["is_red"] == {"type": "noul", "noul": 1.0, "raw": None}
    assert payload["answers"]["color"]["choice"] == "blue" and payload["answers"]["color"]["probabilities"] is None
    assert payload["answers"]["brightness"]["score"] == 2.0 and payload["answers"]["brightness"]["probabilities"] is None

    call = calls[0]
    assert call["model"] == "fakeprovider/vision-large" and call["temperature"] == 0
    assert call["response_format"]["json_schema"]["schema"]["properties"]["q1"]["enum"] == ["red", "blue", "other"]
    kinds = [part["type"] for part in call["messages"][1]["content"]]
    assert kinds == ["text", "image_url", "text"]  # image(s), then context and questions
    assert "is_red" not in json.dumps(call["messages"])  # question ids are never sent to the model

    row = [r for f in (cfg.path("logs") / "calls").glob("*.jsonl") for r in read_jsonl(f)][-1]
    assert row["model_id"] == "fakeprovider/vision-large"  # the exact id is logged
    assert all(q["answer"] == FRONTIER_REDACTED for q in row["output"].values())  # but never the picks
    assert not any(set(q) & {"choice", "pick", "noul", "score"} for q in row["output"].values())


def test_answer_outside_the_enum_is_a_backend_error(cfg, frontier, request_body, monkeypatch):
    import litellm

    completion, _ = _fake_completion({"q0": "Maybe", "q1": "blue", "q2": "2"})
    monkeypatch.setattr(litellm, "completion", completion)
    request_body["model"] = "frontier"
    status, payload = Engine(cfg, backends={"frontier": frontier}, allow_frontier=True).decide_json(request_body)
    assert status == 500 and payload["code"] == "backend_error"


def test_frontier_is_refused_without_opt_in(cfg, frontier, request_body):
    request_body["model"] = "frontier"
    status, payload = Engine(cfg, backends={"frontier": frontier}).decide_json(request_body)
    assert status == 400 and "eval-only" in payload["message"]


def test_eval_skips_frontier_cleanly_without_key(cfg, monkeypatch):
    from glance.evals import run

    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    notes = []
    assert run._frontier_ready(run.RunArgs(confirm_spend=True), notes) is False
    assert "FRONTIER_MODEL is not set" in notes[0]


def test_eval_requires_confirm_spend(cfg, monkeypatch):
    import litellm

    from glance.evals import run

    monkeypatch.setenv("FRONTIER_MODEL", "fakeprovider/vision-large")
    monkeypatch.setattr(litellm, "validate_environment", lambda model: {"keys_in_environment": True, "missing_keys": []})
    notes = []
    assert run._frontier_ready(run.RunArgs(confirm_spend=False), notes) is False
    assert "--confirm-spend" in notes[0]
    assert run._frontier_ready(run.RunArgs(confirm_spend=True), []) is True

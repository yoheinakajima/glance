"""`glance baseline`: interactive frontier baseline with an in-memory key (no .env)."""

import json
import re
import types

import pytest

from glance.evals import baseline, run
from glance.logging_utils import read_jsonl

from .test_m3_evals import KnowingBackend, fake_suite  # noqa: F401  (fixture)

SECRET = "sk-test-THISISASECRETKEY1234567890"


def _fake_litellm(monkeypatch, reject_temperature=False, explode_with_key=False):
    import litellm

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if explode_with_key:
            raise RuntimeError(f"Incorrect API key provided: {kwargs.get('api_key')}")
        if reject_temperature and "temperature" in kwargs:
            raise litellm.BadRequestError(message="`temperature` is not supported for this model",
                                          model=kwargs["model"], llm_provider="fake")
        text = json.dumps(kwargs["messages"])
        enum = kwargs["response_format"]["json_schema"]["schema"]["properties"]["q0"]["enum"]
        match = re.search(r"\(#(\d+)\)", text)
        if match:  # right on most items, wrong on every 4th
            n = int(match.group(1))
            answer = f"k{n % 30}" if n % 4 else enum[0]
        else:
            answer = "Yes"
        message = types.SimpleNamespace(content=json.dumps({"q0": answer}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                     usage=types.SimpleNamespace(prompt_tokens=900))

    monkeypatch.setattr(litellm, "completion", completion)
    return calls


@pytest.fixture
def local_run(cfg, fake_suite, monkeypatch):  # noqa: F811
    monkeypatch.setattr(run, "Engine", lambda cfg, source: __import__("glance.pipeline", fromlist=["Engine"]).Engine(
        cfg, source=source, backends={"vlm": KnowingBackend()}))
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    args = run.RunArgs(suites=[fake_suite], models=["vlm", "frontier"], n=40, choice_methods=["independent"],
                       skip_permutation=True, skip_latency=True)
    return run.run_eval(cfg, args)


def _all_text(*roots):
    out = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".yaml", ".md", ".txt"):
                out.append(path.read_text(errors="ignore"))
    return "\n".join(out)


def test_prompts_take_defaults_and_hide_the_key():
    answers = iter(["", ""])  # Enter, Enter: first provider, its default model
    shown = []
    model, key = baseline.ask_model_and_key(None, input_fn=lambda p: next(answers),
                                            secret_fn=lambda p: shown.append(p) or SECRET, out=open("/dev/null", "w"))
    assert model == baseline.PROVIDERS[0]["model"] == "anthropic/claude-opus-5" and key == SECRET
    assert "hidden" in shown[0]
    other = iter(["4", "someprovider/some-model"])
    assert baseline.ask_model_and_key(None, input_fn=lambda p: next(other), secret_fn=lambda p: SECRET,
                                      out=open("/dev/null", "w"))[0] == "someprovider/some-model"


def test_baseline_fills_the_go_no_go_rows_and_never_stores_the_key(cfg, local_run, fake_suite, monkeypatch, tmp_path):  # noqa: F811
    calls = _fake_litellm(monkeypatch)
    before = json.loads((local_run / "metrics.json").read_text())
    assert "not measured" in before["go_no_go"][0]["measured"]

    done = baseline.add_baseline(cfg, local_run, "fakeprovider/vision-large", SECRET,
                                 input_fn=lambda prompt: "y", out=open("/dev/null", "w"))
    assert done == local_run
    assert len(calls) == 1 + 20  # one check call, then the 20 test items
    assert all(c["api_key"] == SECRET and c["model"] == "fakeprovider/vision-large" for c in calls)

    rows = [r for r in read_jsonl(local_run / "predictions.jsonl") if r["backend"] == "frontier"]
    assert len(rows) == 20 and {r["split"] for r in rows} == {"test"}
    assert all("correct" in r and "raw" not in r and "z" not in r for r in rows)  # right or wrong, never the pick

    after = json.loads((local_run / "metrics.json").read_text())
    assert "points over 1 suites" in after["go_no_go"][0]["measured"] and after["go_no_go"][0]["pass"] in (True, False)
    assert "vs baseline" in after["go_no_go"][2]["measured"]
    assert after["baseline"][fake_suite]["frontier"]["n"] == 20
    report = (local_run / "report.md").read_text()
    assert "frontier baseline added on" in report and "FRONTIER_MODEL is not set" not in report
    assert "fakeprovider/vision-large" in (local_run / "summary.txt").read_text()

    assert SECRET not in _all_text(tmp_path)  # run dir, call logs, everything written during the test

    # Running it again has nothing left to do and spends nothing.
    assert baseline.add_baseline(cfg, local_run, "fakeprovider/vision-large", SECRET,
                                 input_fn=lambda prompt: "y", out=open("/dev/null", "w")) is None
    assert len(calls) == 21


def test_declining_the_estimate_spends_only_the_check_call(cfg, local_run, monkeypatch):
    calls = _fake_litellm(monkeypatch)
    assert baseline.add_baseline(cfg, local_run, "fakeprovider/vision-large", SECRET,
                                 input_fn=lambda prompt: "n", out=open("/dev/null", "w")) is None
    assert len(calls) == 1
    assert not [r for r in read_jsonl(local_run / "predictions.jsonl") if r["backend"] == "frontier"]


def test_temperature_refusal_is_retried_and_reported(cfg, request_body, monkeypatch):
    from glance.backends.frontier import FrontierBackend
    from glance.pipeline import Engine

    calls = _fake_litellm(monkeypatch, reject_temperature=True)
    request_body["model"] = "frontier"
    request_body["questions"] = {"is_red": request_body["questions"]["is_red"]}
    engine = Engine(cfg, backends={"frontier": FrontierBackend(cfg, "fakeprovider/vision-large", SECRET)}, allow_frontier=True)
    status, payload = engine.decide_json(request_body)
    assert status == 200 and "rejects `temperature`" in payload["warnings"][0]
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    engine.decide_json(request_body)
    assert "temperature" not in calls[2] and len(calls) == 3  # remembered: no second refusal round-trip


def test_a_bad_key_stops_at_the_check_and_is_scrubbed_from_logs(cfg, local_run, monkeypatch, tmp_path):
    _fake_litellm(monkeypatch, explode_with_key=True)
    import io

    out = io.StringIO()
    assert baseline.add_baseline(cfg, local_run, "fakeprovider/vision-large", SECRET,
                                 input_fn=lambda prompt: "y", out=out) is None
    assert "test call failed" in out.getvalue() and SECRET not in out.getvalue() and "<api-key>" in out.getvalue()
    assert SECRET not in _all_text(tmp_path)


def test_latest_full_run_picks_the_biggest(cfg, local_run):
    assert baseline.latest_full_run(cfg) == local_run

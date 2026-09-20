import json

import numpy as np
import pytest
from PIL import Image

from glance.evals import metrics as M
from glance.evals import report, run
from glance.evals.suites import SUITES
from glance.evals.suites.base import RawItem, SuiteInfo, SuiteSkipped, materialize
from glance.logging_utils import read_jsonl

from .conftest import FakeBackend

# --- metrics --------------------------------------------------------------------------------------


def test_ece_equal_mass_known_values():
    from glance.calibration import ece_equal_mass

    conf = np.array([0.9] * 10 + [0.6] * 10)
    correct = np.array([1] * 9 + [0] + [1] * 6 + [0] * 4)
    assert ece_equal_mass(conf, correct, n_bins=2) == pytest.approx(0.0)
    assert ece_equal_mass(np.full(10, 0.9), np.array([1] * 5 + [0] * 5), n_bins=1) == pytest.approx(0.4)
    # bins hold equal counts even when confidences are skewed
    skewed = np.concatenate([np.full(28, 0.99), np.array([0.5, 0.6])])
    bins = M.reliability_bins(skewed, np.ones(30), n_bins=15)
    assert [b["n"] for b in bins] == [2] * 15


def test_selective_accuracy_and_risk_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    correct = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0])
    sel = M.selective_accuracy(conf, correct)
    assert sel == {"50": 1.0, "80": 0.75, "90": pytest.approx(6 / 9), "100": 0.6}
    coverage, risk = M.risk_coverage(conf, correct)
    assert coverage[-1] == 1.0 and risk[-1] == pytest.approx(0.4) and risk[0] == 0.0


def _noul_rows(p, y):
    return [{"type": "noul", "label_index": int(t), "raw": float(v), "keys": ["true"], "item_id": str(i), "image_path": "x"}
            for i, (v, t) in enumerate(zip(p, y))]


def test_noul_metrics():
    rows = _noul_rows([0.9, 0.8, 0.3, 0.6], [1, 1, 0, 0])
    m = M.probability_metrics(rows, "raw", n_bins=2)
    assert m["accuracy"] == 0.75 and m["auroc"] == 1.0
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.09, 0.36]))
    assert m["nll"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.7, 0.4])))
    assert M.confusion_patterns(rows, "raw") == [{"true": "no", "predicted": "yes", "count": 1}]
    errors = M.top_confident_errors(rows, "raw")
    assert [e["item_id"] for e in errors] == ["3"]


def test_choice_and_score_metrics():
    keys = ["a", "b", "c"]
    rows = [
        {"type": "choice", "label_index": 0, "raw": [0.7, 0.2, 0.1], "keys": keys, "item_id": "0", "image_path": "x"},
        {"type": "choice", "label_index": 1, "raw": [0.6, 0.3, 0.1], "keys": keys, "item_id": "1", "image_path": "x"},
        {"type": "choice", "label_index": 2, "raw": [0.1, 0.1, 0.8], "keys": keys, "item_id": "2", "image_path": "x"},
    ]
    m = M.probability_metrics(rows, "raw", n_bins=3)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["nll"] == pytest.approx(-np.mean(np.log([0.7, 0.3, 0.8])))
    assert 0 < m["macro_f1"] < 1 and "mae_levels" not in m
    assert M.confusion_patterns(rows, "raw") == [{"true": "b", "predicted": "a", "count": 1}]

    score_rows = [{**r, "type": "score"} for r in rows]
    s = M.probability_metrics(score_rows, "raw", n_bins=3)
    assert s["mae_levels"] == pytest.approx(np.mean([0.4, 0.5, 0.3]))


def test_throughput_and_latency():
    rows = [{"latency_ms": 100.0, "forward_passes": 10, "image_tokens": 300, "off_mass": [0.0, 0.02]},
            {"latency_ms": 300.0, "forward_passes": 30, "image_tokens": 500, "off_mass": [0.04]}]
    t = M.throughput(rows)
    assert t["latency"]["p50_ms"] == 200.0 and t["statements_per_second"] == pytest.approx(100.0)
    assert t["mean_image_tokens"] == 400 and t["off_mass_mean"] == pytest.approx(0.02)


# --- suites ---------------------------------------------------------------------------------------


def _fake_raw_items(n):
    def writer(color):
        return lambda dest: Image.new("RGB", (32, 32), color).save(dest)

    return [RawItem(item_id=f"item{i:03d}", question={"type": "noul", "instructions": "Is `img0` red?"},
                    label=i % 2 == 0, write_image=writer((i * 7 % 255, 0, 0))) for i in range(n)]


def test_materialize_is_seeded_balanced_and_manifested(cfg, tmp_path):
    cfg.paths.eval_images = str(tmp_path / "images")
    cfg.eval.manifest_n = 20
    info = SuiteInfo(name="fake", qtype="noul", source="test", license="CC0")
    manifest = tmp_path / "fake.jsonl"
    a = materialize(cfg, info, _fake_raw_items(50), n=10, manifest_path=manifest)
    b = materialize(cfg, info, _fake_raw_items(50), n=10, manifest_path=manifest)
    assert [i.item_id for i in a] == [i.item_id for i in b] and len(a) == 10
    assert [i.split for i in a] == ["calibration", "test"] * 5  # any prefix of the seeded order stays 50/50
    rows = read_jsonl(manifest)
    assert len(rows) == 20 and set(rows[0]) == {"item_id", "image_sha256", "split"}
    assert [i.item_id for i in a] != sorted(i.item_id for i in a)  # shuffled

    drifted = _fake_raw_items(50)
    drifted[0], drifted[1] = drifted[1], drifted[0]
    with pytest.raises(RuntimeError, match="no longer matches its manifest"):
        materialize(cfg, info, drifted, n=10, manifest_path=manifest)


def test_suite_registry_and_skips(cfg, tmp_path):
    assert set(SUITES) == {"pope", "gqa_yesno", "pets37", "caltech101", "blur_ladder", "doctype16", "human_gold"}
    with pytest.raises(SuiteSkipped, match="license"):
        SUITES["doctype16"].build(cfg, 10)
    cfg.paths.human_gold = str(tmp_path / "gold" / "human_gold.jsonl")
    with pytest.raises(SuiteSkipped):
        SUITES["human_gold"].build(cfg, 10)
    assert "siglip" not in SUITES["gqa_yesno"].INFO.backends


def test_human_gold_loader(cfg, tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    rows = []
    for i in range(6):
        Image.new("RGB", (40, 40), (i * 30, 10, 10)).save(gold / f"img_{i}.jpg")
        rows.append({"image": str(gold / f"img_{i}.jpg"),
                     "question": {"type": "choice", "instructions": "What is `img0`?", "criteria": {"a": None, "b": None}},
                     "label": "a", "annotators": {"a1": "a", "a2": "b"}})
    (gold / "human_gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg.paths.human_gold = str(gold / "human_gold.jsonl")
    cfg.paths.eval_images = str(tmp_path / "images")
    items = SUITES["human_gold"].build(cfg, 4)
    assert len(items) == 4 and items[0].meta["human_disagreement"] == 0.5
    assert (gold / "human_gold.manifest.jsonl").exists()  # private manifest stays next to the gold file


def test_letter_subset_keeps_label_and_order():
    from glance.evals.suites.base import EvalItem

    criteria = {f"opt{i:03d}": None for i in range(101)}
    item = EvalItem("s", "id1", "test", "x.jpg", "0" * 64, {"type": "choice", "instructions": "?", "criteria": criteria}, "opt077")
    q1 = run.letter_subset(item, 26, 7)
    q2 = run.letter_subset(item, 26, 7)
    assert q1 == q2 and len(q1["criteria"]) == 26 and "opt077" in q1["criteria"]
    assert list(q1["criteria"]) == sorted(q1["criteria"])  # original order preserved
    small = EvalItem("s", "id2", "test", "x.jpg", "0" * 64, {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}, "a")
    assert run.letter_subset(small, 26, 7) is small.question


# --- end to end with a fake backend -----------------------------------------------------------------


@pytest.fixture
def fake_suite(cfg, tmp_path, monkeypatch):
    """A 40-item, 30-option choice suite where option `k<label>` is right and the fake backend mostly knows it."""
    cfg.paths.eval_images = str(tmp_path / "images")
    cfg.eval.manifest_n = 40
    cfg.eval.warmup_items = 4
    cfg.eval.latency_repeats = 3
    criteria = {f"k{i}": f"thing number {i}" for i in range(30)}
    info = SuiteInfo(name="fakechoice", qtype="choice", source="test", license="CC0")

    class Module:
        INFO = info

        @staticmethod
        def build(cfg, n):
            raw = []
            for i in range(40):
                raw.append(RawItem(item_id=f"it{i:02d}", label=f"k{i % 30}",
                                   question={"type": "choice", "instructions": f"Which thing is in `img0`? (#{i})", "criteria": criteria},
                                   write_image=lambda dest, i=i: Image.new("RGB", (32, 32), (i * 6, 50, 50)).save(dest)))
            return materialize(cfg, info, raw, n, manifest_path=tmp_path / "fakechoice.jsonl")

    monkeypatch.setitem(SUITES, "fakechoice", Module)
    return "fakechoice"


class KnowingBackend(FakeBackend):
    """Gives the right option a high logit for most items, so accuracy is high but not perfect."""

    def _boost(self, text):
        import re

        m = re.search(r"\(#(\d+)\)", text)
        c = re.search(r"thing number (\d+)", text)
        if m and c and int(m.group(1)) % 30 == int(c.group(1)) and int(m.group(1)) % 5 != 0:
            return 9.0
        return 0.0

    def score_statements(self, images, context, statements):
        out = super().score_statements(images, context, statements)
        out.z = out.z + np.array([self._boost(s.text) for s in statements])
        return out

    def score_labels(self, images, context, prompts, labels):
        out = super().score_labels(images, context, prompts, labels)
        for r, prompt in enumerate(prompts):
            lines = [ln for ln in prompt.splitlines() if len(ln) > 2 and ln[1:3] == ". "]
            for j, line in enumerate(lines):
                out.logits[r, j] += self._boost(prompt.splitlines()[0] + " " + line)
        return out


def test_eval_run_writes_complete_run_directory(cfg, fake_suite, monkeypatch):
    monkeypatch.setattr(run, "Engine", lambda cfg, source: __import__("glance.pipeline", fromlist=["Engine"]).Engine(
        cfg, source=source, backends={"vlm": KnowingBackend()}))
    monkeypatch.delenv("FRONTIER_MODEL", raising=False)
    args = run.RunArgs(suites=[fake_suite], models=["vlm", "frontier"], n=40, calibrate=True)
    run_dir = run.run_eval(cfg, args)

    for name in ("config.yaml", "env.json", "predictions.jsonl", "metrics.json", "report.md", "summary.txt", "extras.json"):
        assert (run_dir / name).exists(), name
    assert list((run_dir / "plots").glob("*__reliability.png")) and list((run_dir / "plots").glob("*__risk_coverage.png"))
    assert list((run_dir / "calibration").glob("*.json"))

    rows = read_jsonl(run_dir / "predictions.jsonl")
    assert len(rows) == 80  # 40 items x (independent, letter)
    letter = [r for r in rows if r["method"] == "letter"]
    assert all(len(r["keys"]) == 26 and r["n_options_full"] == 30 for r in letter)
    assert all(r["calibrated"] is not None and r["request_id"].startswith("req_") for r in rows)
    assert {r["split"] for r in rows} == {"calibration", "test"}

    metrics = json.loads((run_dir / "metrics.json").read_text())
    unit = metrics["units"][f"{fake_suite}|vlm|independent"]
    assert unit["n_test"] == 20 and unit["valid"] and 0.5 < unit["raw"]["accuracy"] < 1.0
    assert np.isfinite(unit["calibrated"]["nll"]) and "calibrated_suite_fit" in unit
    fit = next(iter(metrics["calibration"].values()))["types"]["choice"]
    assert fit["nll_after"] <= fit["nll_before"] + 1e-9  # on the split it was fit on, temperature can only help
    assert metrics["permutation"][f"{fake_suite}|vlm|independent"]["max_abs_dp"] <= 1e-9
    assert metrics["latency"]["vlm"]["statements"] == 11
    assert f"{fake_suite}|vlm" in metrics["letter_vs_independent"]
    assert metrics["verdict"] in ("GO", "PARTIAL", "NO-GO")
    assert [r["metric"] for r in metrics["go_no_go"]][0] == "Accuracy gap vs frontier baseline"
    assert "not measured" in metrics["go_no_go"][0]["measured"]

    text = (run_dir / "report.md").read_text()
    assert text.splitlines()[2].startswith("**Go/no-go:")
    assert "| Metric | Threshold | Measured | Pass |" in text and "Highest-confidence errors" in text
    assert "FRONTIER_MODEL is not set" in text
    assert "GLANCE v0 SUMMARY" in (run_dir / "summary.txt").read_text()


def test_failed_items_are_logged_counted_and_invalidate_the_suite(cfg, fake_suite, monkeypatch):
    class Flaky(KnowingBackend):
        def score_statements(self, images, context, statements):
            if "(#7)" in statements[0].text or "(#9)" in statements[0].text:
                raise RuntimeError("synthetic failure")
            return super().score_statements(images, context, statements)

    monkeypatch.setattr(run, "Engine", lambda cfg, source: __import__("glance.pipeline", fromlist=["Engine"]).Engine(
        cfg, source=source, backends={"vlm": Flaky()}))
    args = run.RunArgs(suites=[fake_suite], models=["vlm"], n=40, choice_methods=["independent"],
                       skip_permutation=True, skip_latency=True)
    run_dir = run.run_eval(cfg, args)
    errors = read_jsonl(run_dir / "errors.jsonl")
    assert len(errors) == 2 and errors[0]["error"]["code"] == "backend_error"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    unit = metrics["units"][f"{fake_suite}|vlm|independent"]
    assert unit["failures"]["failed"] == 2 and unit["valid"] is False  # 2/40 = 5% > 2%
    assert len(read_jsonl(run_dir / "predictions.jsonl")) == 38  # the run continued past the failures


def test_resume_skips_finished_rows(cfg, fake_suite, monkeypatch):
    backend = KnowingBackend()
    monkeypatch.setattr(run, "Engine", lambda cfg, source: __import__("glance.pipeline", fromlist=["Engine"]).Engine(
        cfg, source=source, backends={"vlm": backend}))
    args = run.RunArgs(suites=[fake_suite], models=["vlm"], n=40, choice_methods=["independent"],
                       skip_permutation=True, skip_latency=True)
    run_dir = run.run_eval(cfg, args)
    calls = backend.calls
    args.resume = run_dir.name
    run.run_eval(cfg, args)
    assert backend.calls == calls  # nothing was re-scored
    assert len(read_jsonl(run_dir / "predictions.jsonl")) == 40


def test_trim_to_max_hours(cfg, fake_suite, monkeypatch):
    import time as _time

    class Slow(KnowingBackend):
        def score_statements(self, images, context, statements):
            _time.sleep(0.05)
            return super().score_statements(images, context, statements)

    monkeypatch.setattr(run, "Engine", lambda cfg, source: __import__("glance.pipeline", fromlist=["Engine"]).Engine(
        cfg, source=source, backends={"vlm": Slow()}))
    args = run.RunArgs(suites=[fake_suite], models=["vlm"], n=40, choice_methods=["independent"], max_hours=1.0 / 3600,
                       skip_permutation=True, skip_latency=True)
    run_dir = run.run_eval(cfg, args)
    import yaml

    trim = yaml.safe_load((run_dir / "config.yaml").read_text())["trim"]
    final_n = trim["final_n"][fake_suite]
    assert trim["trimmed"] and final_n < 40 and final_n % 2 == 0
    assert "trimmed" in (run_dir / "report.md").read_text()
    rows = read_jsonl(run_dir / "predictions.jsonl")
    assert len(rows) == final_n


def test_trim_only_cuts_the_expensive_suite(cfg, fake_suite, tmp_path, monkeypatch):
    import time as _time

    import yaml

    info = SuiteInfo(name="fakenoul", qtype="noul", source="test", license="CC0")

    class CheapModule:
        INFO = info

        @staticmethod
        def build(cfg, n):
            return materialize(cfg, info, _fake_raw_items(40), n, manifest_path=tmp_path / "fakenoul.jsonl")

    monkeypatch.setitem(SUITES, "fakenoul", CheapModule)

    class PerStatementCost(KnowingBackend):
        def score_statements(self, images, context, statements):
            _time.sleep(0.004 * len(statements))  # 30 options cost 30x a single noul statement
            return super().score_statements(images, context, statements)

    monkeypatch.setattr(run, "Engine", lambda cfg, source: __import__("glance.pipeline", fromlist=["Engine"]).Engine(
        cfg, source=source, backends={"vlm": PerStatementCost()}))
    args = run.RunArgs(suites=[fake_suite, "fakenoul"], models=["vlm"], n=40, choice_methods=["independent"],
                       max_hours=2.2 / 3600, skip_permutation=True, skip_latency=True)
    run_dir = run.run_eval(cfg, args)
    final_n = yaml.safe_load((run_dir / "config.yaml").read_text())["trim"]["final_n"]
    assert final_n["fakenoul"] == 40 and final_n[fake_suite] < 40


def test_ece_noise_floor_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = M.ece_noise_floor(rng.uniform(0.7, 1.0, 100))
    large = M.ece_noise_floor(rng.uniform(0.7, 1.0, 4000))
    assert small > 0.04 > large > 0.0  # a perfectly calibrated model still shows ECE ~0.05+ at n=100

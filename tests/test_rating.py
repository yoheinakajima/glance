"""Glance elicitation in the harness: prompts identical to the lab's, packed calls, per-rubric calibration, `fit`."""

import json

import numpy as np
import pytest
from PIL import Image

from glance import rating
from glance.lab import score_methods as sm
from glance.pipeline import Engine
from glance.schema import parse_request

from .conftest import FakeBackend

LEVELS = ["Sharp", "Slightly soft", "Blurry", "Very blurry"]
QUESTION = "How blurry is `img0`?"


class CountingBackend(FakeBackend):
    """Records every `score_labels` call: which images it saw and how many prompts were packed into it."""

    def __init__(self, fixed=None):
        super().__init__(fixed)
        self.label_calls = []

    def score_labels(self, images, context, prompts, labels):
        self.label_calls.append({"images": [img.id for img in images], "prompts": list(prompts), "labels": list(labels)})
        return super().score_labels(images, context, prompts, labels)


def _body(png_b64, questions, **options):
    return {"model": "vlm", "state": {"images": [{"id": "img0", "base64": png_b64}]}, "questions": questions, "options": options}


def _rate(instructions=QUESTION, criteria=LEVELS):
    return {"type": "score", "instructions": instructions, "criteria": criteria}


# --- the prompts the shipped calibrations were fit on ---------------------------------------------------


@pytest.mark.parametrize("reverse", [False, True])
def test_prompts_are_identical_to_the_lab(reverse):
    assert rating.digits_block(QUESTION, LEVELS, reverse) == sm.digits_block(QUESTION, LEVELS, reverse=reverse)
    assert rating.digits_block(rating.with_zoom(QUESTION), LEVELS, reverse) == sm.digits_block(sm.with_zoom(QUESTION), LEVELS, reverse=reverse)
    assert rating.RATING_PROMPT_VERSION == sm.LAB_PROMPT_VERSION and rating.ZOOM_FACTOR == sm.ZOOM_FACTOR


def test_zoom_crop_and_matrix_fit_are_identical_to_the_lab():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (90, 123, 3), dtype=np.uint8))
    assert rating.zoom_crop(image).tobytes() == sm.zoom_crop(image, position="c").tobytes()
    x, y = rng.normal(size=(80, 8)), rng.integers(0, 4, 80)
    ours, labs = rating.fit_matrix(x, y, 4), sm.fit_matrix_scaling(x, y, 4)
    assert np.allclose(ours["W"], labs["W"]) and np.allclose(ours["b"], labs["b"]) and ours["rescale"] == pytest.approx(labs["rescale"])
    assert np.allclose(rating.apply_matrix(ours, x), sm.apply_fit("ensemble", x, labs))


def test_cv_rescale_is_less_confident_on_tiny_fits_and_changes_no_prediction():
    rng = np.random.default_rng(1)
    y = np.arange(24) % 4
    x = rng.normal(size=(24, 16)) + np.eye(4)[y].repeat(4, axis=1) * 1.5
    train, cv = rating.fit_matrix(x, y, 4, rescale="train"), rating.fit_matrix(x, y, 4, rescale="cv")
    fresh = rng.normal(size=(200, 16))
    assert (rating.apply_matrix(train, fresh).argmax(1) == rating.apply_matrix(cv, fresh).argmax(1)).all()
    assert cv["rescale"] < train["rescale"] and cv["rescale_mode"] == "cv"
    assert rating.apply_matrix(cv, fresh).max(1).mean() < rating.apply_matrix(train, fresh).max(1).mean()


def test_zoom_sentence_names_the_rated_image():
    assert "`photo`" in rating.with_zoom("How blurry is `photo`?", "photo") and "img0" not in rating.with_zoom("x", "photo")


# --- scoring ------------------------------------------------------------------------------------------------


def test_ens4d_is_the_default_on_the_vlm_and_reads_four_members(cfg, png_b64):
    backend = CountingBackend(fixed={"Sharp": 2.0, "Slightly soft": 0.5, "Blurry": -1.0, "Very blurry": -3.0})
    trace = Engine(cfg, backends={"vlm": backend}).decide(_body(png_b64, {"blur": _rate()}))
    qs = trace.scoring.scores["blur"]
    assert qs.method == "ens4d" and [s["member"] for s in qs.statements] == list(rating.MEMBERS["ens4d"])
    # reversed members are flipped back into level order, so all four agree with the fixed level logits
    assert np.allclose(qs.features.reshape(4, 4), [[2.0, 0.5, -1.0, -3.0]] * 4) and np.allclose(qs.z, [2.0, 0.5, -1.0, -3.0])
    assert trace.response.usage.forward_passes == 4
    answer = trace.response.answers["blur"]
    assert answer.method == "ens4d" and answer.calibration is None and max(answer.probabilities, key=answer.probabilities.get) == "0"
    # two calls: the image alone, then the image plus its magnified crop
    assert [c["images"] for c in backend.label_calls] == [["img0"], ["img0", "zoom"]]
    assert all("`zoom` is a 3x pixel-magnified crop" in p for p in backend.label_calls[1]["prompts"])


def test_questions_are_packed_behind_the_same_views(cfg, png_b64):
    backend = CountingBackend()
    questions = {"blur": _rate(), "noise": _rate("How noisy is `img0`?", ["Clean", "Some grain", "Noisy", "Very noisy"]),
                 "exposure": _rate("How dark is `img0`?", ["Fine", "Dim", "Dark"])}
    trace = Engine(cfg, backends={"vlm": backend}).decide(_body(png_b64, questions))
    # one call per (view, number of levels): 4-level questions share calls, the 3-level one gets its own labels
    assert sorted((c["images"], len(c["prompts"]), len(c["labels"])) for c in backend.label_calls) == sorted([
        (["img0"], 4, 4), (["img0", "zoom"], 4, 4), (["img0"], 2, 3), (["img0", "zoom"], 2, 3)])
    assert trace.response.usage.forward_passes == 12 and list(trace.response.answers) == ["blur", "noise", "exposure"]


def test_digits_is_one_pass_and_statements_is_v0(cfg, png_b64):
    backend = CountingBackend()
    engine = Engine(cfg, backends={"vlm": backend})
    one = engine.decide(_body(png_b64, {"blur": _rate()}, score_method="digits"))
    assert one.response.usage.forward_passes == 1 and len(backend.label_calls) == 1 and one.scoring.scores["blur"].features.shape == (4,)
    v0 = engine.decide(_body(png_b64, {"blur": _rate()}, score_method="statements"))
    assert v0.scoring.scores["blur"].method == "statement" and v0.response.answers["blur"].method == "statements"
    assert v0.response.usage.forward_passes == 4 and len(backend.label_calls) == 1  # no new label call


def test_fast2_is_two_passes_on_the_image_alone(cfg, png_b64):
    backend = CountingBackend()
    trace = Engine(cfg, backends={"vlm": backend}).decide(_body(png_b64, {"blur": _rate(), "noise": _rate("How noisy is `img0`?")}, score_method="fast2"))
    assert [c["images"] for c in backend.label_calls] == [["img0"]] and len(backend.label_calls[0]["prompts"]) == 4
    assert trace.response.usage.forward_passes == 4 and trace.scoring.scores["blur"].features.shape == (8,)
    assert trace.response.answers["blur"].method == "fast2"


def test_dual_encoder_keeps_statements_and_rejects_explicit_ens4d(cfg, png_b64):
    backend = CountingBackend()
    backend.kind = "dual_encoder"
    engine = Engine(cfg, backends={"vlm": backend})
    assert engine.decide(_body(png_b64, {"blur": _rate()})).scoring.scores["blur"].method == "statement"
    status, payload = engine.decide_json(_body(png_b64, {"blur": _rate()}, score_method="ens4d"))
    assert status == 400 and payload["code"] == "unsupported_question_for_backend"


def test_ens4d_needs_one_rated_image_and_a_free_zoom_id(cfg, png_b64):
    engine = Engine(cfg, backends={"vlm": CountingBackend()})
    two = _body(png_b64, {"q": _rate("Which is blurrier?")})
    two["state"]["images"].append({"id": "img1", "base64": png_b64})
    status, payload = engine.decide_json(two)
    assert status == 400 and "name exactly one image" in payload["message"]
    two["questions"]["q"] = _rate("How blurry is `img1`?")
    trace = engine.decide(two)
    assert trace.scoring.scores["q"].method == "ens4d"
    two["options"] = {"score_method": "digits"}
    two["questions"]["q"] = _rate("Which is blurrier?")
    assert engine.decide_json(two)[0] == 200
    reserved = _body(png_b64, {"q": _rate("How blurry is `zoom`?")})
    reserved["state"]["images"][0]["id"] = "zoom"
    assert engine.decide_json(reserved)[0] == 400


# --- calibration -----------------------------------------------------------------------------------------------


def _fit_for(engine, backend, question, n=60, seed=0):
    """A calibration that maps the fake features to a fixed wrong-looking level, so its effect is visible."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 4
    x = rng.normal(size=(n, 16)) + np.eye(4)[y].repeat(4, axis=1) * 3
    key = engine.rating_key(backend, "ens4d", parse_request({"model": "vlm", "state": {"images": [{"id": "img0", "path": "x"}]},
                                                             "questions": {"q": question}}).questions["q"])
    return rating.build_calibration(key, x, y, name="test")


def test_rubric_calibration_is_applied_only_to_its_rubric(cfg, png_b64):
    backend = CountingBackend()
    engine = Engine(cfg, backends={"vlm": backend})
    cal = _fit_for(engine, backend, _rate())
    assert cal.cv["accuracy"] > 0.9 and cal.n_per_level == [15, 15, 15, 15] and len(cal.W) == 4 and len(cal.W[0]) == 16
    rating.save_calibration(cfg.path("calibration") / "ratings", cal)

    other = _rate("How noisy is `img0`?", LEVELS)
    status, payload = engine.decide_json(_body(png_b64, {"blur": _rate(), "noise": other}, calibrated="auto"))
    assert status == 200 and payload["calibration_version"] == cal.version
    assert payload["answers"]["blur"]["calibration"] == cal.version and payload["answers"]["noise"]["calibration"] is None
    assert payload["answers"]["blur"]["probabilities"] != payload["answers"]["blur"]["raw"]
    assert payload["answers"]["noise"]["probabilities"] == payload["answers"]["noise"]["raw"]
    assert any("`noise`" in w and "glance fit" in w for w in payload["warnings"]) and not any("`blur`" in w for w in payload["warnings"])

    # calibrated: true keeps the v0 rule: a missing calibration is an error, never a warning
    status, payload = engine.decide_json(_body(png_b64, {"noise": other}, calibrated=True))
    assert status == 409 and payload["code"] == "calibration_mismatch" and "glance fit" in payload["message"]
    assert engine.decide_json(_body(png_b64, {"blur": _rate()}, calibrated=True))[0] == 200
    # calibrated: false never applies one
    assert engine.decide(_body(png_b64, {"blur": _rate()})).response.answers["blur"].calibration is None


def test_rating_key_separates_rubrics_and_configurations(cfg):
    backend = CountingBackend()
    engine = Engine(cfg, backends={"vlm": backend})
    q = parse_request({"model": "vlm", "state": {"images": [{"id": "img0", "path": "x"}]}, "questions": {"q": _rate()}}).questions["q"]
    base = engine.rating_key(backend, "ens4d", q)
    assert base.hash() == engine.rating_key(backend, "ens4d", q).hash()
    assert base.hash() != engine.rating_key(backend, "digits", q).hash()
    assert base.hash() != base.model_copy(update={"criteria": tuple(LEVELS[:3])}).hash()
    assert base.hash() != base.model_copy(update={"image_token_budget": 128}).hash()
    assert base.hash() != base.model_copy(update={"model": "other/model@1"}).hash()


def test_build_calibration_rejects_missing_levels():
    key = rating.RatingKey(backend="vlm", model="m@1", prompt_version="s1", score_method="ens4d", image_token_budget=64,
                           instructions=QUESTION, criteria=tuple(LEVELS))
    with pytest.raises(ValueError, match="every level"):
        rating.build_calibration(key, np.zeros((6, 16)), np.array([0, 0, 1, 1, 2, 2]))


# --- fit and the Python API --------------------------------------------------------------------------------------


def _label_folder(tmp_path, per_level=3):
    for level in range(4):
        folder = tmp_path / "labels" / f"{level}_{LEVELS[level].lower().replace(' ', '_')}"
        folder.mkdir(parents=True)
        for i in range(per_level):
            Image.new("RGB", (48, 36), (60 * level, 20 * i, 90)).save(folder / f"{i}.png")
    return tmp_path / "labels"


def test_unlabeled_calibration_equals_zscore_then_average(cfg, tmp_path):
    from scipy.special import softmax

    from glance.api import Glance

    rng = np.random.default_rng(3)
    key = rating.RatingKey(backend="vlm", model="m@1", prompt_version="s1", score_method="ens4d", image_token_budget=64,
                           instructions=QUESTION, criteria=tuple(LEVELS))
    pool, fresh = rng.normal(2.0, 3.0, size=(40, 16)), rng.normal(2.0, 3.0, size=(5, 16))
    cal = rating.build_unlabeled_calibration(key, pool)
    expected = softmax(((fresh - pool.mean(0)) / (pool.std(0) + 1e-6)).reshape(5, 4, 4).mean(1), axis=1)
    assert cal.kind == "unlabeled_zscore" and cal.n_per_level == [] and np.allclose(rating.apply_matrix(cal, fresh), expected)
    with pytest.raises(ValueError, match="at least 8"):
        rating.build_unlabeled_calibration(key, pool[:5])

    g = Glance(config=cfg)
    g.engine._backends["vlm"] = CountingBackend()
    folder = _label_folder(tmp_path)  # level folders are ignored in unlabeled mode
    cal = g.fit(QUESTION, LEVELS, folder, unlabeled=True, name="no-labels")
    assert cal.kind == "unlabeled_zscore" and cal.n == 12
    answer = g.score(str(next(folder.rglob("*.png"))), QUESTION, LEVELS)
    assert answer["calibration"] == cal.version


def test_read_labels_from_folder_jsonl_and_csv(tmp_path):
    from glance.fit import read_labels

    folder = _label_folder(tmp_path)
    from_folder = read_labels(folder)
    assert len(from_folder) == 12 and sorted({level for _, level in from_folder}) == [0, 1, 2, 3]
    (tmp_path / "l.jsonl").write_text("".join(json.dumps({"image": p, "level": lv}) + "\n" for p, lv in from_folder))
    (tmp_path / "l.csv").write_text("image,level\n" + "".join(f"{p},{lv}\n" for p, lv in from_folder))
    assert read_labels(tmp_path / "l.jsonl") == from_folder == read_labels(tmp_path / "l.csv")


def test_fit_saves_a_calibration_that_later_requests_pick_up(cfg, tmp_path):
    from glance.api import Glance

    g = Glance(config=cfg)
    g.engine._backends["vlm"] = CountingBackend()
    cal = g.fit(QUESTION, LEVELS, _label_folder(tmp_path), name="blur-test")
    assert cal.n == 12 and cal.name == "blur-test" and (cfg.path("calibration") / "ratings" / f"{cal.key.hash()}.json").exists()
    image = str(next((tmp_path / "labels").rglob("*.png")))
    answer = g.score(image, QUESTION, LEVELS)
    assert answer["calibration"] == cal.version and answer["method"] == "ens4d" and answer["warnings"] == []
    assert g.score(image, "A different question about `img0`?", LEVELS)["calibration"] is None
    assert 0.0 <= g.noul(image, "Is `img0` blurry?")["noul"] <= 1.0
    assert g.choice(image, "What is `img0`?", ["dog", "cat"])["choice"] in ("dog", "cat")
    many = g.ask(image, {"a": {"type": "score", "instructions": QUESTION, "criteria": LEVELS},
                         "b": {"type": "noul", "instructions": "Is `img0` blurry?"}})
    assert set(many["answers"]) == {"a", "b"}

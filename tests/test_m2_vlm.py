"""M2 acceptance checks on the real VLM weights. Run with GLANCE_TEST_MODELS=1 (add -s to see the measurements)."""

import json
import random

import numpy as np
import pytest

from glance.pipeline import Engine

from .conftest import needs_models

pytestmark = needs_models

DOC_TYPES = {
    "receipt": "Itemized proof of purchase from a store or restaurant",
    "invoice": "Bill requesting payment, naming payer and payee",
    "other": None,
}
SUBJECTS = {"dog": None, "cat": None, "printed_document": None, "plate_of_food": None, "car": None}
QUESTIONS = {
    "is_receipt": {"type": "noul", "instructions": "Is `img0` a photo or scan of a purchase receipt?"},
    "has_animal": {"type": "noul", "instructions": "Does `img0` show an animal?"},
    "has_text": {"type": "noul", "instructions": "Does `img0` contain printed text?"},
    "doc_type": {"type": "choice", "instructions": "What kind of document is `img0`?", "criteria": DOC_TYPES},
    "subject": {"type": "choice", "instructions": "What is the main subject of `img0`?", "criteria": SUBJECTS},
    "legibility": {"type": "score", "instructions": "How legible is the text in `img0`?",
                   "criteria": ["Text cannot be read", "Some words readable, totals or names unclear",
                                "All text clearly readable"]},
    "text_cover": {"type": "score", "instructions": "How much of `img0` is covered by text?",
                   "criteria": ["No text at all", "A few words", "About half of the image", "Text fills the image"]},
}


def sanity_items() -> list[dict]:
    """20 single-question requests over the three bundled samples."""
    items = []
    for sample in ("receipt", "invoice", "dog"):
        for qid, question in QUESTIONS.items():
            items.append({
                "model": "vlm",
                "state": {"images": [{"id": "img0", "path": f"samples/{sample}.jpg"}]},
                "questions": {qid: question},
            })
    return items[:20]


@pytest.fixture(scope="module")
def engine():
    from glance.config import load_config

    return Engine(load_config(overrides={"vlm": {"prefix_cache": False}}), source="test")


def test_off_mass_on_sanity_set(engine):
    off = []
    for body in sanity_items():
        trace = engine.decide(body)
        for qs in trace.scoring.scores.values():
            off += [s["off_mass"] for s in qs.statements]
    print(f"\n[M2] off_mass over {len(off)} statements in 20 items: mean={np.mean(off):.5f} max={np.max(off):.5f}")
    assert np.mean(off) < 0.1


def test_reordering_options_does_not_move_independent_probabilities(engine):
    rng = random.Random(7)
    worst = 0.0
    for sample in ("receipt", "invoice", "dog"):
        for qid in ("doc_type", "subject"):
            question = QUESTIONS[qid]
            base = engine.decide({"model": "vlm", "state": {"images": [{"id": "img0", "path": f"samples/{sample}.jpg"}]},
                                  "questions": {qid: question}}).response.answers[qid].probabilities
            for _ in range(3):
                keys = list(question["criteria"])
                rng.shuffle(keys)
                shuffled = {**question, "criteria": {k: question["criteria"][k] for k in keys}}
                probs = engine.decide({"model": "vlm", "state": {"images": [{"id": "img0", "path": f"samples/{sample}.jpg"}]},
                                       "questions": {qid: shuffled}}).response.answers[qid].probabilities
                worst = max(worst, max(abs(probs[k] - base[k]) for k in base))
    print(f"\n[M2] max |dp| under option reordering (independent): {worst:.2e}")
    assert worst <= 1e-3


def test_two_identical_runs_agree_on_z(engine):
    worst = 0.0
    for body in sanity_items():
        a = engine.decide(json.loads(json.dumps(body))).scoring.scores
        b = engine.decide(json.loads(json.dumps(body))).scoring.scores
        for qid in a:
            worst = max(worst, float(np.max(np.abs(a[qid].z - b[qid].z))))
    print(f"\n[M2] max |dz| between two identical runs: {worst:.2e}")
    assert worst <= 1e-3


def test_letter_method(engine):
    for sample, expected in (("receipt", "receipt"), ("invoice", "invoice"), ("dog", "other")):
        body = {"model": "vlm", "state": {"images": [{"id": "img0", "path": f"samples/{sample}.jpg"}]},
                "questions": {"doc_type": QUESTIONS["doc_type"]}, "options": {"choice_method": "letter"}}
        trace = engine.decide(body)
        answer = trace.response.answers["doc_type"]
        qs = trace.scoring.scores["doc_type"]
        print(f"\n[M2] letter {sample}: {answer.choice} {answer.probabilities} off_mass="
              f"{max(s['off_mass'] for s in qs.statements):.4f}")
        assert qs.method == "letter" and len(qs.statements) == 3
        assert trace.response.usage.forward_passes == 3
        assert answer.choice == expected


def test_image_token_budget_is_respected(engine):
    backend = engine.backend("vlm")
    for sample in ("receipt", "invoice", "dog"):
        trace = engine.decide({"model": "vlm", "state": {"images": [{"id": "img0", "path": f"samples/{sample}.jpg"}]},
                               "questions": {"q": QUESTIONS["has_text"]}})
        assert 0 < trace.response.usage.image_tokens <= backend.image_token_budget
    two = engine.decide({"model": "vlm",
                         "state": {"images": [{"id": "a", "path": "samples/invoice.jpg"}, {"id": "b", "path": "samples/dog.jpg"}]},
                         "questions": {"q": {"type": "noul", "instructions": "Do `a` and `b` show the same thing?"}}})
    assert two.response.usage.image_tokens <= backend.image_token_budget
    assert two.response.answers["q"].noul < 0.5


def test_manual_tokenization_matches_the_processor(engine):
    from glance.images import load_image
    from glance.schema import ImageRef

    backend = engine.backend("vlm")
    img = load_image(ImageRef(id="img0", path="samples/dog.jpg"), engine.cfg.limits)
    text = backend.render_prompt([img], {"k": "v"}, "Question: Is `img0` a dog?\nAnswer Yes or No.")
    pixel_values, grid, tokens = backend._encode_images([img])
    mine = backend._tokenize([text], tokens)[0]
    size = {"shortest_edge": 64 * backend.pixels_per_token, "longest_edge": backend.image_token_budget * backend.pixels_per_token}
    theirs = backend.processor(text=[text], images=[[img.image]], return_tensors="pt", images_kwargs={"size": size})
    assert mine == theirs["input_ids"][0].tolist()
    assert np.allclose(pixel_values.numpy(), theirs["pixel_values"].numpy())

"""Offline unit tests for `glance.lab.external_systems` and the `tools/external_collect.py` /
`tools/external_report.py` CLIs (lab/NOTES.md entry 37).

Nothing here downloads or loads a real model. OpenJevV2 and QSitMini are exercised with tiny fake tokenizers,
processors and models (plain Python objects plus small real `torch` tensors), the same way `tests/test_m2_vlm.py`
fakes a `Backend` elsewhere in this repo; `.load()` (the method that would actually fetch a real checkpoint) is
never called anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from glance.lab import external_systems as ext
from glance.lab.collect import load_ladder_meta
from glance.logging_utils import JsonlWriter, read_jsonl
from tools import external_collect, external_report

LADDER_META = {
    "instructions": "How blurry is `img0`?",
    "levels": [
        "Sharp: edges are crisp",
        "Slightly soft",
        "Clearly blurred",
        "Heavily blurred",
    ],
}


# === OpenJevV2: claim construction, no downloads ==================================================================


def test_openjev_claim_premise_uses_lead_in_and_img_mark_and_reads_as_prose():
    system = ext.OpenJevV2()
    premise = system.claim_premise(LADDER_META)
    assert premise.startswith(ext.OPENJEV_LEAD_IN)
    assert ext.OPENJEV_IMG_MARK in premise
    assert "`img0`" not in premise  # the markdown image reference is turned into prose
    assert "How blurry is this image?" in premise


def test_openjev_claim_hypotheses_one_per_level_uses_reranks_own_format():
    system = ext.OpenJevV2()
    hyps = system.claim_hypotheses(LADDER_META)
    assert len(hyps) == len(LADDER_META["levels"])
    for level, hyp in zip(LADDER_META["levels"], hyps):
        assert hyp == ext.OPENJEV_HYP_FORMAT.format(level)
        assert level in hyp
    # no cross-contamination: level i's hypothesis does not contain level j's text
    for i, hyp in enumerate(hyps):
        for j, level in enumerate(LADDER_META["levels"]):
            if i != j:
                assert level not in hyp


def test_openjev_vision_block_repeats_image_pad_token_n_times():
    system = ext.OpenJevV2()
    block = system.vision_block(5)
    assert block == "<|vision_start|>" + "<|image_pad|>" * 5 + "<|vision_end|>"
    assert block.count("<|image_pad|>") == 5


def test_openjev_label_token_ids_are_classifier_head_indices():
    system = ext.OpenJevV2()
    assert system.label_token_ids() == {"contradiction": 0, "entailment": 1, "neutral": 2}
    assert (ext.CON, ext.ENT, ext.NEU) == (0, 1, 2)


def test_openjev_features_without_load_raises_instead_of_loading_the_model():
    system = ext.OpenJevV2()
    with pytest.raises(RuntimeError, match="load"):
        system.features("some/path.jpg", LADDER_META)


class _FakeOpenjevTokenizer:
    """Real-shaped output (torch tensors), fake content: no vocabulary, no download."""

    def __init__(self):
        self.padding_side = None
        self.seen_texts: list[list[str]] = []

    def __call__(self, texts, truncation=True, max_length=4096, padding=True, return_tensors="pt"):
        self.seen_texts.append(list(texts))
        n = len(texts)
        return {"input_ids": torch.zeros((n, 3), dtype=torch.long), "attention_mask": torch.ones((n, 3), dtype=torch.long)}


class _FakeImageProcessor:
    merge_size = 2

    def __call__(self, images, return_tensors="pt"):
        n = len(images)
        # t * h * w = 1 * 4 * 6 = 24; 24 // merge_size**2 (4) == 6 image-pad tokens per image
        return {"pixel_values": torch.zeros((n, 3)), "image_grid_thw": torch.tensor([[1, 4, 6]] * n)}


def _tiny_image(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


def test_openjev_build_inputs_computes_image_token_count_and_renders_template(tmp_path):
    system = ext.OpenJevV2()
    system.tokenizer = _FakeOpenjevTokenizer()
    system.image_processor = _FakeImageProcessor()
    img_path = _tiny_image(tmp_path / "x.jpg")

    built = system.build_inputs(img_path, LADDER_META)
    assert built["n_image_tokens"] == 6
    assert len(built["texts"]) == len(LADDER_META["levels"])
    assert built["input_ids"].shape[0] == len(LADDER_META["levels"])
    assert built["pixel_values"].shape[0] == len(LADDER_META["levels"])
    for text in built["texts"]:
        assert text.startswith("Premise: ")
        assert "Hypothesis: The correct answer is:" in text
        assert "<|image_pad|>" * 6 in text
        assert "<|vision_start|>" in text and "<|vision_end|>" in text


class _FakeOpenjevModel:
    IMAGE_TOKEN_ID = 99

    def __init__(self, logits: torch.Tensor):
        self.logits = logits  # (K, 3): [contradiction, entailment, neutral]
        self.config = type("Config", (), {"image_token_id": self.IMAGE_TOKEN_ID})()
        self.seen_mm_token_type_ids = None

    def __call__(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None):
        # the real model refuses multimodal input without this (first real run, lab/NOTES.md entry 37d)
        assert mm_token_type_ids is not None and mm_token_type_ids.shape == input_ids.shape
        assert torch.equal(mm_token_type_ids, (input_ids == self.IMAGE_TOKEN_ID).long())
        self.seen_mm_token_type_ids = mm_token_type_ids
        return type("Out", (), {"logits": self.logits})()


def test_openjev_features_reads_log_odds_of_entailment_vs_everything_else(tmp_path):
    system = ext.OpenJevV2()
    system.tokenizer = _FakeOpenjevTokenizer()
    system.image_processor = _FakeImageProcessor()
    k = len(LADDER_META["levels"])
    logits = torch.zeros((k, 3))
    logits[0] = torch.tensor([-6.0, 6.0, -6.0])  # confidently entailment: this level's claim is "true"
    for i in range(1, k):
        logits[i] = torch.tensor([6.0, -6.0, -6.0])  # confidently contradiction
    system.model = _FakeOpenjevModel(logits)
    system.device = "cpu"

    feats = system.features(_tiny_image(tmp_path / "x.jpg"), LADDER_META)
    assert len(feats) == k
    assert feats[0] == max(feats)
    assert feats[0] > 0 > feats[1]


# === QSitMini: five-word token lookup (incl. multi-token), no downloads ===========================================


class _FakeQsitTokenizer:
    """`tokenizer(["Excellent", ...])` -> plain python lists; `tokenizer(text, return_tensors="pt")` -> a tensor,
    matching what `QSitMini.level_token_ids`/`build_prompt` each expect from a real HF tokenizer."""

    MULTI_TOKEN_WORD = "Excellent"  # deliberately more than one sub-token, to exercise the fallback

    def __call__(self, text, return_tensors=None):
        if isinstance(text, list):
            return {"input_ids": [self._encode(t) for t in text]}
        seq = self._encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([seq], dtype=torch.long)}
        return {"input_ids": [seq]}

    def _encode(self, word: str) -> list[int]:
        if word == self.MULTI_TOKEN_WORD:
            return [501, 502]  # two sub-tokens; the FIRST (501) is what the adapter must use
        if word in ext.QSIT_LEVEL_WORDS:
            return [600 + ext.QSIT_LEVEL_WORDS.index(word)]
        return [42] * max(1, len(word.split()))  # generic filler for the answer-prefix sentence


def test_qsit_level_token_ids_uses_first_subtoken_and_records_the_multitoken_case():
    system = ext.QSitMini()
    system.tokenizer = _FakeQsitTokenizer()
    ids, counts = system.level_token_ids()
    assert ext.QSIT_LEVEL_WORDS[0] == _FakeQsitTokenizer.MULTI_TOKEN_WORD == "Excellent"
    assert counts[0] == 2  # "Excellent" took two sub-tokens in this fake tokenizer
    assert ids[0] == 501  # the FIRST sub-token, not the second (502)
    assert counts[1:] == [1, 1, 1, 1]  # every other word was a single token
    assert ids[1:] == [601, 602, 603, 604]


def test_qsit_expected_degradation_flips_sign_so_higher_means_worse():
    system = ext.QSitMini()
    confidently_excellent = [10.0, 0.0, 0.0, 0.0, 0.0]
    confidently_bad = [0.0, 0.0, 0.0, 0.0, 10.0]
    d_excellent = system.expected_degradation(confidently_excellent)
    d_bad = system.expected_degradation(confidently_bad)
    assert d_bad > d_excellent  # higher = more degraded, matching this lab's `level` direction


class _FakeQsitProcessor:
    def apply_chat_template(self, conversation, add_generation_prompt=True):
        assert conversation[0]["content"][0]["text"] == ext.QSIT_QUESTION
        return "PROMPT"

    def __call__(self, images=None, text=None, return_tensors="pt"):
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}


class _FakeQsitModel:
    def __init__(self, last_position_logits: torch.Tensor):
        self.last_position_logits = last_position_logits  # (vocab,)

    def __call__(self, **kwargs):
        vocab = self.last_position_logits.shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        full = torch.zeros((1, seq_len, vocab))
        full[0, -1] = self.last_position_logits
        return type("Out", (), {"logits": full})()


def test_qsit_features_reads_five_logits_at_the_answer_position(tmp_path):
    system = ext.QSitMini()
    system.tokenizer = _FakeQsitTokenizer()
    system.processor = _FakeQsitProcessor()
    system.level_ids, system.level_token_counts = system.level_token_ids()
    vocab = 700
    last_logits = torch.zeros(vocab)
    for i, tok_id in enumerate(system.level_ids):
        last_logits[tok_id] = float(i)
    system.model = _FakeQsitModel(last_logits)
    system.device = "cpu"

    feats = system.features(_tiny_image(tmp_path / "x.jpg"), LADDER_META)
    assert feats == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_qsit_features_without_load_raises():
    system = ext.QSitMini()
    with pytest.raises(RuntimeError, match="load"):
        system.features("some/path.jpg", LADDER_META)


# === tools/external_collect.py: row schema and resumability =======================================================


class _FakeSystem(ext.ExternalSystem):
    name = "fake"
    model_id = "fake/model"
    revision = "deadbeef"

    def __init__(self, values: dict[str, list[float]]):
        self.values = values
        self.calls: list[str] = []

    def load(self, device, dtype):
        self.calls.append(f"load:{device}:{dtype}")

    def features(self, image_path, ladder_meta):
        self.calls.append(str(image_path))
        return self.values[Path(image_path).stem]


def _fake_items(n: int, ladder: str = "blur") -> list[dict]:
    return [
        {"item_id": f"{ladder}_{i}", "level": i % 4, "split": "calibration" if i % 2 == 0 else "test", "path": f"{ladder}_{i}.jpg"}
        for i in range(n)
    ]


def test_collect_writes_rows_in_the_collectors_schema(tmp_path):
    items = _fake_items(3)
    values = {it["item_id"]: [0.1, 0.2, 0.3, 0.4] for it in items}
    system = _FakeSystem(values)
    out_path = tmp_path / "run.jsonl"
    writer = JsonlWriter(out_path)

    written = external_collect.collect(system, "fake_method", {"blur": items}, {"blur": LADDER_META}, writer, done=set(), log=None)
    assert written == 3

    rows = read_jsonl(out_path)
    assert len(rows) == 3
    required_keys = {"bench", "ladder", "item_id", "split", "level", "method", "method_key", "system", "model", "logits", "latency_ms"}
    for row, item in zip(rows, items):
        assert required_keys <= row.keys()
        assert row["ladder"] == "blur"
        assert row["item_id"] == item["item_id"]
        assert row["level"] == item["level"]
        assert row["split"] == item["split"]
        assert row["method"] == row["method_key"] == "fake_method"
        assert row["system"] == "fake"
        assert row["model"] == "fake/model@deadbeef"
        assert row["logits"] == values[item["item_id"]]
        assert isinstance(row["latency_ms"], float) and row["latency_ms"] >= 0.0


def test_collect_is_resumable_like_glance_lab_collect(tmp_path):
    items = _fake_items(4)
    values = {it["item_id"]: [1.0, 2.0, 3.0, 4.0] for it in items}
    out_path = tmp_path / "run.jsonl"
    writer = JsonlWriter(out_path)

    system1 = _FakeSystem(values)
    external_collect.collect(system1, "fake_method", {"blur": items[:2]}, {"blur": LADDER_META}, writer, done=set(), log=None)
    assert len(read_jsonl(out_path)) == 2

    # a fresh process would derive `done` from what is already on disk, exactly like main() does
    done_on_resume = {(r["ladder"], r["item_id"]) for r in read_jsonl(out_path)}
    system2 = _FakeSystem(values)
    written = external_collect.collect(system2, "fake_method", {"blur": items}, {"blur": LADDER_META}, writer, done_on_resume, log=None)

    assert written == 2  # only the two items not already in the file
    assert len(system2.calls) == 2  # features() was not called again for the first two items
    rows = read_jsonl(out_path)
    assert len(rows) == 4
    assert {r["item_id"] for r in rows} == {it["item_id"] for it in items}


# === tools/external_report.py: calibration, ranking, hypothesis lines, missing-file handling ======================


def _make_openjev_rows(ladder: str, k: int, n_cal: int = 40, n_test: int = 40, seed: int = 0, signal: float = 4.0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cal + n_test):
        level = int(rng.integers(0, k))
        logits = rng.normal(scale=0.5, size=k)
        logits[level] += signal
        rows.append({
            "ladder": ladder, "item_id": f"{ladder}_oj_{i}", "split": "calibration" if i < n_cal else "test",
            "level": level, "logits": [float(v) for v in logits], "latency_ms": 50.0 + i,
        })
    return rows


def _make_qsit_rows(ladder: str, k: int, n_cal: int = 40, n_test: int = 40, seed: int = 1, informative: bool = True) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cal + n_test):
        level = int(rng.integers(0, k))
        word_idx = min(4, round(level / max(k - 1, 1) * 4)) if informative else int(rng.integers(0, 5))
        logits = rng.normal(scale=0.3, size=5)
        logits[word_idx] += 4.0
        rows.append({
            "ladder": ladder, "item_id": f"{ladder}_qs_{i}", "split": "calibration" if i < n_cal else "test",
            "level": level, "logits": [float(v) for v in logits], "latency_ms": 30.0 + i,
        })
    return rows


def test_evaluate_openjev_informative_logits_get_high_calibrated_accuracy():
    ladder_meta = {name: LADDER_META for name in external_report.LADDERS}
    rows = [row for name in external_report.LADDERS for row in _make_openjev_rows(name, k=4)]
    report = external_report.evaluate_openjev(rows, ladder_meta)
    assert report["summary"]["scales"] == len(external_report.LADDERS)
    assert report["summary"]["mean_accuracy"] > 0.8
    assert report["summary"]["mean_accuracy"] >= report["summary"]["mean_uncalibrated_accuracy"] - 1e-9


def test_evaluate_qsit_informative_logits_rank_and_classify_well():
    ladder_meta = {name: LADDER_META for name in external_report.LADDERS}
    rows = [row for name in external_report.LADDERS for row in _make_qsit_rows(name, k=4, informative=True)]
    report = external_report.evaluate_qsit(rows, ladder_meta, flip_sign=True)
    assert report["summary"]["mean_srcc_vs_true_level"] > 0.5
    assert report["summary"]["mean_accuracy"] > 0.5


def test_evaluate_qsit_wrong_sign_hurts_the_ranking():
    ladder_meta = {name: LADDER_META for name in external_report.LADDERS}
    rows = [row for name in external_report.LADDERS for row in _make_qsit_rows(name, k=4, informative=True)]
    correctly_signed = external_report.evaluate_qsit(rows, ladder_meta, flip_sign=True)
    wrongly_signed = external_report.evaluate_qsit(rows, ladder_meta, flip_sign=False)
    assert correctly_signed["summary"]["mean_srcc_vs_true_level"] > wrongly_signed["summary"]["mean_srcc_vs_true_level"]


def test_level0_is_best_matches_this_labs_own_ladders():
    assert external_report.level0_is_best(load_ladder_meta("ladders")) is True


def test_report_main_prints_not_collected_yet_for_missing_run_files(tmp_path):
    out = tmp_path / "external_systems"
    rc = external_report.main([
        "--openjev-in", str(tmp_path / "missing_openjev.jsonl"),
        "--qsit-in", str(tmp_path / "missing_qsit.jsonl"),
        "--out", str(out),
    ])
    assert rc == 0
    md = out.with_suffix(".md").read_text()
    assert md.count("not collected yet") == 2
    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["openjev"] is None and payload["qsit"] is None
    assert payload["h31"] is None and payload["h32"] is None


def test_report_main_renders_hypothesis_lines_with_synthetic_rows(tmp_path):
    openjev_path, qsit_path = tmp_path / "external_openjev.jsonl", tmp_path / "external_qsit.jsonl"
    openjev_writer, qsit_writer = JsonlWriter(openjev_path), JsonlWriter(qsit_path)
    for ladder in external_report.LADDERS:
        for row in _make_openjev_rows(ladder, k=4):
            openjev_writer.write(row)
        for row in _make_qsit_rows(ladder, k=4, informative=(ladder in external_report.QSIT_TRAINED_FOR)):
            qsit_writer.write(row)

    out = tmp_path / "external_systems"
    rc = external_report.main(["--openjev-in", str(openjev_path), "--qsit-in", str(qsit_path), "--out", str(out)])
    assert rc == 0

    md = out.with_suffix(".md").read_text()
    assert "H31:" in md and "SUPPORTED" in md
    assert "H32:" in md and "SUPPORTED" in md
    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["h31"] is not None and "supported" in payload["h31"]
    assert payload["h32"] is not None and "supported" in payload["h32"]
    assert payload["openjev"]["summary"]["scales"] == len(external_report.LADDERS)
    assert payload["qsit"]["summary"]["scales"] == len(external_report.LADDERS)

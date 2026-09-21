"""Offline, fast tests for the six procedural probe suites (lab/NOTES.md entry 50). No network, no models:
`glance.lab.probes` is exercised directly (8 items/suite, redirected to a tmp dir via monkeypatched module
paths) and `glance.evals.suites.probes` is built on top of that tmp output (source `MANIFEST_DIR`/`IMAGES_ROOT`
and the eval-image output dir all redirected), so nothing under the real `.cache/` or `glance/evals/manifests/`
is ever read or written."""

from __future__ import annotations

import math

import pytest

from glance.evals.suites import base
from glance.evals.suites import probes as SP
from glance.lab import probes as P
from glance.logging_utils import read_jsonl

N = 8  # items per suite; small and fast, exercises every code path (including count's largest disc counts)


def _build_lab(tmp_path, monkeypatch, n=N):
    monkeypatch.setattr(P, "IMAGES_ROOT", tmp_path / "images")
    monkeypatch.setattr(P, "EVAL_MANIFEST_DIR", tmp_path / "manifests")
    return P.build(n_per_suite=n, force=True)


def _source_rows(tmp_path, suite: str) -> list[dict]:
    return read_jsonl(tmp_path / "manifests" / f"probes_{suite}_source.jsonl")


@pytest.fixture
def lab_built(tmp_path, monkeypatch):
    """Builds all six suites (8 items each) under tmp_path/{images,manifests}."""
    return _build_lab(tmp_path, monkeypatch)


@pytest.fixture
def suite_built(tmp_path, monkeypatch, cfg, lab_built):
    """`lab_built`, plus `glance.evals.suites.probes` redirected to read from the same tmp manifests/images,
    and its eval-image output redirected to a tmp dir too, so `MODULES[...].build(cfg, n)` is fully offline."""
    monkeypatch.setattr(SP, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(SP, "IMAGES_ROOT", tmp_path / "images")
    monkeypatch.setattr(base, "MANIFEST_DIR", tmp_path / "manifests")
    cfg.paths.eval_images = str(tmp_path / "eval_images")
    return lab_built


# --- determinism -----------------------------------------------------------------------------------------


def test_build_is_deterministic_same_seed_same_sha256(tmp_path, monkeypatch):
    tmp_a = tmp_path / "a"
    tmp_b = tmp_path / "b"
    tmp_a.mkdir()
    tmp_b.mkdir()

    monkeypatch.setattr(P, "IMAGES_ROOT", tmp_a / "images")
    monkeypatch.setattr(P, "EVAL_MANIFEST_DIR", tmp_a / "manifests")
    results_a = P.build(n_per_suite=N, force=True)

    monkeypatch.setattr(P, "IMAGES_ROOT", tmp_b / "images")
    monkeypatch.setattr(P, "EVAL_MANIFEST_DIR", tmp_b / "manifests")
    results_b = P.build(n_per_suite=N, force=True)

    assert set(results_a) == set(results_b) == set(P.SUITES)
    for suite in P.SUITES:
        rows_a = {r["item_id"]: r for r in results_a[suite]}
        rows_b = {r["item_id"]: r for r in results_b[suite]}
        assert set(rows_a) == set(rows_b), suite
        for item_id, row_a in rows_a.items():
            row_b = rows_b[item_id]
            assert row_a["sha256"] == row_b["sha256"], f"{suite}/{item_id} sha256 differs across identical builds"
            assert row_a["label"] == row_b["label"], f"{suite}/{item_id} label differs across identical builds"
            assert row_a["params"] == row_b["params"], f"{suite}/{item_id} params differ across identical builds"


# --- `count`: non-overlap + border margin, from the stored centres/radii ---------------------------------


def test_count_discs_never_overlap_or_touch_the_border(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "count")
    assert len(rows) == N
    for row in rows:
        p = row["params"]
        centers, radii = p["centers"], p["radii"]
        assert len(centers) == len(radii) == p["count"] == int(row["label"])
        for (cx, cy), r in zip(centers, radii):
            assert cx - r >= -1e-6 and cx + r <= P.WIDTH + 1e-6
            assert cy - r >= -1e-6 and cy + r <= P.HEIGHT + 1e-6
            assert cx - r >= P.DISC_BORDER_MARGIN - 1e-6, "disc closer than the border margin"
            assert cx + r <= P.WIDTH - P.DISC_BORDER_MARGIN + 1e-6
            assert cy - r >= P.DISC_BORDER_MARGIN - 1e-6
            assert cy + r <= P.HEIGHT - P.DISC_BORDER_MARGIN + 1e-6
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                (x1, y1), (x2, y2) = centers[i], centers[j]
                dist = math.hypot(x1 - x2, y1 - y2)
                assert dist >= radii[i] + radii[j] + P.DISC_MIN_GAP - 1e-6, f"{row['item_id']}: discs {i},{j} too close"


def test_count_labels_are_balanced(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "count")
    labels = sorted(int(r["label"]) for r in rows)
    assert labels == list(range(1, 9))  # N=8 -> exactly one of each count 1..8


# --- `count_color`: red count matches params + label, discs also non-overlapping --------------------------


def test_count_color_red_count_matches_params_and_label(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "count_color")
    assert len(rows) == N
    for row in rows:
        p = row["params"]
        colors = p["colors"]
        assert 3 <= p["total"] <= 9
        assert 0 <= p["red_count"] <= 6
        assert p["total"] == len(colors) == len(p["centers"]) == len(p["radii"])
        assert sum(1 for c in colors if c == "red") == p["red_count"] == int(row["label"])
        assert "orange" not in colors and "pink" not in colors  # spec: keep count_color's reds unmistakable


def test_count_color_discs_never_overlap(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "count_color")
    for row in rows:
        p = row["params"]
        centers, radii = p["centers"], p["radii"]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                (x1, y1), (x2, y2) = centers[i], centers[j]
                dist = math.hypot(x1 - x2, y1 - y2)
                assert dist >= radii[i] + radii[j] + P.DISC_MIN_GAP - 1e-6


def test_count_color_red_counts_are_balanced(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "count_color")
    reds = sorted(int(r["label"]) for r in rows)
    assert set(reds) <= set(range(7))
    assert len(reds) == N
    # N=8 over 7 buckets (0..6): every bucket appears, one appears twice
    assert set(reds) == set(range(7))


# --- `spatial`: yes/no labels consistent with the stored geometry, offsets >= 40px -------------------------


def test_spatial_labels_consistent_with_geometry_and_min_offset(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "spatial")
    assert len(rows) == N  # N images -> N items (N even, one axis/question per row here since N == n_per_suite)
    by_axis = {"horizontal": 0, "vertical": 0}
    for row in rows:
        p = row["params"]
        dx, dy = p["disc_center"]
        sx, sy = p["square_center"]
        assert abs(dx - sx) >= P.SPATIAL_MIN_OFFSET - 1e-6
        assert abs(dy - sy) >= P.SPATIAL_MIN_OFFSET - 1e-6
        by_axis[p["axis"]] += 1
        if p["axis"] == "horizontal":
            assert p["asked_word"] in ("left", "right")
            actual = (dx < sx) if p["asked_word"] == "left" else (dx > sx)
        else:
            assert p["asked_word"] in ("above", "below")
            actual = (dy < sy) if p["asked_word"] == "above" else (dy > sy)
        assert actual == row["label"], f"{row['item_id']}: label doesn't match the stored geometry"
        assert p["question"] == row["params"]["question"]
    assert by_axis["horizontal"] == by_axis["vertical"] == N // 2  # one h + one v item per image


def test_spatial_question_wording_matches_axis_and_word(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "spatial")
    for row in rows:
        p = row["params"]
        if p["axis"] == "horizontal":
            assert p["question"] == f"Is the red ball to the {p['asked_word']} of the blue square in `img0`?"
        else:
            assert p["question"] == f"Is the red ball {p['asked_word']} the blue square in `img0`?"


# --- `largest`: label is exactly the shape with the larger area/reach in params ---------------------------


def test_largest_label_is_the_shape_with_the_larger_area(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "largest")
    assert len(rows) == N
    for row in rows:
        p = row["params"]
        shapes = p["shapes"]
        assert len({s["shape"] for s in shapes}) == 1 and shapes[0]["shape"] in P.SHAPE_NAMES  # one kind per image: "largest" is unambiguous
        assert {s["color"] for s in shapes} == set(P.LARGEST_COLORS)
        areas = sorted((s["area"] for s in shapes), reverse=True)
        assert areas[0] > areas[1] + 1e-6, "the largest shape's area must be unambiguously bigger"
        biggest = max(shapes, key=lambda s: s["area"])
        assert biggest["color"] == row["label"] == p["largest_color"]
        assert biggest["shape"] == p["largest_shape"]
        # area ratio r is exact on the stored (nominal) areas
        base_area = min(s["area"] for s in shapes)
        assert biggest["area"] == pytest.approx(base_area * p["ratio"], rel=1e-6)


def test_largest_ratios_are_balanced(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "largest")
    ratios = sorted(r["params"]["ratio"] for r in rows)
    assert ratios == sorted(list(P.RATIOS) * 2)  # N=8 over 4 ratios -> exactly 2 of each


# --- `stripes`: label balance ------------------------------------------------------------------------------


def test_stripes_orientations_are_balanced(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "stripes")
    orientations = sorted(r["label"] for r in rows)
    assert orientations == sorted(list(P.ORIENTATIONS) * 2)  # N=8 over 4 orientations -> exactly 2 of each


# --- `text`: labels come from their own group, groups sane ---------------------------------------------------


def test_text_label_is_in_its_own_group(tmp_path, lab_built):
    rows = _source_rows(tmp_path, "text")
    assert len(rows) == N
    for row in rows:
        p = row["params"]
        assert row["label"] == p["word"]
        assert p["word"] in p["group"]
        assert len(p["group"]) == 6
        assert p["group"] == P.WORD_GROUPS[p["group_index"]]


# --- suites build EvalItems with the exact registered question wording -------------------------------------


def test_probe_count_question_and_meta(suite_built, cfg):
    items = SP.MODULES["probe_count"].build(cfg, N)
    assert len(items) == N
    for item in items:
        assert item.question == {
            "type": "choice", "instructions": "How many balls are in `img0`?",
            "criteria": {str(i): None for i in range(1, 9)},
        }
        assert item.label == str(item.meta["count"])


def test_probe_count_color_question_and_meta(suite_built, cfg):
    items = SP.MODULES["probe_count_color"].build(cfg, N)
    assert len(items) == N
    for item in items:
        assert item.question == {
            "type": "choice", "instructions": "How many RED balls are in `img0`?",
            "criteria": {str(i): None for i in range(0, 7)},
        }
        assert item.label == str(item.meta["red_count"])


def test_probe_spatial_question_and_meta(suite_built, cfg):
    items = SP.MODULES["probe_spatial"].build(cfg, N)
    assert len(items) == N
    for item in items:
        assert item.question["type"] == "noul"
        assert item.question["instructions"] == item.meta["question"]
        assert isinstance(item.label, bool)


def test_probe_largest_question_and_meta(suite_built, cfg):
    items = SP.MODULES["probe_largest"].build(cfg, N)
    assert len(items) == N
    for item in items:
        assert item.question == {
            "type": "choice", "instructions": "Which shape in `img0` is the largest?",
            "criteria": {
                "red": "The red shape", "blue": "The blue shape", "green": "The green shape", "yellow": "The yellow shape",
            },
        }
        assert item.label == item.meta["largest_color"]


def test_probe_stripes_question_and_meta(suite_built, cfg):
    items = SP.MODULES["probe_stripes"].build(cfg, N)
    assert len(items) == N
    for item in items:
        assert item.question == {
            "type": "choice", "instructions": "Which way do the stripes in `img0` run?",
            "criteria": {
                "horizontal": "Horizontal stripes", "vertical": "Vertical stripes",
                "diagonal_rising": "Diagonal stripes rising to the right", "diagonal_falling": "Diagonal stripes falling to the right",
            },
        }
        assert item.label == item.meta["orientation"]


def test_probe_text_question_and_meta(suite_built, cfg):
    items = SP.MODULES["probe_text"].build(cfg, N)
    assert len(items) == N
    for item in items:
        group = item.meta["group"]
        assert item.question == {
            "type": "choice", "instructions": "Which word is printed in `img0`?",
            "criteria": {w: None for w in group},
        }
        assert item.label == item.meta["word"]
        assert item.label in item.question["criteria"]


# --- suite info / skip behaviour -----------------------------------------------------------------------------


def test_suite_info_backends_and_source():
    for name in ("probe_count", "probe_count_color", "probe_largest", "probe_stripes", "probe_text"):
        info = SP.MODULES[name].INFO
        assert info.backends == ("siglip", "vlm", "frontier")
        assert "entry 50" in info.source
    assert SP.MODULES["probe_spatial"].INFO.backends == ("vlm", "frontier")


def test_probe_count_skipped_without_manifest(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(SP, "MANIFEST_DIR", tmp_path / "missing_manifests")
    monkeypatch.setattr(SP, "IMAGES_ROOT", tmp_path / "missing_images")
    with pytest.raises(base.SuiteSkipped, match="probes.build"):
        SP.MODULES["probe_count"].build(cfg, 4)

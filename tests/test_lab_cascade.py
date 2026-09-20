"""Tests for glance.lab.cascade: all synthetic rows, written to a tmp run directory.

No real predictions.jsonl is touched here; every fixture builds its own rows so the numbers are
hand-checkable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glance.lab import cascade


def _local_row(item_id: str, suite: str, split: str, p: float, label_index: int, backend: str = "vlm") -> dict[str, Any]:
    return {
        "suite": suite, "item_id": item_id, "split": split, "backend": backend, "method": "statement",
        "type": "noul", "keys": ["true"], "label_index": label_index, "raw": p, "calibrated": p,
    }


def _frontier_row(item_id: str, suite: str, correct: bool) -> dict[str, Any]:
    return {
        "suite": suite, "item_id": item_id, "split": "test", "backend": "frontier", "method": "pick",
        "type": "noul", "keys": ["true"], "label_index": 1, "correct": correct,
    }


def _write_predictions(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# --- a hand-computable 10-item cascade ------------------------------------------------------------
#
# Rank (by confidence, descending) 1..10 -> item i01..i10, confidence (11-r)/10 = 1.0, 0.9, ..., 0.1.
# Local model is correct on ranks 1-6, wrong on ranks 7-10 (local accuracy 6/10 = 0.6).
# Frontier is correct on ranks 1-8, wrong on ranks 9-10 (frontier accuracy 8/10 = 0.8).
#
# combined_correct(k) = cum_local[k] + (total_frontier - cum_frontier[k])
#   k=0  (c=0.0): 0 + (8 - 0) = 8  -> combined accuracy 0.8 == frontier alone
#   k=5  (c=0.5): 5 + (8 - 5) = 8  -> combined accuracy 0.8
#   k=10 (c=1.0): 6 + (8 - 8) = 6  -> combined accuracy 0.6 == local alone


def _ten_item_rows(suite: str = "synth") -> list[dict[str, Any]]:
    rows = []
    for r in range(1, 11):
        item_id = f"i{r:02d}"
        conf = (11 - r) / 10
        locally_correct = r <= 6
        p = 0.5 + conf / 2 if locally_correct else 0.5 - conf / 2  # label_index is 1 (true) for every item
        rows.append(_local_row(item_id, suite, "test", p, label_index=1))
        rows.append(_frontier_row(item_id, suite, correct=(r <= 8)))
    return rows


def test_hand_computable_cascade_curve(tmp_path):
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, _ten_item_rows())

    result = cascade.analyze_run(run_dir)
    unit = result["suites"]["synth"]["vlm"]
    assert unit["n_joined"] == 10
    assert unit["local_accuracy"] == pytest.approx(0.6)
    assert unit["frontier_accuracy"] == pytest.approx(0.8)

    curve = {p["coverage"]: p["combined_accuracy"] for p in unit["curve"]}
    assert curve[0.0] == pytest.approx(0.8)
    assert curve[0.5] == pytest.approx(0.8)
    assert curve[1.0] == pytest.approx(0.6)


def test_combined_accuracy_at_endpoints_matches_local_and_frontier_alone(tmp_path):
    """c = 0 must equal frontier-alone accuracy; c = 1 must equal local-alone accuracy."""
    run_dir = tmp_path / "run"
    _write_predictions(run_dir, _ten_item_rows())

    result = cascade.analyze_run(run_dir)
    unit = result["suites"]["synth"]["vlm"]
    curve = {p["coverage"]: p["combined_accuracy"] for p in unit["curve"]}
    assert curve[0.0] == pytest.approx(unit["frontier_accuracy"])
    assert curve[1.0] == pytest.approx(unit["local_accuracy"])

    # oracle upper bound: rank 9 and 10 are wrong for both local and frontier -> 8/10 right somewhere
    assert unit["oracle_accuracy"] == pytest.approx(0.8)


# --- threshold selection is calibration-only --------------------------------------------------------


def _cal_rows(suite: str = "synth") -> dict[str, dict[str, Any]]:
    """20 calibration items: confidence 0.05 .. 1.0, predicted "true" throughout, and labelled true
    iff confidence >= 0.5 -- so "correct" flips exactly at confidence 0.5 (11/20 correct overall,
    but accuracy among confidence >= 0.5 is 11/11 = 1.0)."""
    rows = {}
    for i in range(1, 21):
        conf = i / 20
        p = 0.5 + conf / 2  # predicted 1 always; correct iff label_index == 1
        label_index = 1 if conf >= 0.5 else 0  # so "correct" flips exactly at confidence 0.5
        item_id = f"c{i:02d}"
        rows[item_id] = _local_row(item_id, suite, "calibration", p, label_index=label_index)
    return rows


def test_threshold_uses_only_calibration_rows(tmp_path):
    suite = "synth"
    cal_rows = _cal_rows(suite)
    test_rows = {r["item_id"]: r for r in [_local_row(f"i{r:02d}", suite, "test", 0.9, label_index=1) for r in range(1, 6)]}
    frontier_rows = {iid: _frontier_row(iid, suite, correct=True) for iid in test_rows}

    unit_a = cascade.cascade_unit(test_rows, cal_rows, frontier_rows)
    tau_a = unit_a["operating_points"]["0.95"]["tau"]
    assert tau_a is not None

    # Now flip every test label and every frontier `correct` value: the calibration-derived tau must
    # not move, because find_tau never looks at test_rows or frontier_rows.
    flipped_test_rows = {
        iid: {**row, "label_index": 1 - row["label_index"]} for iid, row in test_rows.items()
    }
    flipped_frontier_rows = {iid: _frontier_row(iid, suite, correct=False) for iid in test_rows}
    unit_b = cascade.cascade_unit(flipped_test_rows, cal_rows, flipped_frontier_rows)
    tau_b = unit_b["operating_points"]["0.95"]["tau"]

    assert tau_b == tau_a
    # sanity: the flip actually changed the downstream numbers, so this isn't a vacuous check
    assert unit_a["local_accuracy"] != unit_b["local_accuracy"] or unit_a["frontier_accuracy"] != unit_b["frontier_accuracy"]


def test_find_tau_lowest_threshold_and_null_when_unreachable():
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    corrects = [False, False, True, True, True]
    # confidence >= 0.5 -> 3/3 correct = 1.0 >= 0.95; confidence >= 0.3 -> 3/4 = 0.75 < 0.95.
    assert cascade.find_tau(confidences, corrects, 0.95) == pytest.approx(0.5)
    # Nothing reaches perfect accuracy at any coverage above the single most-confident item alone,
    # and even that subset alone doesn't get us past 1.0 requirement's impossible neighbor: try > 1.0.
    assert cascade.find_tau(confidences, corrects, 1.01) is None
    assert cascade.find_tau([], [], 0.9) is None


# --- tolerant reading and graceful skips -------------------------------------------------------------


def test_trailing_truncated_line_is_ignored(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    rows = _ten_item_rows()
    lines = [json.dumps(r) for r in rows]
    # A partial write: a real JSON object cut off mid-line, no trailing newline.
    partial = '{"suite": "synth", "item_id": "i11", "split": "test", "backend": "vlm", "raw": 0.'
    (run_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n" + partial)

    loaded = cascade.read_predictions_tolerant(run_dir / "predictions.jsonl")
    assert len(loaded) == len(rows)  # the truncated tail line contributed nothing

    # And the full pipeline must not raise either.
    result = cascade.analyze_run(run_dir)
    assert result["suites"]["synth"]["vlm"]["n_joined"] == 10


def test_suite_without_frontier_rows_is_skipped_without_error(tmp_path):
    run_dir = tmp_path / "run"
    rows = [
        _local_row("a1", "no_frontier_suite", "test", 0.8, label_index=1),
        _local_row("a1", "no_frontier_suite", "calibration", 0.8, label_index=1),
    ]
    _write_predictions(run_dir, rows)

    result = cascade.analyze_run(run_dir)
    unit = result["suites"]["no_frontier_suite"]["vlm"]
    assert unit["n_joined"] == 0
    # No frontier-derived fields should have been computed (and none of this should have raised).
    assert "local_accuracy" not in unit
    assert result["macro_vlm"] is None  # the only suite present has no joined items

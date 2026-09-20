"""Unit tests for glance.lab.score_methods and glance.lab.analyze.

Everything here is synthetic numpy data with fixed seeds: no model, no network, no torch/transformers.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.special import expit, softmax

from glance.lab import analyze as an
from glance.lab import score_methods as sm

# --- prompt builders ------------------------------------------------------------------------------

LEVELS = ["tiny", "moderate", "large", "huge"]  # no digits anywhere, no substring of one in another
INSTRUCTIONS = "Rate how blurry the image is"


def test_independent_statements_one_per_level_no_cross_contamination():
    stmts = sm.independent_statements(INSTRUCTIONS, LEVELS)
    assert len(stmts) == len(LEVELS)
    for i, (level, stmt) in enumerate(zip(LEVELS, stmts)):
        assert stmt.candidate == level
        assert level in stmt.text
        # none of the other levels' text leaks into this statement
        for j, other in enumerate(LEVELS):
            if j != i:
                assert other not in stmt.text
        # the scale is never shown for this method: no digits (step numbers) at all
        assert re.search(r"\d", stmt.text) is None


def test_cumulative_statements_shape_and_content():
    K = len(LEVELS)
    stmts = sm.cumulative_statements(INSTRUCTIONS, LEVELS)
    assert len(stmts) == K - 1
    for idx, stmt in enumerate(stmts):
        step = idx + 2  # steps k = 2..K
        # every level appears, numbered from 1
        for i, level in enumerate(LEVELS):
            assert f"{i + 1}. {level}" in stmt.text
        assert f"step {step} or higher" in stmt.text
        assert stmt.candidate == f">={idx + 1}"


def test_digits_block_numbers_from_zero():
    K = len(LEVELS)
    text, labels = sm.digits_block(INSTRUCTIONS, LEVELS)
    for i, level in enumerate(LEVELS):
        assert f"{i}. {level}" in text
    assert labels == [str(i) for i in range(K)]


def test_with_anchors_mentions_ids_and_correct_step_numbers():
    anchors = [sm.Anchor(image_id="ref1", level=0, path="a.png"), sm.Anchor(image_id="ref2", level=2, path="b.png")]
    for start in (0, 1):
        text = sm.with_anchors(INSTRUCTIONS, LEVELS, anchors, start)
        for a in anchors:
            assert f"`{a.image_id}`" in text
            assert f"step {a.level + start}" in text


# --- dist_from_cumulative --------------------------------------------------------------------------


def test_dist_from_cumulative_rows_sum_to_one():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(50, 3))
    p = sm.dist_from_cumulative(c)
    assert np.allclose(p.sum(axis=-1), 1.0)


def _clean_cumulative_logits(level: int, K: int, mag: float = 20.0) -> np.ndarray:
    """Cumulative threshold logits that unambiguously encode `level` for a K-level scale."""
    return np.array([mag if level >= k + 1 else -mag for k in range(K - 1)])


def test_dist_from_cumulative_clean_logits_argmax_every_level():
    K = 4
    for level in range(K):
        c = _clean_cumulative_logits(level, K)
        p = sm.dist_from_cumulative(c)
        assert p.sum() == pytest.approx(1.0, abs=1e-6)
        assert int(np.argmax(p)) == level


def test_dist_from_cumulative_nonmonotone_input_still_valid_distribution():
    c = np.array([5.0, -5.0, 5.0])
    p = sm.dist_from_cumulative(c)
    assert np.all(p >= 0.0)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_dist_from_cumulative_matches_explicit_formula_when_monotone():
    c = np.array([2.0, 0.0, -2.0])  # already monotone: s(c) decreasing, so accumulate is a no-op
    p = sm.dist_from_cumulative(c, a=1.0, b=0.0)
    s = expit(c)
    expected = np.array([1 - s[0], s[0] - s[1], s[1] - s[2], s[2]])
    assert np.allclose(p, expected)


# --- dist_from_level_logits -------------------------------------------------------------------------


def test_dist_from_level_logits_is_softmax():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(10, 4))
    assert np.allclose(sm.dist_from_level_logits(z), softmax(z, axis=-1))


def test_dist_from_level_logits_bias_shifts_argmax():
    z = np.zeros((1, 3))
    assert int(sm.dist_from_level_logits(z).argmax()) == 0  # tie -> first index
    biased = sm.dist_from_level_logits(z, bias=np.array([0.0, 0.0, 5.0]))
    assert int(biased.argmax()) == 2


def test_dist_from_level_logits_higher_temperature_flattens():
    z = np.array([[0.0, 1.0, 2.0, 3.0]])
    p_cold = sm.dist_from_level_logits(z, temperature=1.0)
    p_hot = sm.dist_from_level_logits(z, temperature=10.0)
    assert p_hot.max() < p_cold.max()
    assert np.std(p_hot) < np.std(p_cold)


# --- fits on synthetic data --------------------------------------------------------------------------


def _make_level_logits(rng, n: int, K: int, signal: float = 3.0, noise: float = 0.7):
    """Synthetic 'well-calibrated at T=1' level logits: y ~ softmax(z), z has a boosted true-label entry."""
    y = rng.integers(0, K, size=n)
    z = rng.normal(scale=noise, size=(n, K))
    z[np.arange(n), y] += signal
    return z, y


def test_fit_temperature_recovers_known_temperature():
    rng = np.random.default_rng(2)
    n, K, true_T = 3000, 4, 2.0
    z_true, _ = _make_level_logits(rng, n, K)
    p_true = softmax(z_true, axis=1)
    y = np.array([rng.choice(K, p=p_true[i]) for i in range(n)])
    z_obs = z_true * true_T  # observed logits are "too sharp" by a factor of true_T
    fit = sm.fit_temperature(z_obs, y)
    assert fit["kind"] == "T"
    assert abs(fit["T"] - true_T) / true_T < 0.15


def test_fit_vector_scaling_repairs_systematic_offset():
    rng = np.random.default_rng(3)
    n, n_test, K = 3000, 1000, 4
    true_bias = np.array([0.0, -2.0, -2.5, -3.0])

    z_fit, y_fit = _make_level_logits(rng, n, K)
    z_test, y_test = _make_level_logits(rng, n_test, K)
    z_obs_fit = z_fit + true_bias
    z_obs_test = z_test + true_bias

    fit = sm.fit_vector_scaling(z_obs_fit, y_fit)
    assert fit["kind"] == "bias+T"
    assert fit["bias"][0] == 0.0  # first level pinned

    acc_before = float(np.mean(z_obs_test.argmax(axis=1) == y_test))
    p_after = sm.dist_from_level_logits(z_obs_test, np.array(fit["bias"]), fit["T"])
    acc_after = float(np.mean(p_after.argmax(axis=1) == y_test))
    assert acc_after > acc_before + 0.2
    assert acc_after > 0.75


def _make_cumulative_logits(rng, n: int, K: int, signal: float = 3.0, noise: float = 0.7):
    y = rng.integers(0, K, size=n)
    c = np.zeros((n, K - 1))
    for k in range(K - 1):
        c[:, k] = np.where(y >= k + 1, signal, -signal) + rng.normal(scale=noise, size=n)
    return c, y


def test_fit_threshold_platt_repairs_bias_toward_no():
    rng = np.random.default_rng(4)
    n, n_test, K = 3000, 1000, 4
    bias_toward_no = -4.0  # every threshold is pushed toward "no", i.e. toward lower predicted levels

    c_fit, y_fit = _make_cumulative_logits(rng, n, K)
    c_test, y_test = _make_cumulative_logits(rng, n_test, K)
    c_obs_fit = c_fit + bias_toward_no
    c_obs_test = c_test + bias_toward_no

    fit = sm.fit_threshold_platt(c_obs_fit, y_fit)
    assert fit["kind"] == "platt/threshold"

    acc_before = float(np.mean(sm.dist_from_cumulative(c_obs_test).argmax(axis=1) == y_test))
    p_after = sm.dist_from_cumulative(c_obs_test, np.array(fit["a"]), np.array(fit["b"]))
    acc_after = float(np.mean(p_after.argmax(axis=1) == y_test))
    assert acc_after > acc_before + 0.2
    assert acc_after > 0.75


# --- candidate_fits / fit_name -----------------------------------------------------------------------


def test_candidate_fits_level_logit_methods():
    rng = np.random.default_rng(5)
    K = 3
    z, y = _make_level_logits(rng, 60, K)
    for method in ("independent", "digits", "anchors_digits"):
        fits = sm.candidate_fits(method, z, y)
        assert [sm.fit_name(f) for f in fits] == ["raw", "T", "bias+T"]
        assert fits[0] is None
        assert fits[1]["kind"] == "T"
        assert fits[2]["kind"] == "bias+T"


def test_candidate_fits_cumulative_methods():
    rng = np.random.default_rng(6)
    K = 3
    c, y = _make_cumulative_logits(rng, 60, K)
    for method in ("cumulative", "anchors_cumulative"):
        fits = sm.candidate_fits(method, c, y)
        assert [sm.fit_name(f) for f in fits] == ["raw", "platt/threshold"]
        assert fits[0] is None
        assert fits[1]["kind"] == "platt/threshold"


def test_fit_name_raw():
    assert sm.fit_name(None) == "raw"


# --- analyze.metrics --------------------------------------------------------------------------------


def test_metrics_hand_computable():
    p = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
    ])
    y = np.array([0, 2, 2])
    m = an.metrics(p, y)
    assert m["n"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["within_1"] == pytest.approx(2 / 3)  # row 2: |pred=0 - y=2| = 2 > 1
    expected_mae = np.mean([abs(0.4 - 0), abs(1.7 - 2), abs(0.5 - 2)])
    assert m["mae"] == pytest.approx(expected_mae)


def test_metrics_spearman_perfect_order():
    K = 5
    y = np.arange(K)
    p = np.eye(K)  # perfect, one-hot predictions -> expected score == y exactly
    m = an.metrics(p, y)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)


# --- analyze.analyze / render -------------------------------------------------------------------------


def _rows_for_group(rng, ladder: str, method: str, method_key: str, K: int, n_cal: int, n_test: int) -> list[dict]:
    if "cumulative" in method:
        z_cal, y_cal = _make_cumulative_logits(rng, n_cal, K)
        z_test, y_test = _make_cumulative_logits(rng, n_test, K)
    else:
        z_cal, y_cal = _make_level_logits(rng, n_cal, K)
        z_test, y_test = _make_level_logits(rng, n_test, K)
    rows = []
    for i in range(n_cal):
        rows.append(_row(ladder, method, method_key, f"cal{i}", "calibration", int(y_cal[i]), z_cal[i]))
    for i in range(n_test):
        rows.append(_row(ladder, method, method_key, f"test{i}", "test", int(y_test[i]), z_test[i]))
    return rows


def _row(ladder, method, method_key, item_id, split, level, logits) -> dict:
    return {
        "ladder": ladder, "item_id": item_id, "split": split, "level": level, "method": method,
        "anchors": None, "method_key": method_key, "logits": [float(v) for v in logits],
        "latency_ms": 10.0, "image_tokens": 100.0, "forward_passes": 1, "off_mass_max": 0.01,
    }


def test_analyze_dev_false_chosen_and_fit_depend_only_on_calibration_split():
    rng = np.random.default_rng(7)
    rows = _rows_for_group(rng, "blur", "digits", "digits", K=4, n_cal=200, n_test=60)
    baseline = an.analyze(rows, dev=False)
    key = "blur|digits"
    assert key in baseline
    base_chosen = baseline[key]["chosen"]
    base_fit = baseline[key]["variants"][base_chosen]["fit"]

    rng2 = np.random.default_rng(999)
    corrupted = []
    for r in rows:
        r2 = dict(r)
        if r2["split"] == "test":
            r2["logits"] = [float(v) for v in rng2.normal(scale=50.0, size=len(r2["logits"]))]
            r2["level"] = int(rng2.integers(0, 4))
        corrupted.append(r2)

    result = an.analyze(corrupted, dev=False)
    assert key in result
    assert result[key]["chosen"] == base_chosen
    assert result[key]["variants"][base_chosen]["fit"] == base_fit


def test_analyze_dev_true_ignores_test_split_entirely():
    rng = np.random.default_rng(8)
    rows = _rows_for_group(rng, "noise", "independent", "independent", K=3, n_cal=200, n_test=60)
    with_test = an.analyze(rows, dev=True)
    without_test = an.analyze([r for r in rows if r["split"] != "test"], dev=True)
    assert with_test == without_test
    assert "noise|independent" in with_test


def test_analyze_skips_groups_with_too_few_rows():
    rng = np.random.default_rng(9)
    rows = _rows_for_group(rng, "jpeg", "independent", "independent", K=3, n_cal=5, n_test=5)
    result = an.analyze(rows, dev=False)
    assert "jpeg|independent" not in result
    assert result == {}


def test_render_contains_ladders_and_methods_and_handles_cumulative():
    rng = np.random.default_rng(10)
    rows = []
    rows += _rows_for_group(rng, "blur", "digits", "digits", K=4, n_cal=200, n_test=60)
    rows += _rows_for_group(rng, "blur", "cumulative", "cumulative", K=4, n_cal=200, n_test=60)
    rows += _rows_for_group(rng, "noise", "digits", "digits", K=4, n_cal=200, n_test=60)
    results = an.analyze(rows, dev=False)
    assert set(results) == {"blur|digits", "blur|cumulative", "noise|digits"}
    text = an.render(results, rows, dev=False, title="Test report")
    assert "blur" in text
    assert "noise" in text
    assert "digits" in text
    assert "cumulative" in text

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from glance.lab import ladders as L


def _texture_image(size: int = 96, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.mark.parametrize("ladder", list(L.LADDERS))
def test_degrade_is_deterministic_and_preserves_size_and_mode(ladder):
    base = _texture_image()
    for level in range(4):
        a = L.degrade(base, ladder, level, seed=123)
        b = L.degrade(base, ladder, level, seed=123)
        assert a.size == base.size
        assert b.size == base.size
        assert a.mode == "RGB" and b.mode == "RGB"
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_blur_and_resolution_level0_equal_base_exactly():
    base = _texture_image(seed=1)
    blur0 = L.degrade(base, "blur", 0, seed=1)
    res0 = L.degrade(base, "resolution", 0, seed=1)
    assert np.array_equal(np.asarray(blur0), np.asarray(base))
    assert np.array_equal(np.asarray(res0), np.asarray(base))


def test_exposure_level3_is_about_point_one_times_level0_luminance():
    base = _texture_image(seed=2)
    level0 = L.degrade(base, "exposure", 0, seed=5)
    level3 = L.degrade(base, "exposure", 3, seed=5)
    m0 = np.asarray(level0, dtype=np.float64).mean()
    m3 = np.asarray(level3, dtype=np.float64).mean()
    assert abs(m3 / m0 - 0.1) < 0.02


def test_noise_level3_has_larger_std_than_level1():
    base = _texture_image(seed=3)
    n1 = L.degrade(base, "noise", 1, seed=42)
    n3 = L.degrade(base, "noise", 3, seed=42)
    std1 = np.asarray(n1, dtype=np.float64).std()
    std3 = np.asarray(n3, dtype=np.float64).std()
    assert std3 > std1


def test_jpeg_level3_differs_from_base_more_than_level1():
    base = _texture_image(seed=4)
    j1 = L.degrade(base, "jpeg", 1, seed=7)
    j3 = L.degrade(base, "jpeg", 3, seed=7)
    base_arr = np.asarray(base, dtype=np.float64)
    diff1 = np.abs(np.asarray(j1, dtype=np.float64) - base_arr).mean()
    diff3 = np.abs(np.asarray(j3, dtype=np.float64) - base_arr).mean()
    assert diff3 > diff1


def test_blur_level3_has_lower_laplacian_variance_than_level1():
    base = _texture_image(seed=5)
    b1 = L.degrade(base, "blur", 1, seed=9)
    b3 = L.degrade(base, "blur", 3, seed=9)
    v1 = L._laplacian_variance(L._grayscale_array(b1))
    v3 = L._laplacian_variance(L._grayscale_array(b3))
    assert v3 < v1


def test_split_rule_balances_every_split_level_combo():
    n = 800  # divisible by 8
    counts: dict[tuple[str, int], int] = {}
    for i in range(n):
        level = i % 4
        split = L._split_for(i)
        counts[(split, level)] = counts.get((split, level), 0) + 1
    assert set(counts) == {(s, level) for s in ("calibration", "test") for level in range(4)}
    assert set(counts.values()) == {n // 8}


def test_ladders_dict_has_exactly_five_ladders_with_four_levels_and_params():
    assert set(L.LADDERS) == {"blur", "noise", "jpeg", "exposure", "resolution"}
    for name, spec in L.LADDERS.items():
        assert set(spec) == {"instructions", "levels", "params"}
        assert isinstance(spec["instructions"], str) and "img0" in spec["instructions"]
        assert len(spec["levels"]) == 4 and all(isinstance(s, str) for s in spec["levels"])
        assert len(spec["params"]) == 4

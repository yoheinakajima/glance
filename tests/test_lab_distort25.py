from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glance.lab import distort25 as D

# Distortions with a genuinely stochastic step (independent per-pixel randomness, or random patch/block
# positions) get a small tolerance on the monotonicity check below; distortions that are purely a
# deterministic function of a fixed severity parameter do not.
_STOCHASTIC_KEYS = {
    "white_noise",
    "color_noise",
    "impulse_noise",
    "multiplicative_noise",
    "jitter",
    "patch_shuffle",
    "color_block",
    "motion_blur",
}


def _texture_image(size: int = 128, seed: int = 0) -> Image.Image:
    """A synthetic but *textured* RGB image: smooth gradients (for spatial correlation, like a real
    photo) plus a handful of solid-colored blobs (for edges) plus light per-pixel noise (for fine
    detail). Pure i.i.d. pixel noise is a poor stand-in for a photo here: several distortions
    (pixelate, color_shift, blur, denoise...) only show a clear, monotone trend once neighboring
    pixels are correlated the way they are in an actual image."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    r = 128 + 100 * np.sin(x / 12.0)
    g = 128 + 100 * np.cos(y / 9.0 + x / 20.0)
    b = 128 + 80 * np.sin((x + y) / 15.0)
    arr = np.stack([r, g, b], axis=-1)
    for _ in range(8):
        cx, cy = rng.integers(10, size - 10, size=2)
        radius = rng.integers(8, 20)
        color = rng.integers(0, 256, size=3).astype(np.float64)
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        arr[mask] = color
    arr = arr + rng.normal(0, 10, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_distortions_dict_has_25_keys_each_with_description_and_5_params():
    assert len(D.DISTORTIONS) == 25
    for key, spec in D.DISTORTIONS.items():
        assert set(spec) == {"description", "params"}
        assert isinstance(spec["description"], str) and spec["description"]
        assert len(spec["params"]) == 5, key


@pytest.mark.parametrize("key", list(D.DISTORTIONS))
def test_degrade_is_deterministic_and_preserves_size_and_mode(key):
    base = _texture_image()
    for level in range(5):
        a = D.degrade(base, key, level, seed=123)
        b = D.degrade(base, key, level, seed=123)
        assert a.size == base.size, key
        assert b.size == base.size, key
        assert a.mode == "RGB" and b.mode == "RGB", key
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{key} level {level} not deterministic"


@pytest.mark.parametrize("key", list(D.DISTORTIONS))
def test_severity_is_non_decreasing_and_level0_is_visible(key):
    base = _texture_image()
    base_arr = np.asarray(base, dtype=np.float64)
    diffs = []
    for level in range(5):
        degraded = D.degrade(base, key, level, seed=123)
        diff = np.abs(np.asarray(degraded, dtype=np.float64) - base_arr).mean()
        diffs.append(diff)

    assert diffs[0] > 0, f"{key} level 0 does not differ from the base image"

    tolerance = 0.01 * max(diffs) if key in _STOCHASTIC_KEYS else 0.0
    for lo, hi in zip(diffs, diffs[1:]):
        assert hi + tolerance >= lo, f"{key} severity is not non-decreasing: {diffs}"


def test_split_rule_balances_every_split_level_combo():
    n = D.N_PER_SCALE  # 300, divisible by 10 (2 splits x 5 levels)
    counts: dict[tuple[str, int], int] = {}
    for i in range(n):
        level = i % 5
        split = D._split_for(i)
        counts[(split, level)] = counts.get((split, level), 0) + 1
    assert set(counts) == {(s, level) for s in ("calibration", "test") for level in range(5)}
    assert set(counts.values()) == {n // 10}


@pytest.mark.parametrize("key", list(D.DISTORTIONS))
def test_scales_instructions_reference_description_and_img0(key):
    spec = D.SCALES[key]
    instructions = spec["instructions"]
    description = D.DISTORTIONS[key]["description"]
    assert description in instructions
    assert "`img0`" in instructions
    assert spec["levels"] == D.LEVELS


# --- real-data smoke tests --------------------------------------------------------------------------
# Everything above is synthetic-image-only and needs no network access. The tests below exercise the
# real pets37 **train**-split pool (and a tiny end-to-end build()), which needs the train parquet file
# to be fetchable/cached (network access to the same pinned, CC BY-SA 4.0 Hub repo/revision `pets37`
# already uses). They skip -- rather than fail the suite -- when that file can't be fetched, e.g. no
# network in this environment.


def _real_pool_or_skip():
    from glance.config import load_config

    cfg = load_config()
    try:
        raw_bytes, pool = D._source_pool(cfg)
    except Exception as exc:  # noqa: BLE001 - any fetch/parse failure means "skip", not "fail"
        pytest.skip(f"pets37 train-split parquet unavailable: {exc}")
    return cfg, raw_bytes, pool


def test_real_train_pool_is_large_enough_and_prefixed():
    _cfg, _raw_bytes, pool = _real_pool_or_skip()
    assert len(pool) >= D.N_PER_SCALE, (
        f"train-split pool has only {len(pool)} sources after filtering, need >= {D.N_PER_SCALE}"
    )
    assert all(src.image_id.startswith(D.TRAIN_ID_PREFIX) for src in pool)


def test_real_build_smoke_writes_distinct_sources_per_level(tmp_path, monkeypatch):
    cfg, _raw_bytes, pool = _real_pool_or_skip()
    n = min(20, D.N_PER_SCALE, len(pool) - (len(pool) % 10))
    if n < 10:
        pytest.skip(f"train-split pool too small to exercise build(): {len(pool)}")

    # Redirect output to a temp dir so this test never overwrites the real generated dataset under
    # lab/manifests_distort25/ / .cache/lab_images_distort25/. PROJECT_ROOT is patched too since
    # `_relpath()` requires each output path to be relative to it.
    monkeypatch.setattr(D, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(D, "MANIFESTS_DIR", tmp_path / "manifests_distort25")
    monkeypatch.setattr(D, "IMAGES_DIR", tmp_path / "lab_images_distort25")
    smoke_keys = ["gaussian_blur", "jpeg", "color_block"]
    monkeypatch.setattr(D, "DISTORTIONS", {k: D.DISTORTIONS[k] for k in smoke_keys})

    results = D.build(cfg, n_per_scale=n, force=True)

    assert set(results) == set(smoke_keys)
    for key, rows in results.items():
        assert len(rows) == n
        assert len({r["source_image_id"] for r in rows}) == n  # distinct sources within a distortion
        counts: dict[tuple[str, int], int] = {}
        for row in rows:
            assert Path(row["path"]).is_absolute() is False
            assert (D.PROJECT_ROOT / row["path"]).exists()
            counts[(row["split"], row["level"])] = counts.get((row["split"], row["level"]), 0) + 1
        assert set(counts.values()) == {n // 10}

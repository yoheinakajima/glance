"""Fast tests for tools/classical_baselines.py, on synthetic images only (no manifests, no network,
no cache writes). `tools` has no __init__.py but is importable as a namespace package because
pytest (via tests/__init__.py) puts the project root on sys.path."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from tools.classical_baselines import FEATURE_NAMES, extract_features


def _synthetic_photo(size: int = 128, seed: int = 0) -> Image.Image:
    """A deterministic, textured RGB image: a smooth gradient plus a grid of small squares, so it
    has real structure (edges, gradients) rather than being flat or literal noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = (128 + 60 * np.sin(xx / 7.0) + 40 * np.cos(yy / 11.0)).clip(0, 255)
    rgb = np.stack([base, np.roll(base, 10, axis=0), np.roll(base, 20, axis=1)], axis=-1)
    # A sparse grid of high-contrast squares gives sharp edges for the Laplacian/Tenengrad features.
    for cy in range(8, size, 16):
        for cx in range(8, size, 16):
            rgb[cy : cy + 4, cx : cx + 4, :] = 255 if (cy + cx) % 32 == 0 else 0
    jitter = rng.normal(0, 1.0, size=rgb.shape)  # tiny deterministic jitter so it's not perfectly periodic
    rgb = np.clip(rgb + jitter, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_blur_lowers_laplacian_variance():
    sharp = _synthetic_photo()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=3))
    idx = FEATURE_NAMES.index("lap_var")
    lap_sharp = extract_features(sharp)[idx]
    lap_blurred = extract_features(blurred)[idx]
    assert lap_blurred < lap_sharp


def test_added_noise_raises_noise_sigma_estimate():
    clean = _synthetic_photo()
    arr = np.asarray(clean, dtype=np.float64)
    rng = np.random.default_rng(1)
    noisy_arr = np.clip(arr + rng.normal(0, 25.0, size=arr.shape), 0, 255).astype(np.uint8)
    noisy = Image.fromarray(noisy_arr, mode="RGB")
    idx = FEATURE_NAMES.index("noise_sigma")
    sigma_clean = extract_features(clean)[idx]
    sigma_noisy = extract_features(noisy)[idx]
    assert sigma_noisy > sigma_clean


def test_jpeg_compression_raises_blockiness():
    img = _synthetic_photo()
    mild = _jpeg_roundtrip(img, quality=95)
    heavy = _jpeg_roundtrip(img, quality=10)
    idx = FEATURE_NAMES.index("blockiness_mean")
    b_mild = extract_features(mild)[idx]
    b_heavy = extract_features(heavy)[idx]
    assert b_heavy > b_mild


def test_feature_vector_is_deterministic_and_fixed_length():
    img = _synthetic_photo()
    f1 = extract_features(img)
    f2 = extract_features(img)
    assert f1.shape == (len(FEATURE_NAMES),)
    assert f1.dtype == np.float64
    np.testing.assert_array_equal(f1, f2)
    assert np.all(np.isfinite(f1))
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

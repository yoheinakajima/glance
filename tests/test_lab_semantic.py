from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from glance.lab import semantic as S

EXPECTED_SCALES: dict[str, dict] = {
    "subject_size": {
        "instructions": "How much of the frame does the animal in `img0` fill?",
        "levels": [
            "The animal is tiny in the frame",
            "The animal is small in the frame",
            "The animal fills a moderate part of the frame",
            "The animal fills most of the frame",
            "The animal fills almost the entire frame",
        ],
    },
    "cutoff": {
        "instructions": "How much of the animal in `img0` is cut off by the edge of the frame?",
        "levels": [
            "The whole animal is inside the frame",
            "A small part of the animal is cut off",
            "A noticeable part of the animal is cut off",
            "About half of the animal is cut off",
            "Most of the animal is cut off",
        ],
    },
    "off_center": {
        "instructions": "How far from the centre of the frame is the animal in `img0`?",
        "levels": [
            "The animal is centred",
            "The animal is slightly off-centre",
            "The animal is clearly off-centre",
            "The animal is far off-centre",
            "The animal is at the very edge of the frame",
        ],
    },
    "occlusion": {
        "instructions": "How much of the animal in `img0` is hidden behind the grey rectangle?",
        "levels": [
            "None of the animal is hidden",
            "A small part of the animal is hidden",
            "A noticeable part of the animal is hidden",
            "About half of the animal is hidden",
            "Most of the animal is hidden",
        ],
    },
    "tilt": {
        "instructions": "How tilted is the photo `img0`, compared with a level camera?",
        "levels": ["Level", "Slightly tilted", "Noticeably tilted", "Strongly tilted", "Extremely tilted"],
    },
    "text_legibility": {
        "instructions": "How easy is it to read the caption printed on `img0`?",
        "levels": [
            "The caption cannot be read",
            "The caption is very hard to read",
            "The caption is readable with effort",
            "The caption is easy to read",
            "The caption is very easy to read",
        ],
    },
    "watermark": {
        "instructions": "How intrusive is the repeated SAMPLE watermark on `img0`?",
        "levels": [
            "No watermark is visible",
            "The watermark is barely visible",
            "The watermark is clearly visible",
            "The watermark is distracting",
            "The watermark dominates the image",
        ],
    },
}


def test_scales_shape_and_exact_texts():
    assert set(S.SCALES) == set(EXPECTED_SCALES)
    for key, expected in EXPECTED_SCALES.items():
        spec = S.SCALES[key]
        assert set(spec) == {"instructions", "levels"}, key
        assert spec["instructions"] == expected["instructions"], key
        assert spec["levels"] == expected["levels"], key
        assert len(spec["levels"]) == 5, key


def test_scale_params_cover_every_scale_with_5_levels():
    for key in ("subject_size", "cutoff", "off_center", "occlusion", "tilt", "watermark"):
        assert len(S.SCALE_PARAMS[key]) == 5, key
    assert len(S.TEXT_LEVEL_PARAMS) == 5


def test_captions_are_20_neutral_five_to_seven_word_sentences():
    assert len(S.CAPTIONS) == 20
    assert len(set(S.CAPTIONS)) == 20
    for caption in S.CAPTIONS:
        assert 5 <= len(caption.split()) <= 7, caption


# --- synthetic image + mask -----------------------------------------------------------------------
# A large canvas (so subject_size has room to "zoom out" to its smallest target) with a filled disc
# mask roughly centred, on a smoothly varying textured background (so blur/composite ops have some
# spatial structure to work with, like a real photo).


def _make_synthetic(w: int, h: int, radius: int, seed: int = 0) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 90 + 60 * np.sin(x / 80.0)
    g = 90 + 60 * np.cos(y / 70.0 + x / 130.0)
    b = 90 + 50 * np.sin((x + y) / 100.0)
    arr = np.stack([r, g, b], axis=-1)
    arr = arr + rng.normal(0, 6, size=arr.shape)

    cy, cx = h / 2.0, w / 2.0
    mask = (((y - cy) ** 2 + (x - cx) ** 2) <= radius**2).astype(np.uint8)
    animal_color = np.array([200.0, 150.0, 110.0])
    arr[mask.astype(bool)] = animal_color
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB"), mask


def _synthetic_source() -> tuple[Image.Image, np.ndarray]:
    """A moderate canvas (like a photo, not a square) with a filled disc mask centred on it, on a
    smoothly varying textured background. Sized so subject_size's tightest crop (level 4, 75% fill)
    stays within its 2x upscale limit, its widest crop (level 0, 6% fill) still fits inside the image,
    and occlusion's grey rectangle (sized against the mask *after* resizing to BASE_SIDE) has enough
    pixels to hit its targets without discretization gaps."""
    return _make_synthetic(w=1700, h=1300, radius=155)


def _synthetic_source_wide() -> tuple[Image.Image, np.ndarray]:
    """Like `_synthetic_source`, but substantially wider relative to the disc: off_center's higher
    targets need a source wider than both the animal's bounding box AND off_center's own
    level-independent window (see `_off_center_window`) -- a constraint the other scales don't share
    and that `_synthetic_source`'s more modest canvas doesn't satisfy. Even this (6000px wide, ~26x the
    disc's diameter) tops out at level 3 (target 0.6) -- see `_OFF_CENTER_INFEASIBLE_LEVELS`."""
    return _make_synthetic(w=6000, h=1200, radius=115)


def _source_for(key: str) -> tuple[Image.Image, np.ndarray]:
    return _synthetic_source_wide() if key == "off_center" else _synthetic_source()


# off_center level 4 (target 0.8) needs a source far wider than its own animal for the window to have
# both enough horizontal slack AND stay clear of the image edge (see `_off_center_window` and
# `_fit_off_center_box`'s module notes) -- on the real dataset that is true of essentially no photo
# (verified against the actual pool while building this scale), so `_synthetic_source_wide` is sized
# realistically rather than artificially inflated just to make level 4 pass here too.
_OFF_CENTER_INFEASIBLE_LEVELS = {4}


@pytest.mark.parametrize("key", ["subject_size", "cutoff", "off_center", "occlusion", "tilt", "watermark"])
def test_render_is_deterministic(key):
    base, mask = _source_for(key)
    for level in range(5):
        seed = S._seed_for(key, "synthetic", level)
        if key == "off_center" and level in _OFF_CENTER_INFEASIBLE_LEVELS:
            with pytest.raises(S.Infeasible):
                S.render(base, mask, key, level, seed)
            continue
        a = S.render(base, mask, key, level, seed)
        b = S.render(base, mask, key, level, seed)
        assert a.size == b.size
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{key} level {level} not deterministic"
        assert max(a.size) == S.BASE_SIDE


def test_render_text_legibility_is_deterministic_and_sized():
    base, mask = _synthetic_source()
    for level in range(5):
        seed = S._seed_for("text_legibility", "synthetic", level)
        a = S.render(base, mask, "text_legibility", level, seed)
        b = S.render(base, mask, "text_legibility", level, seed)
        assert np.array_equal(np.asarray(a), np.asarray(b))
        assert max(a.size) == S.BASE_SIDE


def test_every_caption_fits_within_width_at_the_largest_font_size():
    """Regression test for a real bug: at font sizes 29/36 several captions ran off both edges of the
    frame (e.g. "gathered the harvest before th..."). For every caption, measures each line
    `_caption_lines` produces at the LARGEST registered font size against a BASE_SIDE-wide canvas and
    checks it's within `_TEXT_MAX_WIDTH_FRACTION` of the frame width -- the size at which any overflow
    would be worst, since `_caption_lines` picks its layout using that size."""
    font = S._font(S.TEXT_LEVEL_PARAMS[-1]["font_size"])
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    limit = S._TEXT_MAX_WIDTH_FRACTION * S.BASE_SIDE
    for caption in S.CAPTIONS:
        lines = S._caption_lines(caption, S.BASE_SIDE)
        assert 1 <= len(lines) <= 2, f"{caption!r} wrapped to {len(lines)} lines"
        for line in lines:
            width = draw.textbbox((0, 0), line, font=font)[2]
            assert width <= limit + 1e-6, f"{caption!r} line {line!r} is {width}px, over the {limit}px limit"


def test_caption_layout_is_the_same_at_every_level():
    """The line-break decision must be a pure function of the caption text (and image width), never of
    `level` -- otherwise the number of lines itself becomes a legibility cue."""
    for caption in S.CAPTIONS:
        lines_ref = S._caption_lines(caption, S.BASE_SIDE)
        for _ in range(3):
            assert S._caption_lines(caption, S.BASE_SIDE) == lines_ref


@pytest.mark.parametrize("key,fit_fn", [
    ("subject_size", S._fit_subject_size_box),
    ("cutoff", S._fit_cutoff_box),
    ("off_center", S._fit_off_center_box),
])
def test_fit_scales_hit_target_within_tolerance_on_synthetic_mask(key, fit_fn):
    _base, mask = _source_for(key)
    for level, target in enumerate(S.SCALE_PARAMS[key]):
        if key == "off_center" and level in _OFF_CENTER_INFEASIBLE_LEVELS:
            assert fit_fn(mask, target, S._seed_for(key, "synthetic", level)) is None
            continue
        seed = S._seed_for(key, "synthetic", level)
        fit = fit_fn(mask, target, seed)
        assert fit is not None, f"{key} level {level} (target={target}) infeasible on synthetic mask"
        _box, achieved = fit
        tol = S._tol(key, target) if target > 0 else S.TOLERANCES[key][1]
        assert abs(achieved - target) <= tol + 1e-9, f"{key} level {level}: achieved={achieved} target={target}"


def test_off_center_window_is_level_independent():
    """The window's size (only its horizontal position) must not depend on the level/target -- an
    earlier version of `_fit_off_center_box` sized the window itself from the target, which made a
    single source's level 0 and level 4 crops wildly different in aspect ratio and zoom: a giveaway
    cue unrelated to the animal's actual position."""
    _base, mask = _synthetic_source_wide()
    sizes = set()
    for level, target in enumerate(S.SCALE_PARAMS["off_center"]):
        if level in _OFF_CENTER_INFEASIBLE_LEVELS:
            continue
        seed = S._seed_for("off_center", "synthetic", level)
        fit = S._fit_off_center_box(mask, target, seed)
        assert fit is not None, f"off_center level {level} infeasible on synthetic mask"
        (y0, y1, x0, x1), _achieved = fit
        sizes.add((round(y1 - y0, 3), round(x1 - x0, 3)))
    assert len(sizes) == 1, f"off_center window size varies across levels: {sizes}"


def test_occlusion_hits_target_within_tolerance_on_synthetic_mask():
    _base, mask = _synthetic_source()
    mask_small = S._mask_to_base(mask)
    for level, target in enumerate(S.SCALE_PARAMS["occlusion"]):
        seed = S._seed_for("occlusion", "synthetic", level)
        fit = S._fit_occlusion_rect(mask_small, target, seed)
        assert fit is not None, f"occlusion level {level} (target={target}) infeasible on synthetic mask"
        _box, achieved = fit
        tol = S.TOLERANCES["occlusion"][1]
        assert abs(achieved - target) <= tol + 1e-9, f"occlusion level {level}: achieved={achieved} target={target}"


def test_tilt_inscribed_scale_matches_expectations():
    assert S._inscribed_scale(1000, 1000, 0.0) == pytest.approx(1.0)
    for angle in (4, 9, 17, 30):
        k = S._inscribed_scale(1000, 800, angle)
        assert 0.0 < k < 1.0
    # bigger angle -> smaller (or equal) inscribed rectangle
    ks = [S._inscribed_scale(1000, 800, a) for a in (4, 9, 17, 30)]
    assert all(a >= b for a, b in zip(ks, ks[1:]))


def test_tilt_render_never_leaves_rotation_fill_in_the_crop():
    """Regression test for a real bug: `_inscribed_scale`'s crop is exact for the mathematical
    rectangle, but BICUBIC resampling blends in a sliver of the black rotation fill at the true edge,
    which showed up as dark wedges in the corners of `lab/sheets_semantic/tilt.jpg`. Renders a synthetic
    image at all 5 tilt levels and proves each crop is clean by running the *same* rotate+crop transform
    (same angle, resample, expand, fill) over a pure-white mask and checking no pixel dipped below 255."""
    base, mask = _synthetic_source()
    w, h = base.size
    magnitudes = S.SCALE_PARAMS["tilt"]
    for level in range(5):
        seed = S._seed_for("tilt", "synthetic", level)
        img = S.render(base, mask, "tilt", level, seed)
        assert img.size[0] > 0 and img.size[1] > 0

        rng = np.random.default_rng(seed)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        angle = 0.0 if level == 0 else sign * magnitudes[level]
        k_mag = magnitudes[2] if level == 0 else magnitudes[level]
        _rotated, box = S._tilt_crop(base, angle, k_mag)
        assert S._tilt_mask_is_clean(w, h, angle, box), f"tilt level {level}: crop still touches rotation fill"


def test_render_raises_infeasible_when_mask_is_empty():
    base = Image.new("RGB", (200, 200), (100, 100, 100))
    mask = np.zeros((200, 200), dtype=np.uint8)
    with pytest.raises(S.Infeasible):
        S.render(base, mask, "subject_size", 0, seed=1)


def test_split_rule_balances_every_split_level_combo():
    n = S.N_PER_SCALE  # 600, divisible by 10 (2 splits x 5 levels)
    counts: dict[tuple[str, int], int] = {}
    for i in range(n):
        level = i % 5
        split = S._split_for(i)
        counts[(split, level)] = counts.get((split, level), 0) + 1
    assert set(counts) == {(sp, level) for sp in ("calibration", "test") for level in range(5)}
    assert set(counts.values()) == {n // 10}


# --- real-data tests --------------------------------------------------------------------------------
# Everything above is synthetic-only and needs no network/dataset access. The tests below exercise the
# real pets37 train-split pool plus the Oxford-IIIT Pet trimaps and skip -- rather than fail the suite
# -- when either is unavailable (no network, or the trimaps haven't been downloaded yet).


def _real_pool_or_skip():
    from glance.config import load_config

    cfg = load_config()
    try:
        raw_bytes, pool = S._semantic_pool(cfg)
    except Exception as exc:  # noqa: BLE001 - any fetch/parse/extract failure means "skip", not "fail"
        pytest.skip(f"oxford-iiit-pet train pool or trimaps unavailable: {exc}")
    if not pool:
        pytest.skip("semantic source pool is empty (no photos with a matching trimap)")
    return cfg, raw_bytes, pool


def test_real_pool_has_matching_trimaps():
    _cfg, _raw_bytes, pool = _real_pool_or_skip()
    assert len(pool) > 0
    for src in pool[:5]:
        assert S._trimap_path(src.image_id).exists()


def test_real_build_smoke_writes_distinct_sources_per_level(tmp_path, monkeypatch):
    cfg, _raw_bytes, pool = _real_pool_or_skip()
    n = 20  # 4 per (split, level) cell for the FIT_KEYS scales below
    if len(pool) < n:
        pytest.skip(f"semantic pool too small to exercise build(): {len(pool)}")

    # Redirect manifest/image output to a temp dir; leave TRIMAPS_DIR/ANNOTATIONS_DIR pointed at the
    # real cache so the smoke build can still find the (already-downloaded) trimaps.
    monkeypatch.setattr(S, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(S, "MANIFESTS_DIR", tmp_path / "manifests_semantic")
    monkeypatch.setattr(S, "IMAGES_DIR", tmp_path / "lab_images_semantic")
    smoke_keys = ["tilt", "watermark", "off_center"]
    monkeypatch.setattr(S, "SCALES", {k: S.SCALES[k] for k in smoke_keys})

    try:
        results = S.build(cfg, n_per_scale=n, force=True)
    except RuntimeError as exc:
        pytest.skip(f"smoke build could not fill every cell on this pool: {exc}")

    assert set(results) == set(smoke_keys)
    for key, rows in results.items():
        assert len(rows) == n
        assert len({r["source_image_id"] for r in rows}) == n  # distinct sources within a scale
        counts: dict[tuple[str, int], int] = {}
        for row in rows:
            assert Path(row["path"]).is_absolute() is False
            assert (S.PROJECT_ROOT / row["path"]).exists()
            assert "param" in row
            counts[(row["split"], row["level"])] = counts.get((row["split"], row["level"]), 0) + 1
        assert set(counts.values()) == {n // 10}

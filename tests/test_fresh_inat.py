"""Offline tests for the fresh iNaturalist fetcher and suites. No network: the fetcher's keep/skip filter is
exercised on hand-made observation dicts, and the suites are built from a tiny fake manifest with tiny generated
JPEGs in a tmp dir (SOURCE/IMAGES and base.MANIFEST_DIR monkeypatched so nothing under the real .cache/ or
glance/evals/manifests/ is ever read or written)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from glance.evals.suites import base
from glance.evals.suites import fresh_inat as FI
from tools.fetch_fresh_inat import keep_observation, square_to_medium

CUTOFF = "2026-08-15"


def _obs(iconic="Aves", rank="species", license_code="cc-by", observed_on="2026-09-01", login="alice", quality_grade="research"):
    return {
        "quality_grade": quality_grade,
        "taxon": {"iconic_taxon_name": iconic, "rank": rank, "id": 1, "name": "Turdus migratorius", "preferred_common_name": "American Robin"},
        "observed_on": observed_on,
        "user": {"login": login},
        "photos": [{"license_code": license_code, "id": 123, "attribution": "(c) alice",
                    "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/123/square.jpg"}],
        "captive": False,
        "id": 999,
        "num_identification_agreements": 2,
    }


# --- fetcher filter ---------------------------------------------------------------------------------


def test_keep_good_observation():
    photo = keep_observation(_obs(), "Aves", CUTOFF, set())
    assert photo is not None and photo["license_code"] == "cc-by"


def test_skip_wrong_iconic_taxon():
    assert keep_observation(_obs(iconic="Insecta"), "Aves", CUTOFF, set()) is None


def test_skip_nc_license():
    assert keep_observation(_obs(license_code="cc-by-nc"), "Aves", CUTOFF, set()) is None


def test_skip_old_date():
    assert keep_observation(_obs(observed_on="2026-01-01"), "Aves", CUTOFF, set()) is None


def test_skip_duplicate_observer():
    assert keep_observation(_obs(login="alice"), "Aves", CUTOFF, {"alice"}) is None


def test_skip_missing_rank():
    assert keep_observation(_obs(rank=""), "Aves", CUTOFF, set()) is None


def test_skip_non_research_grade():
    assert keep_observation(_obs(quality_grade="needs_id"), "Aves", CUTOFF, set()) is None


def test_captive_is_kept_not_skipped():
    obs = _obs()
    obs["captive"] = True
    assert keep_observation(obs, "Aves", CUTOFF, set()) is not None


# --- URL rewrite -------------------------------------------------------------------------------------


@pytest.mark.parametrize("ext", ["jpg", "jpeg", "png", "JPG"])
def test_square_to_medium_rewrites_known_extensions(ext):
    url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/123/square.{ext}"
    out = square_to_medium(url)
    assert out == f"https://inaturalist-open-data.s3.amazonaws.com/photos/123/medium.{ext}"


def test_square_to_medium_none_when_no_match():
    assert square_to_medium("https://example.com/photos/123/large.jpg") is None


# --- suites --------------------------------------------------------------------------------------------


@pytest.fixture
def fake_manifest(tmp_path, monkeypatch):
    images_dir = tmp_path / "inat_images"
    images_dir.mkdir()
    rows = []
    classes = ["bird", "insect", "plant"]
    for key in classes:
        for i in range(2):
            item_id = f"{key}_{i}"
            file_name = f"{item_id}.jpg"
            Image.new("RGB", (16, 16), (10 * i, 20, 30)).save(images_dir / file_name, "JPEG")
            rows.append({
                "item_id": item_id, "label": key, "file": file_name,
                "page": f"https://www.inaturalist.org/observations/{i}", "photo_license": "cc-by",
                "observed_on": "2026-09-01", "taxon_name": "Some species",
            })
    source = tmp_path / "fresh_inat_source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(FI, "SOURCE", source)
    monkeypatch.setattr(FI, "IMAGES", images_dir)
    monkeypatch.setattr(base, "MANIFEST_DIR", tmp_path / "manifests")
    return rows


def test_inat_choice_builds_one_item_per_row_with_matching_labels(cfg, tmp_path, fake_manifest):
    cfg.paths.eval_images = str(tmp_path / "eval_images")
    items = FI.MODULES["inat_choice"].build(cfg, 6)
    assert len(items) == 6
    assert {i.item_id for i in items} == {r["item_id"] for r in fake_manifest}
    by_id = {r["item_id"]: r["label"] for r in fake_manifest}
    assert all(i.label == by_id[i.item_id] for i in items)
    assert all(i.question["type"] == "choice" and set(i.question["criteria"]) == set(FI.CLASSES) for i in items)


def test_inat_yesno_builds_two_balanced_items_per_row(cfg, tmp_path, fake_manifest):
    cfg.paths.eval_images = str(tmp_path / "eval_images")
    items = FI.MODULES["inat_yesno"].build(cfg, 12)
    assert len(items) == 12  # one yes + one no per photo
    assert sum(i.label is True for i in items) == 6
    assert sum(i.label is False for i in items) == 6
    # every item's own-class question is answered yes, the seeded other-class question no
    by_row = {r["item_id"]: r["label"] for r in fake_manifest}
    for item in items:
        base_id, _, key = item.item_id.partition("__")
        expected = key == by_row[base_id]
        assert item.label == expected


def test_inat_choice_skipped_without_manifest(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(FI, "SOURCE", tmp_path / "missing.jsonl")
    monkeypatch.setattr(FI, "IMAGES", tmp_path / "missing_images")
    with pytest.raises(base.SuiteSkipped, match="fetch_fresh_inat"):
        FI.MODULES["inat_choice"].build(cfg, 4)

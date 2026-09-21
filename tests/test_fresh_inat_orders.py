"""Offline tests for the fresh iNaturalist ORDERS fetcher and suites (the harder E20 set, lab/NOTES.md entry 47).
No network: the fetcher's ancestor-id filter and taxon-id resolver are exercised on hand-made dicts, and the
suites are built from a tiny fake manifest with tiny generated JPEGs in a tmp dir (SOURCE/IMAGES and
base.MANIFEST_DIR monkeypatched so nothing under the real .cache/ or glance/evals/manifests/ is ever read or
written)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from glance.evals.suites import base
from glance.evals.suites import fresh_inat_orders as FIO
from tools.fetch_fresh_inat_orders import keep_observation, resolve_order_taxon_id

CUTOFF = "2026-08-15"
ORDER_ID = 47208  # stand-in for Coleoptera's real taxon id; the filter only cares that it's an int
OTHER_ORDER_ID = 47744  # stand-in for a different order, e.g. Hemiptera


def _obs(order_id=ORDER_ID, ancestor_ids=None, taxon_id=555111, rank="species", license_code="cc-by",
         observed_on="2026-09-01", login="alice", quality_grade="research"):
    if ancestor_ids is None:
        ancestor_ids = [48460, 1, 47120, 372739, 47158, 184884, order_id, taxon_id]
    return {
        "quality_grade": quality_grade,
        "taxon": {"id": taxon_id, "ancestor_ids": ancestor_ids, "rank": rank, "name": "Harmonia axyridis",
                  "preferred_common_name": "Harlequin Ladybird", "iconic_taxon_name": "Insecta"},
        "observed_on": observed_on,
        "user": {"login": login},
        "photos": [{"license_code": license_code, "id": 123, "attribution": "(c) alice",
                    "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/123/square.jpg"}],
        "captive": False,
        "id": 999,
        "num_identification_agreements": 2,
    }


# --- fetcher filter (ancestor-id match instead of iconic-taxon-name equality) ------------------------


def test_keep_right_order_via_ancestor_ids():
    photo = keep_observation(_obs(), ORDER_ID, CUTOFF, set())
    assert photo is not None and photo["license_code"] == "cc-by"


def test_keep_when_taxon_id_itself_is_the_order():
    # identified only to order rank: taxon.id == order_id (may or may not also be in ancestor_ids)
    obs = _obs(taxon_id=ORDER_ID, ancestor_ids=[48460, 1, 47120, 372739, 47158, 184884, ORDER_ID])
    assert keep_observation(obs, ORDER_ID, CUTOFF, set()) is not None


def test_skip_wrong_order():
    obs = _obs(order_id=OTHER_ORDER_ID, ancestor_ids=[48460, 1, 47120, 372739, 47158, 184884, OTHER_ORDER_ID, 555111])
    assert keep_observation(obs, ORDER_ID, CUTOFF, set()) is None


def test_skip_nc_license():
    assert keep_observation(_obs(license_code="cc-by-nc"), ORDER_ID, CUTOFF, set()) is None


def test_skip_old_date():
    assert keep_observation(_obs(observed_on="2026-01-01"), ORDER_ID, CUTOFF, set()) is None


def test_skip_duplicate_observer():
    assert keep_observation(_obs(login="alice"), ORDER_ID, CUTOFF, {"alice"}) is None


def test_skip_non_research_grade():
    assert keep_observation(_obs(quality_grade="needs_id"), ORDER_ID, CUTOFF, set()) is None


def test_captive_is_kept_not_skipped():
    obs = _obs()
    obs["captive"] = True
    assert keep_observation(obs, ORDER_ID, CUTOFF, set()) is not None


# --- taxon-id resolver (pure function over a parsed /v1/taxa payload) ---------------------------------


def test_resolve_order_taxon_id_matches_exact_name_and_order_rank():
    payload = {"results": [
        {"id": 47158, "name": "Insecta", "rank": "class"},
        {"id": 999, "name": "Coleopterinae", "rank": "suborder"},
        {"id": 47208, "name": "Coleoptera", "rank": "order"},
    ]}
    assert resolve_order_taxon_id(payload, "Coleoptera") == 47208


def test_resolve_order_taxon_id_ignores_same_name_wrong_rank():
    payload = {"results": [{"id": 1, "name": "Coleoptera", "rank": "suborder"}]}
    with pytest.raises(ValueError):
        resolve_order_taxon_id(payload, "Coleoptera")


def test_resolve_order_taxon_id_raises_when_no_results():
    with pytest.raises(ValueError):
        resolve_order_taxon_id({"results": []}, "Coleoptera")


# --- suites --------------------------------------------------------------------------------------------


@pytest.fixture
def fake_manifest(tmp_path, monkeypatch):
    images_dir = tmp_path / "inat_orders_images"
    images_dir.mkdir()
    rows = []
    classes = list(FIO.CLASSES)  # all seven classes, so every negative below has a real candidate
    for key in classes:
        for i in range(2):
            item_id = f"{key}_{i}"
            file_name = f"{item_id}.jpg"
            Image.new("RGB", (16, 16), (10 * i, 20, 30)).save(images_dir / file_name, "JPEG")
            rows.append({
                "item_id": item_id, "label": key, "file": file_name,
                "page": f"https://www.inaturalist.org/observations/{i}", "photo_license": "cc-by",
                "observed_on": "2026-09-01", "taxon_name": "Some species", "order_name": key.capitalize(),
            })
    source = tmp_path / "fresh_inat_orders_source.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))

    monkeypatch.setattr(FIO, "SOURCE", source)
    monkeypatch.setattr(FIO, "IMAGES", images_dir)
    monkeypatch.setattr(base, "MANIFEST_DIR", tmp_path / "manifests")
    return rows


def test_inat_orders_choice_builds_one_item_per_row_with_matching_labels(cfg, tmp_path, fake_manifest):
    cfg.paths.eval_images = str(tmp_path / "eval_images")
    items = FIO.MODULES["inat_orders_choice"].build(cfg, 14)
    assert len(items) == 14  # 7 classes x 2 rows
    assert {i.item_id for i in items} == {r["item_id"] for r in fake_manifest}
    by_id = {r["item_id"]: r["label"] for r in fake_manifest}
    assert all(i.label == by_id[i.item_id] for i in items)
    assert all(i.question["type"] == "choice" and set(i.question["criteria"]) == set(FIO.CLASSES) for i in items)


def test_inat_orders_yesno_builds_two_balanced_items_per_row(cfg, tmp_path, fake_manifest):
    cfg.paths.eval_images = str(tmp_path / "eval_images")
    items = FIO.MODULES["inat_orders_yesno"].build(cfg, 28)
    assert len(items) == 28  # one yes + one no per photo
    assert sum(i.label is True for i in items) == 14
    assert sum(i.label is False for i in items) == 14
    by_row = {r["item_id"]: r["label"] for r in fake_manifest}
    for item in items:
        base_id, _, key = item.item_id.partition("__")
        expected = key == by_row[base_id]
        assert item.label == expected
        assert key in FIO.CLASSES  # negatives are always one of the seven look-alike orders...
        if item.label is False:
            assert key != by_row[base_id]  # ...and never the photo's own order


def test_inat_orders_choice_skipped_without_manifest(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(FIO, "SOURCE", tmp_path / "missing.jsonl")
    monkeypatch.setattr(FIO, "IMAGES", tmp_path / "missing_images")
    with pytest.raises(base.SuiteSkipped, match="fetch_fresh_inat_orders"):
        FIO.MODULES["inat_orders_choice"].build(cfg, 4)

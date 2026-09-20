"""Tests for glance.lab.kadid that need no data and no network. Every test that touches the
filesystem points the module's ROOT/DATASET_DIR/MANIFEST_DIR at tmp_path via monkeypatch, so nothing
under the real .cache/ or lab/manifests_kadid/ is ever read or written."""

from __future__ import annotations

import hashlib

import pytest

from glance.lab import kadid as K


def _patch_paths(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "datasets" / "kadid10k"
    monkeypatch.setattr(K, "DATASET_DIR", dataset_dir)
    monkeypatch.setattr(K, "ROOT", dataset_dir / "kadid10k")
    monkeypatch.setattr(K, "MANIFEST_DIR", tmp_path / "manifests_kadid")
    return dataset_dir


def test_reference_split_is_41_calibration_40_test():
    split = K.reference_split()
    assert len(split) == 81
    assert sum(v == "calibration" for v in split.values()) == 41
    assert sum(v == "test" for v in split.values()) == 40
    assert set(split.values()) == {"calibration", "test"}


def test_reference_split_is_deterministic():
    a = K.reference_split()
    b = K.reference_split()
    assert a == b


def test_scales_has_25_entries_with_generic_template_and_5_levels():
    assert len(K.SCALES) == 25
    for key, spec in K.SCALES.items():
        assert set(spec) == {"instructions", "levels", "kadid_index"}
        assert isinstance(spec["instructions"], str)
        assert spec["instructions"].startswith("How strong is the ")
        assert spec["instructions"].endswith(" in `img0`?")
        assert spec["levels"] == K.LEVELS
        assert len(spec["levels"]) == 5


def test_cli_refuses_download_without_accept_flag(monkeypatch, tmp_path, capsys):
    _patch_paths(monkeypatch, tmp_path)
    rc = K.main(["--download"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "no formal license" in captured.err
    assert "evaluation" in captured.err.lower()
    assert "--accept-evaluation-only" in captured.err


def test_cli_zip_with_wrong_hash_refuses(monkeypatch, tmp_path, capsys):
    _patch_paths(monkeypatch, tmp_path)
    bogus = tmp_path / "not_really_kadid10k.zip"
    bogus.write_bytes(b"not a real zip, just needs the wrong hash")
    assert hashlib.sha256(bogus.read_bytes()).hexdigest() != K.ZIP_SHA256

    rc = K.main(["--zip", str(bogus)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "sha256 mismatch" in captured.err
    # nothing should have been unpacked
    assert not K.ROOT.exists()


def test_cli_missing_data_message(monkeypatch, tmp_path, capsys):
    dataset_dir = _patch_paths(monkeypatch, tmp_path)
    assert not dataset_dir.exists()

    rc = K.main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert "--zip" in captured.err
    assert "--download --accept-evaluation-only" in captured.err
    assert K.ZIP_SHA256 in captured.err

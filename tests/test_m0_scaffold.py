import json
import re
import warnings

import pytest

from glance import logging_utils as lu
from glance.config import Config, load_config
from glance.doctor import select_tier


def test_default_config_loads_and_pins_models():
    cfg = load_config()
    assert isinstance(cfg, Config)
    sha = re.compile(r"^[0-9a-f]{40}$")
    assert sha.match(cfg.models.siglip.revision)
    for tier in cfg.models.vlm_tiers.values():
        assert sha.match(tier.revision)
    assert sha.match(cfg.models.mlx.revision)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.path("logs").is_absolute()


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths: {}\nmodels: {}\nnot_a_section: 1\n")
    with pytest.raises(Exception):
        load_config(bad)


def test_config_overrides():
    cfg = load_config(overrides={"vlm": {"prefix_cache": False}})
    assert cfg.vlm.prefix_cache is False


@pytest.mark.parametrize(
    "device,ram,vram,expected",
    [
        ("cuda", 64, 24.0, "cuda_24gb"),
        ("cuda", 64, 80.0, "cuda_24gb"),
        ("cuda", 64, 16.0, "cuda_12gb"),
        ("cuda", 64, 8.0, "cpu"),
        ("mps", 32, 21.0, "apple_32gb"),
        ("mps", 64, 48.0, "apple_32gb"),
        ("mps", 16, 10.0, "apple_8gb"),
        ("mps", 8, 5.0, "apple_8gb"),
        ("mps", 24, 16.0, "apple_8gb"),
        ("cpu", 64, None, "cpu"),
    ],
)
def test_tier_table(device, ram, vram, expected):
    tier, _ = select_tier(device, ram, vram)
    assert tier == expected


def test_tier_gap_warns():
    _, warns = select_tier("mps", 24, 16.0)
    assert warns


def test_request_ids_unique_and_sortable():
    ids = [lu.new_request_id() for _ in range(200)]
    assert len(set(ids)) == 200
    assert all(i.startswith("req_") and len(i) == 30 for i in ids)


def test_jsonl_writer_roundtrip(tmp_path):
    import numpy as np

    writer = lu.JsonlWriter(tmp_path / "sub" / "x.jsonl")
    writer.write({"a": 1, "arr": np.array([1.5, 2.5]), "f": np.float32(0.25)})
    writer.write({"b": "two"})
    rows = lu.read_jsonl(tmp_path / "sub" / "x.jsonl")
    assert rows == [{"a": 1, "arr": [1.5, 2.5], "f": 0.25}, {"b": "two"}]


def test_timer_spans():
    timer = lu.Timer()
    with timer.span("score"):
        pass
    result = timer.result()
    assert set(result) == {"load", "prefix", "score", "calibrate", "total"}
    assert all(isinstance(v, int) for v in result.values())


def test_error_record_has_traceback():
    try:
        raise ValueError("boom")
    except ValueError as exc:
        rec = lu.error_record("backend_error", exc)
    assert rec["code"] == "backend_error"
    assert rec["exception_type"] == "ValueError"
    assert "boom" in rec["traceback"]


def test_mps_fallback_logged_once(tmp_path, monkeypatch):
    monkeypatch.setattr(lu.install_mps_fallback_logger, "_installed", False, raising=False)
    lu._seen_fallback_ops.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        lu.install_mps_fallback_logger(tmp_path)
        msg = "The operator 'aten::fake_op' is not currently supported on the MPS backend and will fall back to run on the CPU."
        warnings.warn(msg, UserWarning)
        warnings.warn(msg, UserWarning)
    rows = lu.read_jsonl(tmp_path / "mps_fallback.jsonl")
    assert [r["op"] for r in rows] == ["aten::fake_op"]
    assert lu.mps_fallback_ops() == ["aten::fake_op"]


def test_doctor_cli_writes_json(capsys):
    from glance.cli import main

    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    for key in ("os", "chip", "ram_gb", "vram_gb", "device", "dtype", "torch_version", "free_disk_gb",
                "selected_tier", "selected_models", "warnings"):
        assert key in report
    cfg = load_config()
    assert json.loads((cfg.path("logs") / "doctor.json").read_text())["selected_tier"] == report["selected_tier"]

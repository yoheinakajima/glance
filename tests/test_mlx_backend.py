"""Experimental MLX runtime plumbing. These tests never import or load MLX."""

import sys

import pytest

from glance.cli import _config_overrides, build_parser
from glance.config import load_config
from glance.schema import BackendError


def test_default_stays_torch_and_mlx_candidate_is_pinned():
    cfg = load_config()
    assert cfg.vlm.backend == "torch"
    assert cfg.models.mlx.id == "mlx-community/Qwen3-VL-2B-Instruct-8bit"
    assert cfg.models.mlx.revision == "b0338e0e843d8e1befe873d144b81fefdc47efa6"
    assert cfg.models.mlx.dtype == "int8"


@pytest.mark.parametrize(
    "argv",
    [
        ["--backend", "mlx", "ask", "x.jpg", "Is it a dog?"],
        ["ask", "--backend", "mlx", "x.jpg", "Is it a dog?"],
        ["serve", "--backend", "mlx", "--preload", "vlm"],
    ],
)
def test_cli_backend_selector_reaches_config(argv):
    args = build_parser().parse_args(argv)
    assert load_config(None, _config_overrides(args)).vlm.backend == "mlx"


def test_python_api_selects_mlx_without_loading_it():
    from glance import Glance

    glance = Glance(backend="mlx")
    assert glance.engine.cfg.vlm.backend == "mlx"
    assert "mlx" not in sys.modules and "mlx_vlm" not in sys.modules


def test_backend_router_selects_mlx_lazily(monkeypatch):
    from glance import backends
    from glance.backends import vlm_mlx

    seen = {}

    def fake_init(self, cfg):
        seen["backend"] = cfg.vlm.backend

    monkeypatch.setattr(vlm_mlx.MlxVlmBackend, "__init__", fake_init)
    backend = backends.load_backend("vlm", load_config(overrides={"vlm": {"backend": "mlx"}}))
    assert isinstance(backend, vlm_mlx.MlxVlmBackend)
    assert seen == {"backend": "mlx"}
    assert "mlx" not in sys.modules and "mlx_vlm" not in sys.modules


def test_non_apple_rejected_before_optional_import(monkeypatch):
    from glance.backends import vlm_mlx

    monkeypatch.setattr(vlm_mlx.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vlm_mlx.platform, "machine", lambda: "x86_64")
    with pytest.raises(BackendError, match="requires Apple Silicon"):
        vlm_mlx.MlxVlmBackend(load_config(overrides={"vlm": {"backend": "mlx"}}))
    assert "mlx" not in sys.modules and "mlx_vlm" not in sys.modules


def test_mlx_and_generic_model_override_are_not_silently_combined():
    from glance import backends

    cfg = load_config(overrides={
        "vlm": {"backend": "mlx"},
        "models": {"generic": {"id": "org/other-model"}},
    })
    with pytest.raises(BackendError, match="supports only the pinned"):
        backends.load_backend("vlm", cfg)


def test_doctor_reports_the_selected_mlx_runtime_without_importing_it():
    from glance.doctor import run_doctor

    report = run_doctor(load_config(overrides={"vlm": {"backend": "mlx"}}))
    assert report["vlm_backend"] == "mlx"
    assert report["dtype"] == "int8"
    assert report["selected_models"]["vlm"].startswith("mlx-community/Qwen3-VL-2B-Instruct-8bit@")
    assert "mlx" not in sys.modules and "mlx_vlm" not in sys.modules

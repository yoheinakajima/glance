"""The any-model backend (`--model-id`): configuration plumbing and routing, without loading a model."""
from glance.cli import _config_overrides, build_parser
from glance.config import load_config


def test_cli_model_id_reaches_the_config():
    args = build_parser().parse_args(["--model-id", "org/some-vlm", "--revision", "abc123", "--image-longest-edge", "768", "ask", "x.jpg", "Is it a dog?"])
    cfg = load_config(None, _config_overrides(args))
    assert cfg.models.generic is not None
    assert (cfg.models.generic.id, cfg.models.generic.revision, cfg.models.generic.image_longest_edge, cfg.models.generic.dtype) == ("org/some-vlm", "abc123", 768, "auto")


def test_default_config_keeps_the_tier_table():
    assert load_config().models.generic is None


def test_load_backend_routes_to_the_generic_backend(monkeypatch):
    from glance import backends
    from glance.backends import generic_hf

    seen = {}

    def fake_init(self, cfg, model_id, revision=None, longest_edge=None, dtype="auto", device=None):
        seen.update(model_id=model_id, revision=revision, longest_edge=longest_edge, dtype=dtype)

    monkeypatch.setattr(generic_hf.GenericVlmBackend, "__init__", fake_init)
    cfg = load_config(None, {"models": {"generic": {"id": "org/some-vlm", "revision": None, "dtype": "float32", "image_longest_edge": 512}}})
    backend = backends.load_backend("vlm", cfg)
    assert isinstance(backend, generic_hf.GenericVlmBackend) and backend.name == "vlm" and backend.kind == "vlm"
    assert seen == {"model_id": "org/some-vlm", "revision": None, "longest_edge": 512, "dtype": "float32"}


def test_python_api_accepts_a_model_id_without_loading_it():
    from glance import Glance

    g = Glance(model_id="org/some-vlm", revision="abc123")
    assert g.engine.cfg.models.generic.id == "org/some-vlm"


def test_lab_alias_still_imports():
    from glance.backends.generic_hf import GenericVlmBackend
    from glance.lab.generic_vlm import GenericVlm

    assert GenericVlm is GenericVlmBackend

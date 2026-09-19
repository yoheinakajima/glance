"""Config loader. Imports nothing from glance so every module can depend on it.

Also owns the project root and sets HF_HOME before anything imports huggingface_hub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(os.environ.get("GLANCE_ROOT") or Path(__file__).resolve().parent.parent)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class PathsConfig(_Strict):
    hf_home: str
    eval_images: str
    logs: str
    runs: str
    calibration: str
    human_gold: str


class SiglipModelConfig(_Strict):
    id: str
    revision: str
    photo_prefix: bool = False


class VlmTierConfig(_Strict):
    id: str
    revision: str
    dtype: str
    image_token_budget: int


class ModelsConfig(_Strict):
    siglip: SiglipModelConfig
    vlm_tiers: dict[str, VlmTierConfig]
    tier_override: str | None = None
    image_token_budget_override: int | None = None


class VlmConfig(_Strict):
    batch_size: int = 8
    prefix_cache: bool = True
    letter_rotations: int = 4
    off_mass_warn: float = 0.1


class LimitsConfig(_Strict):
    max_image_mb: int = 20
    formats: list[str] = ["JPEG", "PNG", "WEBP"]
    max_side: int = 2048


class EvalConfig(_Strict):
    seed: int = 7
    n_cuda: int = 1000
    n_apple: int = 500
    manifest_n: int = 1000
    max_hours: float = 4
    warmup_items: int = 10
    baseline_n: int = 300
    ece_bins: int = 15
    permutation_items: int = 100
    permutation_orders: int = 3
    letter_max_options: int = 26
    latency_repeats: int = 20
    max_failure_rate: float = 0.02
    top_errors: int = 20


class CalibrationConfig(_Strict):
    isotonic_min_n: int = 1000


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8077


class Config(_Strict):
    paths: PathsConfig
    models: ModelsConfig
    vlm: VlmConfig = VlmConfig()
    limits: LimitsConfig = LimitsConfig()
    eval: EvalConfig = EvalConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    server: ServerConfig = ServerConfig()

    def path(self, name: str) -> Path:
        """Resolve a configured path against the project root."""
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else PROJECT_ROOT / p


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load configs/default.yaml (or `path`), apply `overrides`, validate, and set up the environment."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        _deep_update(raw, overrides)
    cfg = Config.model_validate(raw)
    setup_environment(cfg)
    return cfg


def setup_environment(cfg: Config) -> None:
    """Read .env and point the Hugging Face cache at ./.cache/hf. Must run before importing transformers."""
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("HF_HOME", str(cfg.path("hf_home")))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

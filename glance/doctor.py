"""`glance doctor`: detect the device and pick the model tier from HANDOFF section 3."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any

from .config import PROJECT_ROOT, Config

DISK_BUDGET_GB = 30


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _ram_gb() -> float:
    if platform.system() == "Darwin":
        mem = _sysctl("hw.memsize")
        if mem:
            return round(int(mem) / 2**30, 1)
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)


def _chip() -> str:
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    return platform.processor() or platform.machine()


def select_tier(device: str, ram_gb: float, vram_gb: float | None) -> tuple[str, list[str]]:
    """Tier table from HANDOFF section 3. Gaps in the table resolve to the smaller tier, with a warning."""
    warnings: list[str] = []
    if device == "cuda":
        vram = vram_gb or 0.0
        if vram >= 24:
            return "cuda_24gb", warnings
        if vram >= 12:
            return "cuda_12gb", warnings
        warnings.append(f"CUDA device has {vram:.1f} GB (< 12 GB, outside the tier table): dual encoder only")
        return "cpu", warnings
    if device == "mps":
        if ram_gb >= 32:
            return "apple_32gb", warnings
        if ram_gb >= 8:
            if ram_gb > 16:
                warnings.append(f"{ram_gb:.0f} GB unified memory sits between tiers: using the smaller 8-16 GB tier")
            return "apple_8gb", warnings
        warnings.append(f"{ram_gb:.0f} GB unified memory is below the smallest VLM tier: dual encoder only")
        return "cpu", warnings
    return "cpu", warnings


def run_doctor(cfg: Config) -> dict[str, Any]:
    import torch

    warnings: list[str] = []
    ram_gb = _ram_gb()
    vram_gb: float | None = None
    if torch.cuda.is_available():
        device = "cuda"
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    elif torch.backends.mps.is_available():
        device = "mps"
        # Unified memory: report what Metal recommends as the working-set ceiling.
        vram_gb = round(torch.mps.recommended_max_memory() / 2**30, 1)
    else:
        device = "cpu"

    tier, tier_warnings = select_tier(device, ram_gb, vram_gb)
    warnings += tier_warnings
    if cfg.models.tier_override:
        warnings.append(f"tier_override set in config: {tier} -> {cfg.models.tier_override}")
        tier = cfg.models.tier_override

    selected_models: dict[str, Any] = {
        "siglip": f"{cfg.models.siglip.id}@{cfg.models.siglip.revision}",
        "vlm": None,
    }
    dtype = "float32"
    image_token_budget = None
    if tier in cfg.models.vlm_tiers:
        vlm = cfg.models.vlm_tiers[tier]
        selected_models["vlm"] = f"{vlm.id}@{vlm.revision}"
        dtype = vlm.dtype
        image_token_budget = cfg.models.image_token_budget_override or vlm.image_token_budget
    else:
        warnings.append("no VLM tier for this device: dual encoder only")

    free_disk_gb = round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 1)
    if free_disk_gb < DISK_BUDGET_GB:
        warnings.append(f"free disk {free_disk_gb} GB is below the {DISK_BUDGET_GB} GB budget")

    return {
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "chip": _chip(),
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "device": device,
        "dtype": dtype,
        "torch_version": torch.__version__,
        "free_disk_gb": free_disk_gb,
        "selected_tier": tier,
        "selected_models": selected_models,
        "image_token_budget": image_token_budget,
        "warnings": warnings,
    }

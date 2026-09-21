"""The package must work from a wheel (pip install glance-vlm), where no repository surrounds it."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_packaged_default_config_is_identical_to_the_repository_one():
    assert (ROOT / "glance" / "default_config.yaml").read_text() == (ROOT / "configs" / "default.yaml").read_text()


def test_config_loads_without_a_checkout(tmp_path):
    """Simulate an installed package: copy only the package directory somewhere else and load the default config from there."""
    import shutil

    site = tmp_path / "site"
    shutil.copytree(ROOT / "glance", site / "glance", ignore=shutil.ignore_patterns("__pycache__", "manifests", "lab"))
    code = ("import os, pathlib; from glance import config; cfg = config.load_config(None); "
            "assert config.INSTALLED and config.PROJECT_ROOT == pathlib.Path.home() / '.glance', config.PROJECT_ROOT; "
            "assert config.DEFAULT_CONFIG_PATH.name == 'default_config.yaml'; assert 'HF_HOME' not in os.environ; print(cfg.models.model_dump_json())")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site), "HOME": str(tmp_path / "home")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-800:]
    assert "Qwen3-VL" in out.stdout

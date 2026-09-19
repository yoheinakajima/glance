"""Dataset downloads. Every source is pinned and its license is recorded in DATASETS.md.

Downloads happen only when a suite is built, never at inference time.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from ...config import Config

USER_AGENT = "glance-eval/0.1"


def datasets_dir(cfg: Config) -> Path:
    path = cfg.path("eval_images").parent / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path, md5: str | None = None, quiet: bool = False) -> Path:
    """Fetch `url` to `dest` once. Prints size and destination first; verifies md5 when given."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as resp:  # noqa: S310 - fixed https/http dataset hosts
        size = int(resp.headers.get("Content-Length") or 0)
        if not quiet:
            print(f"DOWNLOAD {url} ({size / 1e6:.1f} MB) -> {dest}", file=sys.stderr)
        with open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
    if md5 is not None:
        digest = hashlib.md5(tmp.read_bytes()).hexdigest()  # noqa: S324 - integrity check against the publisher's md5
        if digest != md5:
            tmp.unlink()
            raise RuntimeError(f"md5 mismatch for {url}: got {digest}, expected {md5}")
    tmp.rename(dest)
    return dest


def hf_parquet(cfg: Config, repo: str, revision: str, filename: str) -> Path:
    """One parquet file from a Hub dataset repo at a pinned revision (cached under HF_HOME)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=filename))

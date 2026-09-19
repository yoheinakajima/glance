"""JSONL writer, request ids, timing context manager, MPS-fallback op logging.

Imports nothing from glance except config, so every module can use it.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from .config import PROJECT_ROOT

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_write_lock = threading.Lock()


def _b32(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def new_request_id() -> str:
    """ULID-style id: 48-bit ms timestamp + 80 random bits, so ids sort by time."""
    return "req_" + _b32(int(time.time() * 1000), 10) + _b32(secrets.randbits(80), 16)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@lru_cache(maxsize=1)
def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False)


class JsonlWriter:
    """Append-only JSONL file. One line per write, flushed, safe across threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        line = dumps(record) + "\n"
        with _write_lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)


def call_log_writer(logs_dir: str | Path) -> JsonlWriter:
    """logs/calls/YYYY-MM-DD.jsonl (UTC date)."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return JsonlWriter(Path(logs_dir) / "calls" / f"{day}.jsonl")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Timer:
    """Collects named spans in milliseconds: `with timer.span("score"): ...`."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - start) * 1000

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + ms

    def result(self, keys: tuple[str, ...] = ("load", "prefix", "score", "calibrate")) -> dict[str, int]:
        out = {k: int(round(self.ms.get(k, 0.0))) for k in keys}
        out["total"] = int(round((time.perf_counter() - self._t0) * 1000))
        return out


def error_record(code: str, exc: BaseException) -> dict[str, Any]:
    return {
        "code": code,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


# --- MPS fallback: log each op that falls back to CPU once, by name -------------------------------

_MPS_FALLBACK_RE = re.compile(r"The operator '([^']+)' is not currently supported on the MPS backend")
_seen_fallback_ops: set[str] = set()


def install_mps_fallback_logger(logs_dir: str | Path) -> None:
    """Route PyTorch's MPS->CPU fallback warnings into logs/mps_fallback.jsonl, once per op name."""
    if getattr(install_mps_fallback_logger, "_installed", False):
        return
    writer = JsonlWriter(Path(logs_dir) / "mps_fallback.jsonl")
    previous = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
        match = _MPS_FALLBACK_RE.search(str(message))
        if match:
            op = match.group(1)
            if op not in _seen_fallback_ops:
                _seen_fallback_ops.add(op)
                writer.write({"ts": utc_now(), "event": "mps_fallback", "op": op, "pid": os.getpid()})
            return
        previous(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    install_mps_fallback_logger._installed = True  # type: ignore[attr-defined]


def mps_fallback_ops() -> list[str]:
    return sorted(_seen_fallback_ops)

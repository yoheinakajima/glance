"""path | url | base64 -> PIL image, sha256, resize policy (HANDOFF sections 4 and 5)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import PROJECT_ROOT, LimitsConfig
from .schema import ImageLoadError, ImageRef

URL_TIMEOUT_S = 15


@dataclass
class LoadedImage:
    id: str
    image: Image.Image  # RGB, EXIF-rotated, downscaled to limits.max_side
    sha256: str  # of the original bytes
    width: int  # original size, before the resize policy
    height: int
    format: str
    source: str  # the path or URL as given; "base64" for inline bytes
    image_tokens: int | None = None  # filled in by the backend that encodes it

    def log_record(self) -> dict:
        return {
            "id": self.id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "image_tokens": self.image_tokens,
            "source": self.source,
        }


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path  # lets `samples/receipt.jpg` work from any cwd


def _read_bytes(ref: ImageRef, max_bytes: int) -> tuple[bytes, str]:
    limit_mb = max_bytes // 2**20
    if ref.path is not None:
        path = _resolve_path(ref.path)
        if not path.is_file():
            raise ImageLoadError(f"image `{ref.id}`: no file at {ref.path}", {"image": ref.id, "path": ref.path})
        if path.stat().st_size > max_bytes:
            raise ImageLoadError(f"image `{ref.id}` is over the {limit_mb} MB limit", {"image": ref.id})
        return path.read_bytes(), ref.path
    if ref.url is not None:
        if not ref.url.lower().startswith(("http://", "https://")):
            raise ImageLoadError(f"image `{ref.id}`: only http(s) URLs are allowed", {"image": ref.id})
        try:
            with urllib.request.urlopen(ref.url, timeout=URL_TIMEOUT_S) as resp:  # noqa: S310 - scheme checked above
                data = resp.read(max_bytes + 1)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise ImageLoadError(f"image `{ref.id}`: could not fetch URL ({exc})", {"image": ref.id, "url": ref.url}) from exc
        if len(data) > max_bytes:
            raise ImageLoadError(f"image `{ref.id}` is over the {limit_mb} MB limit", {"image": ref.id})
        return data, ref.url
    payload = ref.base64 or ""
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageLoadError(f"image `{ref.id}`: invalid base64", {"image": ref.id}) from exc
    if len(data) > max_bytes:
        raise ImageLoadError(f"image `{ref.id}` is over the {limit_mb} MB limit", {"image": ref.id})
    return data, "base64"


def load_image(ref: ImageRef, limits: LimitsConfig) -> LoadedImage:
    data, source = _read_bytes(ref, limits.max_image_mb * 2**20)
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
        if fmt not in limits.formats:
            raise ImageLoadError(
                f"image `{ref.id}`: format {fmt or 'unknown'} is not one of {limits.formats}", {"image": ref.id}
            )
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"image `{ref.id}`: bad image bytes ({exc})", {"image": ref.id}) from exc

    width, height = img.size
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    img = img.convert("RGB")
    if max(img.size) > limits.max_side:
        img.thumbnail((limits.max_side, limits.max_side), Image.LANCZOS)

    return LoadedImage(
        id=ref.id,
        image=img,
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
        format=fmt,
        source=source,
    )


def load_images(refs: list[ImageRef], limits: LimitsConfig) -> list[LoadedImage]:
    return [load_image(ref, limits) for ref in refs]

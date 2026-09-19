"""glance: typed image decisions from single forward passes."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("glance")
except _metadata.PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0+uninstalled"

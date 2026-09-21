"""Moved: the lab's second-family wrapper is now the any-model backend, `glance/backends/generic_hf.py`."""
from ..backends.generic_hf import GenericVlmBackend as GenericVlm

__all__ = ["GenericVlm"]

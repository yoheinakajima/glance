"""One loader per suite. Each recasts a dataset into typed questions with ground-truth labels."""

from __future__ import annotations

from . import blur_ladder, caltech101, doctype16, gqa_yesno, human_gold, pets37, pets37_openset, pope, pope_injection
from .base import EvalItem, SuiteInfo, SuiteSkipped, item_to_request
from .fresh_commons import MODULES as _FRESH_COMMONS_MODULES
from .fresh_inat import MODULES as _FRESH_INAT_MODULES
from .fresh_inat_orders import MODULES as _FRESH_INAT_ORDERS_MODULES
from .ladders import MODULES as _LADDER_MODULES
from .probes import MODULES as _PROBE_MODULES

SUITES = {
    module.INFO.name: module
    for module in (pope, gqa_yesno, pets37, caltech101, blur_ladder, doctype16, human_gold, pets37_openset, pope_injection)
}
DEFAULT_SUITES = ["pope", "gqa_yesno", "pets37", "caltech101", "blur_ladder", "doctype16", "human_gold"]
SUITES.update(_LADDER_MODULES)
SUITES.update(_FRESH_COMMONS_MODULES)  # opt-in: fresh real photos, labels from Commons structured data
SUITES.update(_FRESH_INAT_MODULES)  # opt-in: fresh real photos, labels from iNaturalist research-grade IDs
SUITES.update(_FRESH_INAT_ORDERS_MODULES)  # opt-in: fresh real photos, harder labels at insect-ORDER level (entry 47)
SUITES.update(_PROBE_MODULES)  # opt-in: images drawn by program, labels exact by construction (lab/NOTES.md entry 50)
STRETCH_SUITES = ["pets37_openset", "pope_injection"]  # opt-in: glance eval --suite <name>
LADDER_SUITES = list(_LADDER_MODULES)  # opt-in: the score lab's five rating scales
FRESH_SUITES = list(_FRESH_COMMONS_MODULES) + list(_FRESH_INAT_MODULES) + list(_FRESH_INAT_ORDERS_MODULES)  # opt-in: fresh real photos, no in-house labeling
SYNTHETIC_SUITES = list(_PROBE_MODULES)  # opt-in: generated images (procedural probes; synthetic UI screens when built)

__all__ = ["SUITES", "DEFAULT_SUITES", "EvalItem", "SuiteInfo", "SuiteSkipped", "item_to_request"]

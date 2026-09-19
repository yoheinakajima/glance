"""One loader per suite. Each recasts a dataset into typed questions with ground-truth labels."""

from __future__ import annotations

from . import blur_ladder, caltech101, doctype16, gqa_yesno, human_gold, pets37, pope
from .base import EvalItem, SuiteInfo, SuiteSkipped, item_to_request

SUITES = {
    module.INFO.name: module
    for module in (pope, gqa_yesno, pets37, caltech101, blur_ladder, doctype16, human_gold)
}
DEFAULT_SUITES = ["pope", "gqa_yesno", "pets37", "caltech101", "blur_ladder", "doctype16", "human_gold"]

__all__ = ["SUITES", "DEFAULT_SUITES", "EvalItem", "SuiteInfo", "SuiteSkipped", "item_to_request"]

"""Optional 16-way document-type suite from RVL-CDIP. Skipped in v0.

HANDOFF section 8: "Verify first; skip if unclear." The Hub cards for RVL-CDIP (`aharley/rvl_cdip`,
`chainyo/rvl-cdip`) list the license as "other" (checked 2026-09-19), and the source collection (IIT-CDIP,
tobacco litigation documents) has no clear reuse terms. That is unclear, so the suite does not run.
"""

from __future__ import annotations

from ...config import Config
from .base import EvalItem, SuiteInfo, SuiteSkipped

INFO = SuiteInfo(
    name="doctype16", qtype="choice",
    source="https://huggingface.co/datasets/aharley/rvl_cdip",
    license='unclear ("other" on the Hub card)',
)


def build(cfg: Config, n: int) -> list[EvalItem]:
    raise SuiteSkipped('RVL-CDIP license is "other" on its dataset card: unclear, so the suite is skipped')

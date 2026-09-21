"""Five suites on the synthetic UI screens built by `glance/lab/ui_screens.py` (lab/NOTES.md entry 49).

Every screen is self-contained HTML + inline CSS rendered by this project's own generator, so every label
below is exact by construction -- no human labeling, nothing any model could have seen before this repo
existed. Build the corpus first with `uv run python -c "from glance.lab import ui_screens; ui_screens.build()"`.

- `ui_state`: two yes/no questions per base screen, drawn from the six recorded state toggles (dialog open,
  error shown, main button disabled, a named checkbox checked, signed in, cookie banner) so yes and no are
  balanced over the whole suite. The checkbox question only appears on screens that have a checkbox.
- `ui_page`: which of the five page types `img0` is.
- `ui_click`: a goal sentence plus 6 to 8 numbered marks; pick the mark that serves the goal.
- `ui_done`: paired before/after screens; has the stated goal already been accomplished?
- `ui_reason`: a one-step reasoning question (cheaper plan, out-of-stock item, earliest slot) with its own
  numbered marks.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from types import SimpleNamespace

from ...config import PROJECT_ROOT, Config
from ...logging_utils import read_jsonl
from .base import MANIFEST_DIR, EvalItem, RawItem, SuiteInfo, SuiteSkipped, materialize

SOURCE = MANIFEST_DIR / "ui_screens_source.jsonl"
IMAGES = PROJECT_ROOT / ".cache" / "lab_images_ui"

BUILD_CMD = 'uv run python -c "from glance.lab import ui_screens; ui_screens.build()"'

PAGE_CRITERIA = {
    "sign_in": "A sign-in or log-in page", "search_results": "A page of search results",
    "cart": "A shopping cart or checkout page", "settings": "A settings or preferences page",
    "article": "An article or blog post",
}
STATE_KEYS = ("dialog_open", "error_shown", "primary_disabled", "checkbox_checked", "signed_in", "cookie_banner")
STATE_WORDING = {
    "dialog_open": "Is a dialog box open on top of the page in `img0`?",
    "error_shown": "Is an error message shown in `img0`?",
    "primary_disabled": "Is the main button in `img0` disabled (greyed out)?",
    "signed_in": "Is a user signed in on the page in `img0`?",
    "cookie_banner": "Is a cookie banner visible in `img0`?",
}


def _checkbox_wording(label: str) -> str:
    return f'Is the "{label}" checkbox checked in `img0`?'


def _all_rows() -> list[dict]:
    rows = read_jsonl(SOURCE)
    if not rows:
        raise SuiteSkipped(f"no synthetic UI screens manifest; run `{BUILD_CMD}`")
    if not (IMAGES / rows[0]["file"]).is_file():
        raise SuiteSkipped(f"synthetic UI screen images are missing; run `{BUILD_CMD}` to render them again")
    return rows


def _base_rows() -> list[dict]:
    """The 300 base screens (state / page / click ground truth): rows carrying a `goal`."""
    return [r for r in _all_rows() if "goal" in r]


def _done_rows() -> list[dict]:
    return [r for r in _all_rows() if "done" in r]


def _reason_rows() -> list[dict]:
    return [r for r in _all_rows() if "reason" in r]


def _raw(item_id: str, row: dict, question: dict, label, extra_meta: dict | None = None) -> RawItem:
    src = IMAGES / row["file"]
    meta = {"page_type": row["page_type"], "theme": row["theme"]}
    if extra_meta:
        meta.update(extra_meta)
    return RawItem(
        item_id=item_id, question=question, label=label, ext=".png",
        write_image=lambda dest, src=src: shutil.copyfile(src, dest), meta=meta,
    )


# --- ui_state -----------------------------------------------------------------------------------------


def _candidate_state_keys(row: dict) -> list[str]:
    keys = list(STATE_KEYS)
    if row["states"].get("checkbox_label") is None:
        keys.remove("checkbox_checked")
    return keys


def _pick_state_keys(cfg: Config, rows: list[dict]) -> list[tuple[dict, list[str]]]:
    """For each row, greedily pick 2 of its available state keys so True/False stay balanced over the suite,
    with a seeded shuffle breaking ties among equally-good candidates."""
    rng = random.Random(f"{cfg.eval.seed}:ui_state")
    yes = no = 0
    picked: list[tuple[dict, list[str]]] = []
    for row in rows:
        candidates = _candidate_state_keys(row)
        rng.shuffle(candidates)

        def cost(key: str) -> int:
            value = row["states"]["checkbox_checked"] if key == "checkbox_checked" else row["states"][key]
            return (yes - no) if value else (no - yes)

        candidates.sort(key=cost)
        chosen = candidates[:2]
        for key in chosen:
            value = row["states"]["checkbox_checked"] if key == "checkbox_checked" else row["states"][key]
            if value:
                yes += 1
            else:
                no += 1
        picked.append((row, chosen))
    return picked


def _state_question(key: str, row: dict) -> dict:
    if key == "checkbox_checked":
        instructions = _checkbox_wording(row["states"]["checkbox_label"])
    else:
        instructions = STATE_WORDING[key]
    return {"type": "noul", "instructions": instructions}


def _ui_state(cfg: Config, n: int) -> list[EvalItem]:
    items = []
    for row, keys in _pick_state_keys(cfg, _base_rows()):
        for key in keys:
            value = row["states"]["checkbox_checked"] if key == "checkbox_checked" else row["states"][key]
            items.append(_raw(f"{row['screen_id']}__{key}", row, _state_question(key, row), value, {"state": key}))
    return materialize(cfg, STATE_INFO, items, n)


# --- ui_page --------------------------------------------------------------------------------------------


def _ui_page(cfg: Config, n: int) -> list[EvalItem]:
    question = {"type": "choice", "instructions": "What kind of page is `img0`?", "criteria": dict(PAGE_CRITERIA)}
    items = [_raw(row["screen_id"], row, question, row["page_type"]) for row in _base_rows()]
    return materialize(cfg, PAGE_INFO, items, n)


# --- ui_click -------------------------------------------------------------------------------------------


def _ui_click(cfg: Config, n: int) -> list[EvalItem]:
    items = []
    for row in _base_rows():
        n_marks = len(row["marks"])
        instructions = f"Goal: {row['goal']['text']}. Which numbered element in `img0` should be clicked?"
        criteria = {str(i): None for i in range(1, n_marks + 1)}
        label = str(row["goal"]["target_mark"])
        items.append(_raw(row["screen_id"], row, {"type": "choice", "instructions": instructions, "criteria": criteria},
                           label, {"goal": row["goal"]["text"], "n_marks": n_marks}))
    return materialize(cfg, CLICK_INFO, items, n)


# --- ui_done --------------------------------------------------------------------------------------------


def _ui_done(cfg: Config, n: int) -> list[EvalItem]:
    items = []
    for row in _done_rows():
        goal_text = row["done"]["goal_text"]
        instructions = f"Goal: {goal_text}. Has this goal already been accomplished on the screen in `img0`?"
        items.append(_raw(row["screen_id"], row, {"type": "noul", "instructions": instructions}, row["done"]["is_done"],
                           {"goal": goal_text}))
    return materialize(cfg, DONE_INFO, items, n)


# --- ui_reason ------------------------------------------------------------------------------------------


def _ui_reason(cfg: Config, n: int) -> list[EvalItem]:
    items = []
    for row in _reason_rows():
        reason = row["reason"]
        question = {"type": "choice", "instructions": reason["question"], "criteria": dict(reason["options"])}
        items.append(_raw(row["screen_id"], row, question, reason["answer"], {"question": reason["question"]}))
    return materialize(cfg, REASON_INFO, items, n)


_SOURCE_NOTE = "synthetic screens rendered from generated HTML; labels exact by construction (lab/NOTES.md entry 49)"
_LICENSE = "generated by this project (Apache-2.0)"

STATE_INFO = SuiteInfo(name="ui_state", qtype="noul", source=_SOURCE_NOTE, license=_LICENSE, backends=("vlm", "frontier"))
PAGE_INFO = SuiteInfo(name="ui_page", qtype="choice", source=_SOURCE_NOTE, license=_LICENSE, backends=("siglip", "vlm", "frontier"))
CLICK_INFO = SuiteInfo(name="ui_click", qtype="choice", source=_SOURCE_NOTE, license=_LICENSE, backends=("vlm", "frontier"))
DONE_INFO = SuiteInfo(name="ui_done", qtype="noul", source=_SOURCE_NOTE, license=_LICENSE, backends=("vlm", "frontier"))
REASON_INFO = SuiteInfo(name="ui_reason", qtype="choice", source=_SOURCE_NOTE, license=_LICENSE, backends=("vlm", "frontier"))

MODULES = {
    "ui_state": SimpleNamespace(INFO=STATE_INFO, build=_ui_state),
    "ui_page": SimpleNamespace(INFO=PAGE_INFO, build=_ui_page),
    "ui_click": SimpleNamespace(INFO=CLICK_INFO, build=_ui_click),
    "ui_done": SimpleNamespace(INFO=DONE_INFO, build=_ui_done),
    "ui_reason": SimpleNamespace(INFO=REASON_INFO, build=_ui_reason),
}

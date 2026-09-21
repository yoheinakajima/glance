"""Offline tests for the synthetic UI screen generator and its five suites (lab/NOTES.md entry 49).

No Chrome, no network: `render_screen`/`render_reason_screen` are pure functions of a seeded spec, exercised
directly; the end-to-end `build()` pipeline is exercised with `ui_screens._render` monkeypatched to write a
tiny blank PNG instead of shelling out to Chrome. All paths are redirected under `tmp_path` so nothing under
the real `.cache/` or `glance/evals/manifests/` is ever read or written.
"""

from __future__ import annotations

import html as html_lib
import json
import re

import pytest
from PIL import Image

from glance.evals.suites import base
from glance.evals.suites import ui_screens as S
from glance.lab import ui_screens as U

INDEX_SAMPLE = list(range(0, 300, 7))  # sparse but covers all 5 page types many times over
N_SCREENS = 60  # multiple of 5; small enough to build fast, big enough to exercise every code path


def _toggle_checked(html: str, label: str) -> bool:
    """Whether the toggle whose visible label is `label` carries the "on" class in `html`."""
    pattern = re.compile(r'<div class="(el toggle(?: on)?)"[^>]*>.*?<span class="toggle-label">'
                          + re.escape(html_lib.escape(label)) + r"</span></div>")
    m = pattern.search(html)
    assert m, f"toggle for {label!r} not found in html"
    return "on" in m.group(1).split()


# --- HTML generation: determinism and ground truth --------------------------------------------------------


def test_html_generation_is_deterministic():
    for i in INDEX_SAMPLE:
        spec1 = U._make_spec(i)
        html1, marks1, goal1 = U.render_screen(spec1, "before")
        spec2 = U._make_spec(i)
        html2, marks2, goal2 = U.render_screen(spec2, "before")
        assert html1 == html2
        assert [m.to_dict() for m in marks1] == [m.to_dict() for m in marks2]
        assert goal1 == goal2


def test_mark_count_in_range_for_every_base_screen():
    for i in range(300):
        spec = U._make_spec(i)
        _, marks, _ = U.render_screen(spec, "before")
        assert 6 <= len(marks) <= 8, (i, spec["page_type"], len(marks))
        assert [m.number for m in marks] == list(range(1, len(marks) + 1))


def test_primary_disabled_reflected_in_html():
    for i in range(300):
        spec = U._make_spec(i)
        html, _, _ = U.render_screen(spec, "before")
        assert (" disabled" in html) == spec["states"]["primary_disabled"]


def test_dialog_open_reflected_in_html():
    for i in range(300):
        spec = U._make_spec(i)
        html, _, _ = U.render_screen(spec, "before")
        assert ('class="modal"' in html) == spec["states"]["dialog_open"]
        assert ('class="backdrop"' in html) == spec["states"]["dialog_open"]


def test_error_shown_reflected_in_html():
    for i in range(300):
        spec = U._make_spec(i)
        html, _, _ = U.render_screen(spec, "before")
        assert ('class="error-banner"' in html) == spec["states"]["error_shown"]


def test_cookie_banner_reflected_in_html():
    for i in range(300):
        spec = U._make_spec(i)
        html, _, _ = U.render_screen(spec, "before")
        assert ('class="cookie-banner"' in html) == spec["states"]["cookie_banner"]


def test_checkbox_checked_reflected_in_html():
    for i in range(300):
        spec = U._make_spec(i)
        html, _, _ = U.render_screen(spec, "before")
        if spec["checkbox"]["present"]:
            assert _toggle_checked(html, spec["checkbox"]["label"]) == spec["checkbox"]["checked"]
        else:
            assert spec["checkbox"]["label"] is None


def test_signed_in_reflected_in_header():
    for i in range(300):
        spec = U._make_spec(i)
        if spec["page_type"] == "sign_in" or spec["states"]["dialog_open"]:
            continue  # no header account mark in these cases
        _, marks, _ = U.render_screen(spec, "before")
        account_mark = marks[0]
        if spec["states"]["signed_in"]:
            assert spec["user_name"] in account_mark.text
        else:
            assert account_mark.text == "Sign in"


GOAL_TARGET_TEXT = {"sign_in": "Sign in", "search_results": "Next", "article": "Share"}


def test_goal_target_mark_exists_and_matches_the_goal_object():
    for i in range(300):
        spec = U._make_spec(i)
        _, marks, goal_mark = U.render_screen(spec, "before")
        assert goal_mark is not None
        target = next(m for m in marks if m.number == goal_mark)
        page_type = spec["page_type"]
        goal_text_lower = spec["goal"]["text"].lower()

        if spec["states"]["dialog_open"]:
            assert target.text in ("Keep editing", "Discard")
            assert ("discard" in goal_text_lower) if target.text == "Discard" else ("keep editing" in goal_text_lower)
        elif page_type == "cart":
            item = spec["content"]["items"][spec["content"]["goal_idx"]]
            assert target.role == "link" and target.text == "Remove"
            assert item in spec["goal"]["text"]
        elif page_type == "settings":
            assert target.text == spec["content"]["label"]
            assert target.text in spec["goal"]["text"]
        else:
            expected = GOAL_TARGET_TEXT[page_type]
            assert target.text == expected
            assert expected.lower() in goal_text_lower


def test_ui_done_eligible_screens_exclude_dialog_open():
    for i in range(300):
        spec = U._make_spec(i)
        assert spec["done_eligible"] == (not spec["states"]["dialog_open"])


def test_reason_screens_have_unambiguous_answers():
    for spec in U._make_reason_specs(90):
        html, marks, options, answer = U.render_reason_screen(spec)
        assert list(options.keys()) == [str(k) for k in range(1, len(marks) + 1)]
        assert answer in options
        if spec["kind"] == "pricing":
            plans = spec["content"]["plans"]
            cheapest = min(p["price"] for p in plans)
            assert sum(1 for p in plans if p["price"] == cheapest) == 1
            assert plans[int(answer) - 1]["price"] == cheapest
        elif spec["kind"] == "stock":
            assert int(answer) - 1 == spec["content"]["oos_idx"]
        else:
            slots = spec["content"]["slots"]
            earliest = min(s["date"] for s in slots)
            assert slots[int(answer) - 1]["date"] == earliest


# --- end-to-end build() + suites, with Chrome faked out ---------------------------------------------------


def _fake_render(html_path, png_path):
    Image.new("RGB", (32, 24), (12, 12, 12)).save(png_path, "PNG")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui_screens_build")
    orig = (U.IMAGES_DIR, U.MANIFEST_PATH, U._render, S.SOURCE, S.IMAGES, base.MANIFEST_DIR)
    U.IMAGES_DIR = tmp / "images"
    U.MANIFEST_PATH = tmp / "manifest.jsonl"
    U._render = _fake_render
    info = U.build(n_screens=N_SCREENS, force=True)
    S.SOURCE = U.MANIFEST_PATH
    S.IMAGES = U.IMAGES_DIR
    base.MANIFEST_DIR = tmp / "suite_manifests"
    rows = [json.loads(line) for line in open(U.MANIFEST_PATH)]
    yield {"tmp": tmp, "info": info, "rows": rows}
    U.IMAGES_DIR, U.MANIFEST_PATH, U._render, S.SOURCE, S.IMAGES, base.MANIFEST_DIR = orig


@pytest.fixture
def suite_cfg(cfg, built):
    cfg.paths.eval_images = str(built["tmp"] / "eval_images")
    return cfg


def test_build_writes_expected_row_counts(built):
    info = built["info"]
    assert info["n_base"] == N_SCREENS
    # each base screen contributes one row; a done-selected screen's "after" variant adds one more row
    assert info["n_rows"] == N_SCREENS + info["n_done_pairs"] + info["n_reason"]
    rows = built["rows"]
    assert sum("goal" in r for r in rows) == N_SCREENS
    assert sum("done" in r for r in rows) == 2 * info["n_done_pairs"]
    assert sum(r.get("done", {}).get("is_done") is True for r in rows) == info["n_done_pairs"]
    assert sum(r.get("done", {}).get("is_done") is False for r in rows) == info["n_done_pairs"]
    assert sum("reason" in r for r in rows) == info["n_reason"]
    for r in rows:
        assert "sha256" in r and len(r["sha256"]) == 64


def test_ui_click_options_are_exactly_1_to_n(suite_cfg):
    items = S.MODULES["ui_click"].build(suite_cfg, 10_000)
    assert items
    for it in items:
        n = len(it.question["criteria"])
        assert list(it.question["criteria"].keys()) == [str(k) for k in range(1, n + 1)]
        assert it.label in it.question["criteria"]


def test_ui_state_yes_no_roughly_balanced(suite_cfg):
    items = S.MODULES["ui_state"].build(suite_cfg, 10_000)
    yes = sum(1 for it in items if it.label)
    no = len(items) - yes
    assert abs(yes - no) <= max(4, round(0.15 * len(items)))


def test_ui_done_yes_no_exactly_balanced(suite_cfg):
    items = S.MODULES["ui_done"].build(suite_cfg, 10_000)
    yes = sum(1 for it in items if it.label is True)
    no = sum(1 for it in items if it.label is False)
    assert yes == no == len(items) // 2


def test_ui_state_checkbox_question_only_on_screens_with_a_checkbox(suite_cfg, built):
    rows_by_id = {r["screen_id"]: r for r in built["rows"]}
    items = S.MODULES["ui_state"].build(suite_cfg, 10_000)
    for it in items:
        if it.meta["state"] == "checkbox_checked":
            screen_id = it.item_id.rsplit("__", 1)[0]
            assert rows_by_id[screen_id]["states"]["checkbox_label"] is not None


def test_five_suites_build_with_exact_wordings(suite_cfg):
    state_items = S.MODULES["ui_state"].build(suite_cfg, 10_000)
    fixed_wordings = {
        "Is a dialog box open on top of the page in `img0`?",
        "Is an error message shown in `img0`?",
        "Is the main button in `img0` disabled (greyed out)?",
        "Is a user signed in on the page in `img0`?",
        "Is a cookie banner visible in `img0`?",
    }
    checkbox_pattern = re.compile(r'^Is the ".+" checkbox checked in `img0`\?$')
    for it in state_items:
        instr = it.question["instructions"]
        assert instr in fixed_wordings or checkbox_pattern.match(instr), instr
        assert it.question["type"] == "noul"

    page_items = S.MODULES["ui_page"].build(suite_cfg, 10_000)
    assert page_items
    for it in page_items:
        assert it.question["instructions"] == "What kind of page is `img0`?"
        assert it.question["type"] == "choice"
        assert set(it.question["criteria"]) == {"sign_in", "search_results", "cart", "settings", "article"}
    assert page_items[0].question["criteria"]["sign_in"] == "A sign-in or log-in page"
    assert page_items[0].question["criteria"]["search_results"] == "A page of search results"
    assert page_items[0].question["criteria"]["cart"] == "A shopping cart or checkout page"
    assert page_items[0].question["criteria"]["settings"] == "A settings or preferences page"
    assert page_items[0].question["criteria"]["article"] == "An article or blog post"

    click_items = S.MODULES["ui_click"].build(suite_cfg, 10_000)
    assert click_items
    for it in click_items:
        assert it.question["type"] == "choice"
        assert it.question["instructions"].startswith("Goal: ")
        assert it.question["instructions"].endswith(". Which numbered element in `img0` should be clicked?")

    done_items = S.MODULES["ui_done"].build(suite_cfg, 10_000)
    assert done_items
    for it in done_items:
        assert it.question["type"] == "noul"
        assert it.question["instructions"].startswith("Goal: ")
        assert it.question["instructions"].endswith(". Has this goal already been accomplished on the screen in `img0`?")

    reason_items = S.MODULES["ui_reason"].build(suite_cfg, 10_000)
    assert reason_items
    known_questions = {
        "Click the cheaper plan.", "Click the cheapest plan.",
        "Which item cannot be ordered right now?", "Which is the earliest available date?",
    }
    for it in reason_items:
        assert it.question["type"] == "choice"
        assert it.question["instructions"] in known_questions
        n = len(it.question["criteria"])
        assert list(it.question["criteria"].keys()) == [str(k) for k in range(1, n + 1)]
        assert it.label in it.question["criteria"]


def test_suite_backends():
    assert S.MODULES["ui_page"].INFO.backends == ("siglip", "vlm", "frontier")
    for name in ("ui_state", "ui_click", "ui_done", "ui_reason"):
        assert S.MODULES[name].INFO.backends == ("vlm", "frontier")


def test_suite_meta_carries_page_type_and_theme(suite_cfg):
    for name in S.MODULES:
        items = S.MODULES[name].build(suite_cfg, 10_000)
        for it in items:
            assert "page_type" in it.meta
            assert "theme" in it.meta


def test_suite_skipped_without_manifest(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(S, "SOURCE", tmp_path / "missing.jsonl")
    monkeypatch.setattr(S, "IMAGES", tmp_path / "missing_images")
    with pytest.raises(base.SuiteSkipped, match="ui_screens"):
        S.MODULES["ui_page"].build(cfg, 4)

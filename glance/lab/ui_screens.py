"""Synthetic UI screens with EXACT ground truth from the generator's parameters (lab/NOTES.md entry 49).

300 screens rendered from self-contained HTML + inline CSS (no external fonts, images or scripts): five page
types (`sign_in`, `search_results`, `cart`, `settings`, `article`), 60 each, with a theme (light/dark), one of
3 system font stacks, an accent colour (6 choices) and a density (comfortable/compact) all drawn from a seed
that is a pure function of the screen's index -- nothing here depends on a model, and no label was assigned by
a human. Every screen records six boolean state toggles (`dialog_open`, `error_shown`, `primary_disabled`,
`checkbox_checked` + its visible label, `signed_in`, `cookie_banner`), 6 to 8 numbered interactive marks
(number, role, visible text, bounding box) and one unambiguous goal sentence naming a single correct mark.
100 of the 300 screens also get an AFTER variant showing that goal visibly accomplished (paired with the
untouched BEFORE screen for `ui_done`). 100 additional dedicated screens (pricing cards, an out-of-stock list,
three appointment slots) carry a one-step reasoning question with its own numbered marks, for `ui_reason`.

Rendering is a real headless Chrome screenshot (`_render`); everything else -- HTML, marks, states, goals -- is
plain deterministic Python with no model or network involved. `build()` writes the PNGs, the HTML files next to
them, and one manifest row per screen to `glance/evals/manifests/ui_screens_source.jsonl`.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..config import PROJECT_ROOT

# --- paths & constants ---------------------------------------------------------------------------------

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CANVAS_W, CANVAS_H = 1024, 768
HEADER_H = 64

IMAGES_DIR = PROJECT_ROOT / ".cache" / "lab_images_ui"
MANIFEST_PATH = PROJECT_ROOT / "glance" / "evals" / "manifests" / "ui_screens_source.jsonl"
SHEETS_DIR = PROJECT_ROOT / "lab" / "sheets_ui"

PAGE_TYPES = ["sign_in", "search_results", "cart", "settings", "article"]
THEMES = ["light", "dark"]
FONT_KEYS = ["sans", "serif", "mono"]
DENSITIES = ["comfortable", "compact"]
ACCENTS = {
    "indigo": "#4f46e5",
    "teal": "#0f766e",
    "amber": "#92400e",
    "rose": "#be123c",
    "emerald": "#15803d",
    "violet": "#6d28d9",
}

SANS_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
SERIF_STACK = "Georgia, 'Times New Roman', Times, serif"
MONO_STACK = "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace"
HEADING_STACKS = {"sans": SANS_STACK, "serif": SERIF_STACK, "mono": MONO_STACK}
BODY_STACKS = {"sans": SANS_STACK, "serif": SERIF_STACK, "mono": SANS_STACK}

LIGHT = {
    "bg": "#eef0f4", "card": "#ffffff", "text": "#15171c", "muted": "#5b6472", "border": "#dfe2e8",
    "input_bg": "#f7f8fa", "danger_bg": "#fdecec", "danger_text": "#a3271e", "danger_border": "#f2b6b0",
    "badge_bg": "#15171c", "badge_text": "#ffffff", "backdrop": "rgba(15,17,20,0.55)", "success_text": "#146c3e",
}
DARK = {
    "bg": "#12141a", "card": "#1b1e27", "text": "#eef0f4", "muted": "#a2a8b6", "border": "#2c303c",
    "input_bg": "#232733", "danger_bg": "#3a1a1c", "danger_text": "#ff8a80", "danger_border": "#5c2326",
    "badge_bg": "#f5f6f8", "badge_text": "#15171c", "backdrop": "rgba(0,0,0,0.65)", "success_text": "#6fd6a0",
}

SITE_NAMES = ["Driftline", "Harborlist", "Maplecrest", "Kindlewood", "Stonebrook Co.", "Fernway"]
NAMES = [
    "Jamie Ortiz", "Priya Shah", "Marcus Lindqvist", "Sofia Bergman", "Devon Clarke", "Elena Petrova",
    "Noah Whitfield", "Aiko Tanaka", "Liam O'Sullivan", "Zara Malik", "Owen Fitzgerald", "Maya Kessler",
    "Theo Bramwell", "Ines Duarte", "Callum Reyes", "Nadia Volkov",
]
PRODUCTS = [
    "Aurora Desk Lamp", "Terra Ceramic Mug", "Cobalt Backpack", "Nimbus Water Bottle", "Solace Reading Chair",
    "Granite Cutting Board", "Lumen Desk Fan", "Echo Table Speaker", "Drift Wireless Mouse", "Haven Throw Blanket",
    "Ridge Hiking Socks", "Cinder Cast Iron Pan", "Willow Wall Clock", "Quartz Phone Stand", "Meadow Scented Candle",
    "Basalt Yoga Mat", "Fable Notebook Set", "Harbor Duffel Bag", "Prairie Wool Scarf", "Cascade Shower Head",
    "Pixel Notebook Stand", "Vantage Bike Helmet", "Amber Reading Lamp", "Coral Beach Towel",
]
ARTICLE_HEADLINES = [
    "The Quiet Rise of Vertical Gardens", "Why Coastal Towns Are Rethinking Their Ferries",
    "A Beginner's Guide to Mechanical Keyboards", "Inside the Push for Four-Day Workweeks",
    "How Small Bakeries Are Surviving Rising Flour Prices", "The Long Road Back to Analog Photography",
    "What Makes a Neighborhood Walkable", "Five Lessons From a Decade of Urban Beekeeping",
    "The Hidden Cost of Free Shipping", "Why Board Games Are Having a Moment Again",
    "Inside the World of Competitive Speed Cubing", "How Night Markets Are Reshaping City Food Scenes",
    "The Slow Return of the Corner Bookstore", "Why Everyone Is Talking About Soil Health",
    "A Short History of the Modern Umbrella", "The Case for Fixing Instead of Replacing",
    "How Community Gardens Changed One Block", "Why Train Travel Is Making a Comeback",
    "The Science Behind a Good Night's Sleep", "Inside the Renaissance of Hand-Bound Books",
]
SETTINGS_LABELS = [
    "Email notifications", "Push notifications", "Marketing emails", "Weekly digest",
    "Two-factor authentication", "Dark mode", "Auto-save drafts", "Public profile",
]
EMAIL_LOCALS = ["jordan.kim", "alex.rivera", "morgan.lee", "sam.patel", "casey.wu", "riley.nguyen"]
EMAIL_DOMAINS = ["mailbox.test", "inboxsample.test", "workmail.test"]
PLAN_NAMES = ["Starter", "Growth", "Scale", "Basic", "Plus", "Pro", "Essential", "Team"]
PRICE_POOL = [9, 12, 15, 19, 24, 29, 35, 39, 49, 59, 79, 99]
FEATURE_POOL = [
    "10 active projects", "Unlimited exports", "Priority email support", "Up to 5 team seats",
    "Custom domain", "Advanced analytics", "Version history", "Single sign-on", "Audit log access",
]
SLOT_TIMES = ["8:00 AM", "9:00 AM", "10:30 AM", "1:00 PM", "2:30 PM", "4:00 PM"]
SLOT_ANCHOR = date(2025, 10, 6)  # a Monday, used only as a stable calendar reference

ROW_SIZES = {
    "comfortable": {"input": 64, "button": 52, "link": 32, "toggle": 40, "product": 88, "gap": 16},
    "compact": {"input": 52, "button": 44, "link": 28, "toggle": 34, "product": 72, "gap": 10},
}


def esc(value: Any) -> str:
    return html_lib.escape(str(value))


# --- marks -----------------------------------------------------------------------------------------------


@dataclass
class Mark:
    number: int
    role: str
    text: str
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "role": self.role, "text": self.text, "box": list(self.box)}


class _Doc:
    """Accumulates marks + HTML snippets for one screen, in visual (top-to-bottom) order."""

    def __init__(self) -> None:
        self.marks: list[Mark] = []
        self.html: list[str] = []
        self.goal_mark: int | None = None

    def add(self, role: str, text: str, x: int, y: int, w: int, h: int, snippet: str, *, is_goal: bool = False) -> int:
        number = len(self.marks) + 1
        self.marks.append(Mark(number, role, text, (x, y, w, h)))
        self.html.append(snippet)
        self.html.append(_badge_html(number, x, y))
        if is_goal:
            self.goal_mark = number
        return number

    def add_plain(self, snippet: str) -> None:
        self.html.append(snippet)


# --- HTML element emitters ---------------------------------------------------------------------------------


def _btn_snippet(text: str, x: int, y: int, w: int, h: int, *, primary: bool = False, disabled: bool = False) -> str:
    cls = "el btn" + (" primary" if primary else "") + (" disabled" if disabled else "")
    dis = " disabled" if disabled else ""
    return f'<button class="{cls}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px"{dis}>{esc(text)}</button>'


def _link_snippet(text: str, x: int, y: int, w: int, h: int) -> str:
    return f'<a class="el link" href="#" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">{esc(text)}</a>'


def _input_snippet(label: str, value: str, x: int, y: int, w: int, h: int) -> str:
    return (
        f'<div class="el field" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
        f'<span class="field-label">{esc(label)}</span><span class="field-value">{esc(value)}</span></div>'
    )


def _toggle_snippet(label: str, checked: bool, x: int, y: int, w: int, h: int) -> str:
    cls = "el toggle" + (" on" if checked else "")
    return (
        f'<div class="{cls}" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px">'
        f'<span class="toggle-track"><span class="toggle-thumb"></span></span>'
        f'<span class="toggle-label">{esc(label)}</span></div>'
    )


def _badge_html(number: int, x: int, y: int) -> str:
    bx, by = max(x - 11, 2), max(y - 11, 2)
    return f'<div class="badge" style="left:{bx}px;top:{by}px">{number}</div>'


def _error_banner(text: str, x: int, y: int, w: int) -> str:
    return f'<div class="error-banner" style="left:{x}px;top:{y}px;width:{w}px;height:48px">&#9888; {esc(text)}</div>'


def _toast(text: str, x: int, y: int, w: int) -> str:
    return f'<div class="toast" style="left:{x}px;top:{y}px;width:{w}px;height:40px">{esc(text)}</div>'


def _add_cookie_banner(doc: _Doc, accent: str) -> None:
    doc.add_plain(
        '<div class="cookie-banner">'
        "We use cookies to keep this demo working the way you expect."
        "</div>"
    )
    doc.add("button", "Accept", CANVAS_W - 220, CANVAS_H - 50, 88, 36, _btn_snippet("Accept", CANVAS_W - 220, CANVAS_H - 50, 88, 36, primary=True))
    doc.add("button", "Decline", CANVAS_W - 120, CANVAS_H - 50, 96, 36, _btn_snippet("Decline", CANVAS_W - 120, CANVAS_H - 50, 96, 36))


def _add_header_account(doc: _Doc, spec: dict, signed_in: bool) -> int:
    x, y, w, h = CANVAS_W - 210, 16, 178, 32
    if signed_in:
        initials = "".join(part[0] for part in spec["user_name"].split()[:2]).upper()
        text = f"{initials} {spec['user_name']}"
    else:
        text = "Sign in"
    return doc.add("link", text, x, y, w, h, _link_snippet(text, x, y, w, h))


# --- CSS -----------------------------------------------------------------------------------------------


def _css(theme: str, accent_hex: str, font_key: str) -> str:
    t = LIGHT if theme == "light" else DARK
    body_font = BODY_STACKS[font_key]
    heading_font = HEADING_STACKS[font_key]
    return f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width: {CANVAS_W}px; height: {CANVAS_H}px; overflow: hidden; background: {t['bg']}; }}
body {{ font-family: {body_font}; color: {t['text']}; font-size: 16px; }}
.page {{ position: relative; width: {CANVAS_W}px; height: {CANVAS_H}px; }}
h1, h2, h3 {{ font-family: {heading_font}; font-weight: 700; }}
.header {{ position: absolute; left: 0; top: 0; width: {CANVAS_W}px; height: {HEADER_H}px;
  background: {t['card']}; border-bottom: 1px solid {t['border']}; }}
.logo {{ position: absolute; left: 32px; top: 18px; font-family: {heading_font}; font-size: 19px; font-weight: 700; color: {t['text']}; }}
.el {{ position: absolute; display: flex; align-items: center; font-size: 15px; font-family: {body_font}; z-index: 42; }}
.btn {{ justify-content: center; border-radius: 8px; background: {t['input_bg']}; border: 1px solid {t['border']};
  color: {t['text']}; font-weight: 600; }}
.btn.primary {{ background: {accent_hex}; border-color: {accent_hex}; color: #ffffff; }}
.btn.disabled {{ opacity: 0.45; }}
.link {{ color: {accent_hex}; text-decoration: none; font-weight: 600; padding-left: 14px; }}
.field {{ flex-direction: column; align-items: flex-start; justify-content: center; background: {t['input_bg']};
  border: 1px solid {t['border']}; border-radius: 8px; padding: 14px 16px 8px 16px; gap: 4px; }}
.field-label {{ font-size: 13px; color: {t['muted']}; }}
.field-value {{ font-size: 16px; color: {t['text']}; }}
.toggle {{ gap: 12px; padding-left: 6px; }}
.toggle-track {{ width: 38px; height: 22px; border-radius: 11px; background: {t['border']}; position: relative; flex: none; }}
.toggle.on .toggle-track {{ background: {accent_hex}; }}
.toggle-thumb {{ position: absolute; top: 2px; left: 2px; width: 18px; height: 18px; border-radius: 50%; background: #fff; }}
.toggle.on .toggle-thumb {{ left: 18px; }}
.toggle-label {{ font-size: 15px; color: {t['text']}; }}
.badge {{ position: absolute; width: 22px; height: 22px; border-radius: 50%; background: {t['badge_bg']};
  color: {t['badge_text']}; font-size: 12px; font-weight: 700; display: flex; align-items: center; justify-content: center;
  z-index: 60; box-shadow: 0 1px 3px rgba(0,0,0,0.35); font-family: {SANS_STACK}; }}
.card {{ position: absolute; background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; }}
.error-banner {{ position: absolute; background: {t['danger_bg']}; border: 1px solid {t['danger_border']}; color: {t['danger_text']};
  border-radius: 8px; display: flex; align-items: center; font-size: 15px; font-weight: 600; padding: 0 16px; }}
.cookie-banner {{ position: absolute; left: 0; bottom: 0; width: {CANVAS_W}px; height: 68px; background: {t['card']};
  border-top: 1px solid {t['border']}; display: flex; align-items: center; padding: 0 32px; font-size: 14px; color: {t['muted']}; }}
.backdrop {{ position: absolute; inset: 0; background: {t['backdrop']}; z-index: 40; }}
.modal {{ position: absolute; background: {t['card']}; border-radius: 14px; border: 1px solid {t['border']};
  box-shadow: 0 12px 32px rgba(0,0,0,0.4); z-index: 41; }}
.toast {{ position: absolute; background: {t['text']}; color: {t['bg']}; border-radius: 8px; font-size: 14px;
  font-weight: 700; display: flex; align-items: center; padding: 0 16px; z-index: 30; }}
.muted {{ color: {t['muted']}; }}
.success {{ color: {t['success_text']}; font-weight: 700; font-size: 14px; }}
.avatar {{ width: 56px; height: 56px; border-radius: 50%; background: {accent_hex}; color: #fff; font-weight: 700;
  font-size: 20px; display: flex; align-items: center; justify-content: center; }}
p {{ font-size: 15px; line-height: 1.5; color: {t['muted']}; }}
"""


def _wrap_html(spec: dict, body: str) -> str:
    css = _css(spec["theme"], ACCENTS[spec["accent"]], spec["font"])
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<style>{css}</style></head><body><div class=\"page\">{body}</div></body></html>"
    )


def _pick_optionals(rng: random.Random, allow: bool, candidates: list[str], max_n: int = 2) -> set[str]:
    chosen: set[str] = set()
    if not allow:
        return chosen
    for name in candidates:
        if len(chosen) >= max_n:
            break
        if rng.random() < 0.5:
            chosen.add(name)
    return chosen


# --- spec generation (all randomness lives here; rendering below is pure) ---------------------------------


def _make_spec(index: int, per_type: int = 60) -> dict[str, Any]:
    page_type = PAGE_TYPES[index // per_type]
    rng = random.Random(f"glance-ui-screens-v1:{index}")
    dialog_open = rng.random() < 0.22
    if dialog_open:
        cookie_banner = False
        error_shown = False
    else:
        cookie_banner = rng.random() < 0.3
        error_shown = rng.random() < 0.25

    spec: dict[str, Any] = {
        "index": index,
        "screen_id": f"{page_type}_{index:03d}",
        "page_type": page_type,
        "theme": rng.choice(THEMES),
        "font": rng.choice(FONT_KEYS),
        "accent": rng.choice(list(ACCENTS)),
        "density": rng.choice(DENSITIES),
        "site_name": rng.choice(SITE_NAMES),
        "user_name": rng.choice(NAMES),
        "states": {
            "dialog_open": dialog_open,
            "error_shown": error_shown,
            "primary_disabled": rng.random() < 0.2,
            "signed_in": False if page_type == "sign_in" else rng.random() < 0.5,
            "cookie_banner": cookie_banner,
        },
        "content": {},
        "checkbox": {"present": False, "label": None, "checked": None},
        "goal": {},
    }

    if dialog_open:
        spec["checkbox"] = {"present": True, "label": "Don't show this again", "checked": rng.random() < 0.5}
        if rng.random() < 0.5:
            spec["goal"] = {"text": "Discard the unsaved changes", "target_key": "confirm"}
        else:
            spec["goal"] = {"text": "Keep editing without discarding your changes", "target_key": "cancel"}
        spec["done_eligible"] = False
    else:
        _CONTENT_BUILDERS[page_type](spec, rng, allow_optionals=not cookie_banner)
        spec["done_eligible"] = True
    return spec


def _build_sign_in(spec: dict, rng: random.Random, allow_optionals: bool) -> None:
    email = f"{rng.choice(EMAIL_LOCALS)}@{rng.choice(EMAIL_DOMAINS)}"
    opts = _pick_optionals(rng, allow_optionals, ["sso", "lang"])
    spec["content"] = {"email": email, "show_sso": "sso" in opts, "show_lang": "lang" in opts}
    spec["checkbox"] = {"present": True, "label": "Remember me", "checked": rng.random() < 0.5}
    spec["goal"] = {"text": "Sign in with the email and password already typed", "target_key": "sign_in_button"}


def _build_search_results(spec: dict, rng: random.Random, allow_optionals: bool) -> None:
    products = rng.sample(PRODUCTS, 2)
    opts = _pick_optionals(rng, allow_optionals, ["checkbox", "sort", "filters"], max_n=2)
    total_pages = rng.choice([3, 4, 5])
    spec["content"] = {
        "products": products, "query": products[0].split(" ", 1)[1].lower(), "total_pages": total_pages,
        "show_sort": "sort" in opts, "show_filters": "filters" in opts,
    }
    if "checkbox" in opts:
        spec["checkbox"] = {"present": True, "label": "Save this search", "checked": rng.random() < 0.5}
    spec["goal"] = {"text": "Go to the next page of results", "target_key": "next_page"}


def _build_cart(spec: dict, rng: random.Random, allow_optionals: bool) -> None:
    items = rng.sample(PRODUCTS, 3)
    prices = [round(rng.uniform(9.5, 89.5), 2) for _ in items]
    goal_idx = rng.randrange(3)
    opts = _pick_optionals(rng, allow_optionals, ["checkbox", "continue"], max_n=2)
    spec["content"] = {"items": items, "prices": prices, "goal_idx": goal_idx, "show_continue": "continue" in opts}
    if "checkbox" in opts:
        spec["checkbox"] = {"present": True, "label": "Gift wrap this order", "checked": rng.random() < 0.5}
    spec["goal"] = {"text": f"Remove the {items[goal_idx]} from the cart", "target_key": f"remove_{goal_idx}"}


def _build_settings(spec: dict, rng: random.Random, allow_optionals: bool) -> None:
    label = rng.choice(SETTINGS_LABELS)
    checked = rng.random() < 0.5
    secondary_labels = rng.sample([label_ for label_ in SETTINGS_LABELS if label_ != label], 2)
    opts = _pick_optionals(rng, allow_optionals, ["secondary2", "delete"], max_n=2)
    spec["content"] = {
        "label": label, "secondary_labels": secondary_labels,
        "show_secondary2": "secondary2" in opts, "show_delete": "delete" in opts,
        "display_name": rng.choice(NAMES),
    }
    spec["checkbox"] = {"present": True, "label": label, "checked": checked}
    action = "off" if checked else "on"
    spec["goal"] = {"text": f'Turn {action} "{label}" notifications', "target_key": "goal_toggle"}


def _build_article(spec: dict, rng: random.Random, allow_optionals: bool) -> None:
    headline = rng.choice(ARTICLE_HEADLINES)
    author = rng.choice(NAMES)
    related = rng.choice([h for h in ARTICLE_HEADLINES if h != headline])
    opts = _pick_optionals(rng, allow_optionals, ["checkbox", "author_link", "related"], max_n=2)
    spec["content"] = {
        "headline": headline, "author": author, "related": related,
        "show_author_link": "author_link" in opts, "show_related": "related" in opts,
    }
    if "checkbox" in opts:
        spec["checkbox"] = {"present": True, "label": "Notify me of replies", "checked": rng.random() < 0.5}
    spec["goal"] = {"text": "Share this article", "target_key": "share_button"}


_CONTENT_BUILDERS: dict[str, Callable[[dict, random.Random, bool], None]] = {
    "sign_in": _build_sign_in, "search_results": _build_search_results, "cart": _build_cart,
    "settings": _build_settings, "article": _build_article,
}


# --- rendering (pure functions of spec + variant) ------------------------------------------------------


def render_screen(spec: dict, variant: str = "before") -> tuple[str, list[Mark], int | None]:
    """Deterministic HTML + marks for `spec` at `variant` ("before" or "after"). No randomness here."""
    doc = _Doc()
    doc.add_plain(f'<div class="header"><div class="logo">{esc(spec["site_name"])}</div></div>')

    if spec["states"]["dialog_open"]:
        _render_dialog(doc, spec)
    else:
        y = HEADER_H + 26
        _PAGE_RENDERERS[spec["page_type"]](doc, spec, variant, y)
        if spec["states"]["cookie_banner"]:
            _add_cookie_banner(doc, ACCENTS[spec["accent"]])

    return _wrap_html(spec, "".join(doc.html)), doc.marks, doc.goal_mark


def _render_dialog(doc: _Doc, spec: dict) -> None:
    doc.add_plain('<div class="backdrop"></div>')
    mx, my, mw, mh = 272, 184, 480, 400
    doc.add_plain(f'<div class="modal" style="left:{mx}px;top:{my}px;width:{mw}px;height:{mh}px">'
                   f'<h2 style="position:absolute;left:28px;top:24px;width:{mw - 56}px;font-size:20px">Discard unsaved changes?</h2>'
                   f'<p style="position:absolute;left:28px;top:64px;width:{mw - 56}px">You have unsaved changes on this page. '
                   f'If you leave now, they will be lost.</p></div>')
    close_x, close_y = mx + mw - 44, my + 16
    doc.add("button", "Close", close_x, close_y, 28, 28, _btn_snippet("✕", close_x, close_y, 28, 28))
    cb = spec["checkbox"]
    cb_y = my + 128
    doc.add("toggle", cb["label"], mx + 28, cb_y, 220, 40, _toggle_snippet(cb["label"], cb["checked"], mx + 28, cb_y, 220, 40))
    learn_y = my + 176
    doc.add("link", "Learn more", mx + 28, learn_y, 140, 28, _link_snippet("Learn more", mx + 28, learn_y, 140, 28))
    btn_y = my + mh - 72
    cancel_x, confirm_x = mx + 28, mx + 28 + 160
    doc.add("button", "Keep editing", cancel_x, btn_y, 150, 48,
            _btn_snippet("Keep editing", cancel_x, btn_y, 150, 48),
            is_goal=spec["goal"]["target_key"] == "cancel")
    disabled = spec["states"]["primary_disabled"]
    doc.add("button", "Discard", confirm_x, btn_y, 150, 48,
            _btn_snippet("Discard", confirm_x, btn_y, 150, 48, primary=True, disabled=disabled),
            is_goal=spec["goal"]["target_key"] == "confirm")
    support_y = my + mh - 20
    doc.add("link", "Contact support", mx + 28, support_y, 160, 24, _link_snippet("Contact support", mx + 28, support_y, 160, 24))


def _rowsizes(spec: dict) -> dict:
    return ROW_SIZES[spec["density"]]


def _render_sign_in(doc: _Doc, spec: dict, variant: str, y: int) -> None:
    rs = _rowsizes(spec)
    x, w = 312, 400

    if variant == "after":
        card_y = y + 20
        card_h = 330
        initials = "".join(part[0] for part in spec["user_name"].split()[:2]).upper()
        doc.add_plain(
            f'<div class="card" style="left:{x}px;top:{card_y}px;width:{w}px;height:{card_h}px;padding:32px">'
            f'<div class="avatar" style="position:absolute;left:32px;top:32px">{esc(initials)}</div>'
            f'<h2 style="position:absolute;left:32px;top:104px;width:{w - 64}px;font-size:19px;line-height:1.3">'
            f'You are signed in as {esc(spec["user_name"])}</h2>'
            f'<p style="position:absolute;left:32px;top:172px;width:{w - 64}px">Your session will stay active on this device.</p>'
            "</div>"
        )
        btn_y = card_y + 220
        doc.add("button", "Continue", x + 32, btn_y, w - 64, rs["button"],
                 _btn_snippet("Continue", x + 32, btn_y, w - 64, rs["button"], primary=True))
        link_y = btn_y + rs["button"] + 12
        doc.add("link", "Sign out", x + 32, link_y, 100, rs["link"], _link_snippet("Sign out", x + 32, link_y, 100, rs["link"]))
        return

    content = spec["content"]
    doc.add("input", "Email address", x, y, w, rs["input"], _input_snippet("Email address", content["email"], x, y, w, rs["input"]))
    y += rs["input"] + rs["gap"]
    doc.add("input", "Password", x, y, w, rs["input"], _input_snippet("Password", "•" * 10, x, y, w, rs["input"]))
    y += rs["input"] + rs["gap"]
    if spec["states"]["error_shown"]:
        doc.add_plain(_error_banner("Incorrect email or password. Please try again.", x, y, w))
        y += 48 + rs["gap"]
    cb = spec["checkbox"]
    doc.add("toggle", cb["label"], x, y, 200, rs["toggle"], _toggle_snippet(cb["label"], cb["checked"], x, y, 200, rs["toggle"]))
    fp_w = 150
    doc.add("link", "Forgot password?", x + w - fp_w, y + (rs["toggle"] - rs["link"]) // 2, fp_w, rs["link"],
             _link_snippet("Forgot password?", x + w - fp_w, y + (rs["toggle"] - rs["link"]) // 2, fp_w, rs["link"]))
    y += rs["toggle"] + rs["gap"]
    disabled = spec["states"]["primary_disabled"]
    doc.add("button", "Sign in", x, y, w, rs["button"], _btn_snippet("Sign in", x, y, w, rs["button"], primary=True, disabled=disabled),
             is_goal=True)
    y += rs["button"] + rs["gap"]
    if content["show_sso"]:
        doc.add("button", "Continue with SSO", x, y, w, rs["button"], _btn_snippet("Continue with SSO", x, y, w, rs["button"]))
        y += rs["button"] + rs["gap"]
    doc.add("link", "Create an account", x, y, 200, rs["link"], _link_snippet("Create an account", x, y, 200, rs["link"]))
    if content["show_lang"]:
        ly = CANVAS_H - 40
        doc.add("link", "Language: English", x, ly, 200, rs["link"], _link_snippet("Language: English", x, ly, 200, rs["link"]))


def _render_search_results(doc: _Doc, spec: dict, variant: str, y: int) -> None:
    rs = _rowsizes(spec)
    x, w = 64, 896
    _add_header_account(doc, spec, spec["states"]["signed_in"])
    content = spec["content"]
    doc.add("input", "Search", x, y, 380, rs["input"], _input_snippet("Search", content["query"], x, y, 380, rs["input"]))
    disabled = spec["states"]["primary_disabled"]
    doc.add("button", "Search", x + 396, y, 120, rs["input"], _btn_snippet("Search", x + 396, y, 120, rs["input"], primary=True, disabled=disabled))
    if content["show_sort"]:
        doc.add("button", "Sort: Price low to high", x + 532, y, 220, rs["input"],
                 _btn_snippet("Sort: Price low to high", x + 532, y, 220, rs["input"]))
    if content["show_filters"]:
        doc.add("link", "Filters", x + 760, y + (rs["input"] - rs["link"]) // 2, 100, rs["link"],
                 _link_snippet("Filters", x + 760, y + (rs["input"] - rs["link"]) // 2, 100, rs["link"]))
    y += rs["input"] + rs["gap"] + 8
    if spec["states"]["error_shown"]:
        doc.add_plain(_error_banner("We couldn't load all results. Please try again.", x, y, w))
        y += 48 + rs["gap"]

    cb = spec["checkbox"]
    if cb["present"]:
        doc.add("toggle", cb["label"], x, y, 220, rs["toggle"], _toggle_snippet(cb["label"], cb["checked"], x, y, 220, rs["toggle"]))
        y += rs["toggle"] + rs["gap"]

    for product in content["products"]:
        doc.add_plain(f'<div class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{rs["product"]}px">'
                       f'<span style="position:absolute;left:20px;top:16px;font-weight:700">{esc(product)}</span>'
                       f'<span class="muted" style="position:absolute;left:20px;top:{rs["product"] - 30}px">In stock</span></div>')
        bx = x + w - 160
        by = y + (rs["product"] - rs["button"]) // 2
        doc.add("button", "Add to cart", bx, by, 130, rs["button"], _btn_snippet("Add to cart", bx, by, 130, rs["button"]))
        y += rs["product"] + 12

    page = 1 if variant == "before" else 2
    total = content["total_pages"]
    doc.add_plain(f'<span class="muted" style="position:absolute;left:{x}px;top:{y + 10}px">Page {page} of {total}</span>')
    nx = x + w - 100
    doc.add("pagination", "Next", nx, y, 100, rs["button"], _btn_snippet("Next", nx, y, 100, rs["button"]), is_goal=True)


def _render_cart(doc: _Doc, spec: dict, variant: str, y: int) -> None:
    rs = _rowsizes(spec)
    x, w = 64, 896
    _add_header_account(doc, spec, spec["states"]["signed_in"])
    content = spec["content"]
    if spec["states"]["error_shown"]:
        doc.add_plain(_error_banner("One item in your cart is no longer available.", x, y, w))
        y += 48 + rs["gap"]

    items = content["items"]
    prices = content["prices"]
    goal_idx = content["goal_idx"]
    removed_name = items[goal_idx] if variant == "after" else None
    if removed_name is not None:
        doc.add_plain(_toast(f"Removed {removed_name} from cart.", x, y, w))
        y += 40 + rs["gap"]

    running_total = 0.0
    for i, (item, price) in enumerate(zip(items, prices)):
        if variant == "after" and i == goal_idx:
            continue
        running_total += price
        doc.add_plain(f'<div class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{rs["product"]}px">'
                       f'<span style="position:absolute;left:20px;top:16px;font-weight:700">{esc(item)}</span>'
                       f'<span class="muted" style="position:absolute;left:20px;top:{rs["product"] - 30}px">Qty: 1 &middot; ${price:.2f}</span></div>')
        rx = x + w - 120
        ry = y + (rs["product"] - rs["link"]) // 2
        doc.add("link", "Remove", rx, ry, 90, rs["link"], _link_snippet("Remove", rx, ry, 90, rs["link"]), is_goal=(variant == "before" and i == goal_idx))
        y += rs["product"] + 12

    cb = spec["checkbox"]
    if cb["present"]:
        doc.add("toggle", cb["label"], x, y, 220, rs["toggle"], _toggle_snippet(cb["label"], cb["checked"], x, y, 220, rs["toggle"]))
        y += rs["toggle"] + rs["gap"]

    doc.add("input", "Promo code", x, y, 260, rs["input"], _input_snippet("Promo code", "", x, y, 260, rs["input"]))
    doc.add_plain(f'<span style="position:absolute;left:{x + 296}px;top:{y + rs["input"] // 2 - 10}px;font-weight:700">'
                   f'Subtotal: ${running_total:.2f}</span>')
    if content["show_continue"]:
        cont_x = x + 560
        doc.add("link", "Continue shopping", cont_x, y + (rs["input"] - rs["link"]) // 2, 170, rs["link"],
                 _link_snippet("Continue shopping", cont_x, y + (rs["input"] - rs["link"]) // 2, 170, rs["link"]))
    y += rs["input"] + rs["gap"]
    disabled = spec["states"]["primary_disabled"]
    doc.add("button", "Checkout", x, y, 200, rs["button"], _btn_snippet("Checkout", x, y, 200, rs["button"], primary=True, disabled=disabled))


def _render_settings(doc: _Doc, spec: dict, variant: str, y: int) -> None:
    rs = _rowsizes(spec)
    x, w = 64, 500
    _add_header_account(doc, spec, spec["states"]["signed_in"])
    content = spec["content"]
    if spec["states"]["error_shown"]:
        doc.add_plain(_error_banner("We couldn't save your changes. Please try again.", x, y, w))
        y += 48 + rs["gap"]

    doc.add("input", "Display name", x, y, w, rs["input"], _input_snippet("Display name", content["display_name"], x, y, w, rs["input"]))
    y += rs["input"] + rs["gap"]

    checked = spec["checkbox"]["checked"]
    if variant == "after":
        checked = not checked
    label = content["label"]
    doc.add("toggle", label, x, y, 260, rs["toggle"], _toggle_snippet(label, checked, x, y, 260, rs["toggle"]), is_goal=True)
    y += rs["toggle"] + rs["gap"]

    sec1 = content["secondary_labels"][0]
    doc.add("toggle", sec1, x, y, 260, rs["toggle"], _toggle_snippet(sec1, False, x, y, 260, rs["toggle"]))
    y += rs["toggle"] + rs["gap"]
    if content["show_secondary2"]:
        sec2 = content["secondary_labels"][1]
        doc.add("toggle", sec2, x, y, 260, rs["toggle"], _toggle_snippet(sec2, True, x, y, 260, rs["toggle"]))
        y += rs["toggle"] + rs["gap"]

    disabled = spec["states"]["primary_disabled"]
    doc.add("button", "Save changes", x, y, 200, rs["button"], _btn_snippet("Save changes", x, y, 200, rs["button"], primary=True, disabled=disabled))
    if variant == "after":
        doc.add_plain(f'<span class="success" style="position:absolute;left:{x + 216}px;top:{y + rs["button"] // 2 - 8}px">&#10003; Saved</span>')
    y += rs["button"] + rs["gap"]
    doc.add("link", "Change password", x, y, 180, rs["link"], _link_snippet("Change password", x, y, 180, rs["link"]))
    if content["show_delete"]:
        y += rs["link"] + rs["gap"]
        doc.add("link", "Delete account", x, y, 160, rs["link"], _link_snippet("Delete account", x, y, 160, rs["link"]))


def _render_article(doc: _Doc, spec: dict, variant: str, y: int) -> None:
    rs = _rowsizes(spec)
    x, w = 64, 720
    _add_header_account(doc, spec, spec["states"]["signed_in"])
    content = spec["content"]
    doc.add_plain(f'<h1 style="position:absolute;left:{x}px;top:{y}px;width:{w}px;font-size:26px;line-height:1.25">{esc(content["headline"])}</h1>')
    y += 82  # reserves room for a headline that wraps to two lines (the longest pool entry never wraps to three)
    doc.add_plain(f'<span class="muted" style="position:absolute;left:{x}px;top:{y}px">By {esc(content["author"])} &middot; 6 min read</span>')
    y += 30
    if content["show_author_link"]:
        doc.add("link", content["author"], x, y, 200, rs["link"], _link_snippet(content["author"], x, y, 200, rs["link"]))
        y += rs["link"] + rs["gap"]
    if spec["states"]["error_shown"]:
        doc.add_plain(_error_banner("We couldn't post your comment. Please try again.", x, y, w))
        y += 48 + rs["gap"]

    doc.add_plain(
        f'<p style="position:absolute;left:{x}px;top:{y}px;width:{w}px">Over the last few years, a quiet shift has been '
        "changing how people think about small spaces and shared resources, one modest project at a time.</p>"
    )
    y += 70

    share_x = x + w - 200
    if variant == "after":
        doc.add("button", "Copied!", share_x, y, 90, rs["button"], _btn_snippet("Copied!", share_x, y, 90, rs["button"]))
    else:
        doc.add("button", "Share", share_x, y, 90, rs["button"], _btn_snippet("Share", share_x, y, 90, rs["button"]), is_goal=True)
    save_x = share_x + 100
    doc.add("button", "Save", save_x, y, 90, rs["button"], _btn_snippet("Save", save_x, y, 90, rs["button"]))
    y += rs["button"] + rs["gap"]
    if variant == "after":
        doc.add_plain(_toast("Link copied to clipboard.", share_x - 20, y, 220))
        y += 40 + rs["gap"]

    if content["show_related"]:
        doc.add("link", content["related"], x, y, w, rs["link"], _link_snippet(content["related"], x, y, w, rs["link"]))
        y += rs["link"] + rs["gap"]

    cb = spec["checkbox"]
    if cb["present"]:
        doc.add("toggle", cb["label"], x, y, 260, rs["toggle"], _toggle_snippet(cb["label"], cb["checked"], x, y, 260, rs["toggle"]))
        y += rs["toggle"] + rs["gap"]

    doc.add("input", "Add a comment", x, y, 480, rs["input"], _input_snippet("Add a comment", "", x, y, 480, rs["input"]))
    doc.add("button", "Post", x + 500, y, 100, rs["input"], _btn_snippet("Post", x + 500, y, 100, rs["input"]))
    y += rs["input"] + rs["gap"]
    disabled = spec["states"]["primary_disabled"]
    doc.add("button", "Subscribe", x, y, 220, rs["button"], _btn_snippet("Subscribe", x, y, 220, rs["button"], primary=True, disabled=disabled))


_PAGE_RENDERERS: dict[str, Callable[[_Doc, dict, str, int], None]] = {
    "sign_in": _render_sign_in, "search_results": _render_search_results, "cart": _render_cart,
    "settings": _render_settings, "article": _render_article,
}


# --- reasoning screens -----------------------------------------------------------------------------------


def _split_evenly(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _build_reason_pricing(spec: dict, rng: random.Random) -> None:
    n_plans = rng.choice([2, 3])
    names = rng.sample(PLAN_NAMES, n_plans)
    prices = rng.sample(PRICE_POOL, n_plans)
    features = rng.sample(FEATURE_POOL, 3)
    plans = [{"name": names[i], "price": prices[i]} for i in range(n_plans)]
    rng.shuffle(plans)
    spec["content"] = {"plans": plans, "features": features}
    spec["question"] = "Click the cheaper plan." if n_plans == 2 else "Click the cheapest plan."
    spec["answer_idx"] = min(range(n_plans), key=lambda k: plans[k]["price"])


def _build_reason_stock(spec: dict, rng: random.Random) -> None:
    n_items = rng.choice([4, 5, 6])
    items = rng.sample(PRODUCTS, n_items)
    prices = [round(rng.uniform(8, 120), 2) for _ in items]
    oos_idx = rng.randrange(n_items)
    spec["content"] = {"items": items, "prices": prices, "oos_idx": oos_idx}
    spec["question"] = "Which item cannot be ordered right now?"
    spec["answer_idx"] = oos_idx


def _build_reason_slots(spec: dict, rng: random.Random) -> None:
    offsets = rng.sample(range(1, 25), 3)
    times = rng.sample(SLOT_TIMES, 3)
    slots = []
    for off, tm in zip(offsets, times):
        d = SLOT_ANCHOR + timedelta(days=off)
        slots.append({"date": d.isoformat(), "label": f"{d.strftime('%a, %b')} {d.day} &middot; {tm}"})
    order = list(range(3))
    rng.shuffle(order)
    display = [slots[i] for i in order]
    spec["content"] = {"slots": display}
    spec["question"] = "Which is the earliest available date?"
    spec["answer_idx"] = min(range(3), key=lambda k: display[k]["date"])


_REASON_BUILDERS: dict[str, Callable[[dict, random.Random], None]] = {
    "pricing": _build_reason_pricing, "stock": _build_reason_stock, "slots": _build_reason_slots,
}


def _make_reason_specs(n: int) -> list[dict]:
    kinds = ["pricing", "stock", "slots"]
    counts = _split_evenly(n, 3)
    specs = []
    for kind, count in zip(kinds, counts):
        for i in range(count):
            rng = random.Random(f"glance-ui-reason-v1:{kind}:{i}")
            spec: dict[str, Any] = {
                "screen_id": f"reason_{kind}_{i:03d}", "kind": kind,
                "theme": rng.choice(THEMES), "font": rng.choice(FONT_KEYS),
                "accent": rng.choice(list(ACCENTS)), "density": rng.choice(DENSITIES),
                "site_name": rng.choice(SITE_NAMES),
            }
            _REASON_BUILDERS[kind](spec, rng)
            specs.append(spec)
    return specs


def render_reason_screen(spec: dict) -> tuple[str, list[Mark], dict[str, None], str]:
    doc = _Doc()
    doc.add_plain(f'<div class="header"><div class="logo">{esc(spec["site_name"])}</div></div>')
    kind = spec["kind"]
    if kind == "pricing":
        doc.add_plain(f'<h1 style="position:absolute;left:64px;top:96px;font-size:24px">Choose a plan</h1>')
        _layout_pricing(doc, spec)
    elif kind == "stock":
        doc.add_plain(f'<h1 style="position:absolute;left:64px;top:96px;font-size:24px">Search results</h1>')
        _layout_stock(doc, spec)
    else:
        doc.add_plain(f'<h1 style="position:absolute;left:64px;top:96px;font-size:24px">Pick a time</h1>')
        _layout_slots(doc, spec)
    doc.add_plain(f'<p style="position:absolute;left:64px;top:66px;width:800px">{esc(spec["question"])}</p>')
    html = _wrap_html(spec, "".join(doc.html))
    options = {str(m.number): None for m in doc.marks}
    answer = str(spec["answer_idx"] + 1)
    return html, doc.marks, options, answer


def _layout_pricing(doc: _Doc, spec: dict) -> None:
    plans = spec["content"]["plans"]
    features = spec["content"]["features"]
    n = len(plans)
    card_w, gap, card_h = (360, 40, 380) if n == 2 else (280, 32, 380)
    total_w = card_w * n + gap * (n - 1)
    x0 = (CANVAS_W - total_w) // 2
    y0 = 150
    for i, plan in enumerate(plans):
        x = x0 + i * (card_w + gap)
        doc.add_plain(
            f'<div class="card" style="left:{x}px;top:{y0}px;width:{card_w}px;height:{card_h}px">'
            f'<h2 style="position:absolute;left:24px;top:24px;font-size:20px">{esc(plan["name"])}</h2>'
            f'<span style="position:absolute;left:24px;top:64px;font-size:28px;font-weight:700">${plan["price"]}/mo</span>'
            + "".join(
                f'<span class="muted" style="position:absolute;left:24px;top:{116 + j * 26}px;font-size:14px">&#10003; {esc(f)}</span>'
                for j, f in enumerate(features)
            )
            + "</div>"
        )
        by = y0 + card_h - 64
        doc.add("button", f"Choose {plan['name']}", x + 24, by, card_w - 48, 44,
                 _btn_snippet(f"Choose {plan['name']}", x + 24, by, card_w - 48, 44, primary=True))


def _layout_stock(doc: _Doc, spec: dict) -> None:
    items = spec["content"]["items"]
    prices = spec["content"]["prices"]
    oos_idx = spec["content"]["oos_idx"]
    x, w = 64, 896
    y = 150
    row_h = 84
    for i, (item, price) in enumerate(zip(items, prices)):
        in_stock = i != oos_idx
        status = "In stock" if in_stock else "Out of stock"
        doc.add_plain(
            f'<div class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{row_h}px">'
            f'<span style="position:absolute;left:20px;top:16px;font-weight:700">{esc(item)}</span>'
            f'<span class="muted" style="position:absolute;left:20px;top:{row_h - 30}px">${price:.2f}</span>'
            f'<span style="position:absolute;left:260px;top:{row_h - 30}px;font-weight:700;'
            f'color:{"inherit" if in_stock else "#c0392b"}">{status}</span></div>'
        )
        bx = x + w - 170
        by = y + (row_h - 44) // 2
        text = "Add to cart" if in_stock else "Notify me"
        doc.add("button", text, bx, by, 140, 44, _btn_snippet(text, bx, by, 140, 44, primary=in_stock, disabled=not in_stock))
        y += row_h + 12


def _layout_slots(doc: _Doc, spec: dict) -> None:
    slots = spec["content"]["slots"]
    n = len(slots)
    card_w, gap, card_h = 280, 32, 300
    total_w = card_w * n + gap * (n - 1)
    x0 = (CANVAS_W - total_w) // 2
    y0 = 170
    for i, slot in enumerate(slots):
        x = x0 + i * (card_w + gap)
        doc.add_plain(
            f'<div class="card" style="left:{x}px;top:{y0}px;width:{card_w}px;height:{card_h}px">'
            f'<span style="position:absolute;left:24px;top:28px;font-size:18px;font-weight:700">{slot["label"]}</span>'
            "</div>"
        )
        by = y0 + card_h - 64
        doc.add("button", "Select", x + 24, by, card_w - 48, 44, _btn_snippet("Select", x + 24, by, card_w - 48, 44, primary=True))


# --- rendering to PNG via headless Chrome -----------------------------------------------------------------



# On a fresh --user-data-dir, Chrome's own main process can take 60-90s+ to fully exit after the screenshot
# is written, because it kicks off a Keystone/GoogleUpdater "wake" check in the background; in this sandbox
# that check has no network to complete against, so it just stalls. These flags tell Chrome not to bother
# with any of that (still headless, still a fresh throwaway profile, still the same rendered pixels).
_NO_BACKGROUND_SERVICES = [
    "--no-first-run", "--disable-background-networking", "--disable-component-update", "--disable-sync",
    "--disable-default-apps", "--disable-extensions", "--disable-client-side-phishing-detection",
    "--metrics-recording-only", "--no-default-browser-check", "--disable-backgrounding-occluded-windows",
]


def _render(html_path: Path, png_path: Path, timeout: float = 60) -> None:
    """Render `html_path` to a 1024x768 PNG at `png_path` with a fresh, throwaway Chrome profile.

    Waits for the screenshot file to appear (which happens within a couple of seconds) rather than for the
    Chrome process to exit on its own, then terminates that process -- see `_NO_BACKGROUND_SERVICES` above for
    why the process can otherwise linger well past `timeout` without ever failing to render.
    """
    if png_path.exists():
        png_path.unlink()
    with tempfile.TemporaryDirectory(prefix="glance-ui-chrome-") as profile_dir:
        cmd = [
            CHROME_BIN, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=1", "--window-size=1024,768",
            f"--user-data-dir={profile_dir}", *_NO_BACKGROUND_SERVICES,
            f"--screenshot={png_path}", f"file://{html_path}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if png_path.exists() and png_path.stat().st_size > 0:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise RuntimeError(f"Chrome did not produce a screenshot for {html_path} within {timeout}s")


def _run_renders(jobs: list[tuple[Path, Path, bool]]) -> None:
    """Render each (html_path, png_path, force) job, up to 3 at a time, skipping existing files unless forced."""
    import concurrent.futures

    pending = [(h, p) for h, p, force in jobs if force or not p.exists()]
    if not pending:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_render, h, p) for h, p in pending]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()


# --- orchestration ---------------------------------------------------------------------------------------


def build(n_screens: int = 300, force: bool = False) -> dict[str, Any]:
    if n_screens % len(PAGE_TYPES) != 0:
        raise ValueError(f"n_screens must be a multiple of {len(PAGE_TYPES)} (one of five page types each)")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    per_type = n_screens // len(PAGE_TYPES)
    specs = [_make_spec(i, per_type) for i in range(n_screens)]
    done_n = n_screens // 3
    reason_n = n_screens // 3
    per_type_n = (done_n // len(PAGE_TYPES)) if done_n else 0

    eligible_by_type: dict[str, list[dict]] = {}
    for s in specs:
        if s["done_eligible"]:
            eligible_by_type.setdefault(s["page_type"], []).append(s)
    chosen_done_ids: set[str] = set()
    for pt in PAGE_TYPES:
        pool = eligible_by_type.get(pt, [])
        order = list(range(len(pool)))
        random.Random(f"glance-ui-done-select:{pt}").shuffle(order)
        for idx in order[:per_type_n]:
            chosen_done_ids.add(pool[idx]["screen_id"])

    render_jobs: list[tuple[Path, Path, bool]] = []
    rows: list[tuple[dict, Path]] = []

    for spec in specs:
        html, marks, goal_mark = render_screen(spec, "before")
        screen_id = spec["screen_id"]
        html_path = IMAGES_DIR / f"{screen_id}.html"
        png_path = IMAGES_DIR / f"{screen_id}.png"
        if force or not html_path.exists():
            html_path.write_text(html)
        render_jobs.append((html_path, png_path, force))

        states = {
            **spec["states"], "checkbox_checked": spec["checkbox"]["checked"], "checkbox_label": spec["checkbox"]["label"],
        }
        row: dict[str, Any] = {
            "screen_id": screen_id, "file": png_path.name, "html_file": html_path.name,
            "page_type": spec["page_type"], "theme": spec["theme"], "font": spec["font"],
            "accent": spec["accent"], "density": spec["density"],
            "states": states, "marks": [m.to_dict() for m in marks],
            "goal": {"text": spec["goal"]["text"], "target_mark": goal_mark},
        }
        is_done_selected = screen_id in chosen_done_ids
        if is_done_selected:
            row["done"] = {"goal_text": spec["goal"]["text"], "is_done": False}
        rows.append((row, png_path))

        if is_done_selected:
            after_html, after_marks, _ = render_screen(spec, "after")
            after_id = f"{screen_id}__after"
            after_html_path = IMAGES_DIR / f"{after_id}.html"
            after_png_path = IMAGES_DIR / f"{after_id}.png"
            if force or not after_html_path.exists():
                after_html_path.write_text(after_html)
            render_jobs.append((after_html_path, after_png_path, force))

            after_states = dict(states)
            if spec["page_type"] == "sign_in":
                after_states["signed_in"] = True
            elif spec["page_type"] == "settings":
                after_states["checkbox_checked"] = not states["checkbox_checked"]
            after_row: dict[str, Any] = {
                "screen_id": after_id, "file": after_png_path.name, "html_file": after_html_path.name,
                "page_type": spec["page_type"], "theme": spec["theme"], "font": spec["font"],
                "accent": spec["accent"], "density": spec["density"],
                "states": after_states, "marks": [m.to_dict() for m in after_marks],
                "done": {"goal_text": spec["goal"]["text"], "is_done": True},
            }
            rows.append((after_row, after_png_path))

    reason_specs = _make_reason_specs(reason_n)
    for rspec in reason_specs:
        html, marks, options, answer = render_reason_screen(rspec)
        screen_id = rspec["screen_id"]
        html_path = IMAGES_DIR / f"{screen_id}.html"
        png_path = IMAGES_DIR / f"{screen_id}.png"
        if force or not html_path.exists():
            html_path.write_text(html)
        render_jobs.append((html_path, png_path, force))
        row = {
            "screen_id": screen_id, "file": png_path.name, "html_file": html_path.name,
            "page_type": rspec["kind"], "theme": rspec["theme"], "font": rspec["font"],
            "accent": rspec["accent"], "density": rspec["density"],
            "marks": [m.to_dict() for m in marks],
            "reason": {"question": rspec["question"], "options": options, "answer": answer},
        }
        rows.append((row, png_path))

    _run_renders(render_jobs)

    for row, png_path in rows:
        row["sha256"] = hashlib.sha256(png_path.read_bytes()).hexdigest()

    with open(MANIFEST_PATH, "w") as f:
        for row, _ in rows:
            f.write(json.dumps(row) + "\n")

    return {
        "n_base": len(specs), "n_done_pairs": len(chosen_done_ids), "n_reason": len(reason_specs),
        "n_images": len(render_jobs), "n_rows": len(rows), "manifest": str(MANIFEST_PATH),
    }


# --- contact sheets ----------------------------------------------------------------------------------------


def make_contact_sheets(n_per_sheet: int = 12) -> list[Path]:
    """One JPEG sheet per page type under `lab/sheets_ui/<page_type>.jpg`, first `n_per_sheet` screens, 4x3 grid."""
    from ..logging_utils import read_jsonl

    rows = read_jsonl(MANIFEST_PATH)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    cols, cell_w, cell_h = 4, 240, 180
    written = []
    for page_type in PAGE_TYPES:
        candidates = [r for r in rows if r["page_type"] == page_type and "reason" not in r and "__after" not in r["screen_id"]]
        candidates = candidates[:n_per_sheet]
        if not candidates:
            continue
        n_rows = (len(candidates) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, n_rows * cell_h), (30, 30, 30))
        for i, row in enumerate(candidates):
            img = Image.open(IMAGES_DIR / row["file"]).convert("RGB")
            img.thumbnail((cell_w - 4, cell_h - 4))
            cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
            sheet.paste(img, (cx + 2, cy + 2))
        out_path = SHEETS_DIR / f"{page_type}.jpg"
        sheet.save(out_path, "JPEG", quality=85)
        written.append(out_path)
    return written

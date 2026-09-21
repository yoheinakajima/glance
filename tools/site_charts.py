"""The two figures that lead the project page, drawn as static SVG from results/lab/matrix.json (the owner picked these
two forms, 2026-09-20): cost against speed with each system's three tests joined to their centre, and accuracy against
cost with one panel per test. Every chart comes in a wide and a narrow (phone) drawing; CSS shows one of them.
"""
import html
import json
import math
import pathlib
import statistics

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = {"yesno": "yes/no", "choice": "pick-one", "rating": "rating"}
# display names: the model, and what is done with it ("+ Glance" = the answer is read from one forward pass; "written" = the same model generates JSON)
LABEL = {"Qwen3-VL-4B, written": "Qwen3-VL-4B, written", "Qwen3-VL-2B, written": "Qwen3-VL-2B, written", "Qwen3-VL-8B, written": "Qwen3-VL-8B, written", "Qwen3-VL-4B, read (Glance)": "Qwen3-VL-4B + Glance",
         "Qwen3-VL-2B, read (Glance)": "Qwen3-VL-2B + Glance", "Qwen3-VL-8B, read (Glance)": "Qwen3-VL-8B + Glance"}
LETTER = {"Gemini 3.1 Pro": "G", "Claude Opus 5": "O", "GPT-5.6": "P", "Qwen3-VL-4B, written": "4B written", "Qwen3-VL-2B, written": "2B written", "Qwen3-VL-8B, written": "8B written", "Qwen3-VL-4B, read (Glance)": "4B", "Qwen3-VL-2B, read (Glance)": "2B", "Qwen3-VL-8B, read (Glance)": "8B",  # sizes: the caption says they are read with Glance
          "Claude Haiku 4.5": "H", "GPT-5.6 Luna": "L", "GPT-5 nano": "N", "Gemini 3.1 Flash-Lite": "F"}


def hosted_key(rows):
    """'G Gemini 3.1 Pro, O Claude Opus 5, ...' for the hosted systems actually present."""
    return ", ".join(f"{LETTER.get(n, n[:1])} {n}" for n in dict.fromkeys(r["name"] for r in rows if not r["open"]))


def matrix_rows():
    """One row per system and test. A cell without a clean timing falls back to a contended one and is flagged provisional."""
    m = json.loads((ROOT / "results/lab/matrix.json").read_text())
    gpu = statistics.mean(json.loads((ROOT / "results/lab/cost_model.json").read_text())["assumptions"]["cloud_gpu_usd_per_hour"])
    json_ms = statistics.median(r["latency_ms"] for r in read_jsonl(ROOT / "lab/runs/lab_jsondigits.jsonl"))
    contended = {"Qwen3-VL-4B, written": {"choice": 3.75, "rating": 2.40}, "Qwen3-VL-4B, read (Glance)": {"choice": 1.40, "rating": json_ms / 1000}}
    rows = []
    for name, s in m["systems"].items():
        for test in TESTS:
            sec, prov = s["seconds"][test]["value"], False
            if sec is None:
                sec, prov = contended[name][test], True
            usd = s["usd_per_1000"][test]["value"]
            rows.append({"name": name, "open": name.startswith("Qwen"), "written": name.endswith("written"), "test": test, "acc": s["accuracy"][test]["accuracy"],
                         "ci": s["accuracy"][test]["ci95"], "sec": sec, "usd": statistics.mean(usd) if usd else sec / 3600 * 1000 * gpu, "prov": prov,
                         "estimate": "estimate" in s["usd_per_1000"][test]["how"]})
    return rows


def _log(v, lo, hi, a, b):
    return a + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (b - a)


def _money(v):
    return f"${v:.2f}" if v < 1 else f"${v:.0f}"


def _t(x, y, s, anchor="middle", cls="m-tick"):
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{html.escape(s)}</text>'


def _small(test, x, y, cls):
    if test == "yesno":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" class="{cls}"/>'
    if test == "choice":
        return f'<rect x="{x - 2.6:.1f}" y="{y - 2.6:.1f}" width="5.2" height="5.2" class="{cls}"/>'
    return f'<polygon points="{x:.1f},{y - 3.4:.1f} {x + 3.2:.1f},{y + 2.4:.1f} {x - 3.2:.1f},{y + 2.4:.1f}" class="{cls}"/>'


class _Placer:
    """Greedy label placement: marks are registered first, then each label takes the first of eight positions around its mark that
    touches no mark and no earlier label and stays inside the plot. Labels of the open model are placed first by the callers."""

    def __init__(self, x0, x1, y0, y1):
        self.bounds, self.boxes = (x0, x1, y0, y1), []

    def block(self, x, y, r):
        self.boxes.append((x - r, y - r, x + r, y + r))

    def place(self, x, y, text, r=7.0, char=6.3):
        w, best, best_cost = len(text) * char, None, None
        around = lambda d: ((x + d + 6, y + 4, "start"), (x - d - 6, y + 4, "end"), (x, y - d - 6, "middle"), (x, y + d + 14, "middle"),  # noqa: E731
                            (x + d + 3, y - d - 3, "start"), (x + d + 3, y + d + 11, "start"), (x - d - 3, y - d - 3, "end"), (x - d - 3, y + d + 11, "end"))
        for tx, ty, anchor in around(r) + around(r + 12):  # close positions first, then a ring further out
            left = tx if anchor == "start" else tx - w if anchor == "end" else tx - w / 2
            box = (left, ty - 10, left + w, ty + 3)
            outside = max(0.0, self.bounds[0] - 2 - box[0]) + max(0.0, box[2] - self.bounds[1] - 12) + max(0.0, self.bounds[2] - 14 - box[1]) + max(0.0, box[3] - self.bounds[3] - 2)
            overlap = sum(max(0.0, min(box[2], b[2]) - max(box[0], b[0])) * max(0.0, min(box[3], b[3]) - max(box[1], b[1])) for b in self.boxes)
            cost = overlap + 40 * outside
            if best is None or cost < best_cost - 1e-9:
                best, best_cost = (tx, ty, anchor, box), cost
            if cost == 0:
                break
        self.boxes.append(best[3])
        return best[:3]

def cost_speed(rows, w, h):
    x0, x1, y0, y1 = 50, w - 14, 12, h - 40
    sx, sy = (lambda v: _log(v, 0.25, 6, x0, x1)), (lambda v: _log(v, 0.04, 30, y1, y0))
    g = [f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{y0}" y2="{y1}" class="m-grid"/>' + _t(sx(v), y1 + 16, f"{v:g} s") for v in (0.3, 1, 3)]
    g += [f'<line x1="{x0}" x2="{x1}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="m-grid"/>' + _t(x0 - 7, sy(v) + 4, _money(v), "end") for v in (0.1, 1, 10)]
    big, centres, placer = [], {}, _Placer(x0, x1, y0, y1)
    for name in dict.fromkeys(r["name"] for r in rows):
        mine = [r for r in rows if r["name"] == name]
        cx = sx(math.exp(statistics.mean(math.log(r["sec"]) for r in mine)))
        cy = sy(math.exp(statistics.mean(math.log(r["usd"]) for r in mine)))
        centres[name] = (cx, cy, mine[0]["open"])
        placer.block(cx, cy, 8)
        for r in mine:
            placer.block(sx(r["sec"]), sy(r["usd"]), 4)
        g += [f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{sx(r["sec"]):.1f}" y2="{sy(r["usd"]):.1f}" class="m-spoke"/>' for r in mine]
        g += [_small(r["test"], sx(r["sec"]), sy(r["usd"]), "m-s-prov" if r["prov"] else ("m-s-own" if r["open"] else "m-s-host")) for r in mine]
        big.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" class="{"m-own" if mine[0]["open"] else "m-host"}"/>')
    for name, (cx, cy, is_open) in sorted(centres.items(), key=lambda kv: not kv[1][2]):  # the open model's labels first
        tx, ty, anchor = placer.place(cx, cy, LABEL.get(name, name), char=7.0 if is_open else 6.3)
        big.append(_t(tx, ty, LABEL.get(name, name), anchor, "m-lab m-strong" if is_open else "m-lab"))
    g += big
    g.append(_t((x0 + x1) / 2, h - 6, "median seconds per answer, log scale"))
    g.append(f'<text transform="translate(11,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle" class="m-tick">US dollars per 1,000 answers, log scale</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Cost against speed; each system is three tests joined to their centre">{"".join(g)}</svg>'


def accuracy_cost(rows, w, h, tests):
    n, gap, left = len(tests), 26, 40
    pw = (w - left - 8 - gap * (n - 1)) / n
    g = []
    for j, test in enumerate(tests):
        lo, hi = (0.45, 0.80) if test == "rating" else (0.75, 1.00)  # the 2B model rates at 0.49
        x0, x1, y0, y1 = left + j * (pw + gap), left + j * (pw + gap) + pw, 28, h - 40
        sx, sy = (lambda v, a=x0, b=x1: _log(v, 0.04, 30, a, b)), (lambda v, lo=lo, hi=hi: y1 + (v - lo) / (hi - lo) * (y0 - y1))
        g.append(_t(x0, 14, TESTS[test], "start", "m-head"))
        for k in range(3):
            v = lo + (hi - lo) * k / 2
            g.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="m-grid"/>' + _t(x0 - 6, sy(v) + 4, f"{v:.2f}", "end"))
        g += [f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{y0}" y2="{y1}" class="m-grid"/>' + _t(sx(v), y1 + 16, _money(v)) for v in (0.1, 1, 10)]
        panel, placer = [r for r in rows if r["test"] == test], _Placer(x0, x1, y0, y1)
        for r in panel:
            placer.block(sx(r["usd"]), sy(r["acc"]), 6)
            placer.boxes.append((sx(r["usd"]) - 1.5, sy(min(r["ci"][1], hi)), sx(r["usd"]) + 1.5, sy(max(r["ci"][0], lo))))  # the interval line: keep letters off it
        for r in panel:
            x, y = sx(r["usd"]), sy(r["acc"])
            g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(max(r["ci"][0], lo)):.1f}" y2="{sy(min(r["ci"][1], hi)):.1f}" class="m-whisk"/>')
            cls = "m-prov" if r["prov"] else ("m-own" if r["open"] else "m-host")
            g.append(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" class="{cls}"/>' if r["written"] else f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="{cls}"/>')
        for r in sorted(panel, key=lambda r: not r["open"]):  # the open model's labels first
            text = LETTER.get(r["name"], r["name"][:1])
            tx, ty, anchor = placer.place(sx(r["usd"]), sy(r["acc"]), text, r=5.0, char=7.0 if r["open"] else 6.3)
            g.append(_t(tx, ty, text, anchor, "m-lab m-strong" if r["open"] else "m-lab"))
    g.append(_t((left + w - 8) / 2, h - 6, "US dollars per 1,000 answers, log scale; vertical axis is accuracy"))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Accuracy against cost, {", ".join(TESTS[t] for t in tests)}">{"".join(g)}</svg>'


MEASURES = (("acc", "Accuracy · longer is better", lambda v: f"{v:.2f}"), ("sec", "Seconds per answer · shorter is better", lambda v: f"{v:.2f} s"),
            ("usd", "US dollars per 1,000 answers · shorter is better", lambda v: f"${v:.2f}"))


def bars_by_type(rows, wide, tests=None):
    """The headline figure: bars bundled by question type, real quantities on scales that start at zero. An outlined bar is a
    provisional timing or an estimated cost; an estimate longer than the measured range is clipped and marked."""
    tests = list(tests or TESTS)
    rows = [r for r in rows if r["test"] in tests]
    names = list(dict.fromkeys(r["name"] for r in rows))
    tops = {"acc": 1.0, "sec": max(r["sec"] for r in rows), "usd": max([r["usd"] for r in rows if not r["estimate"]] or [1.0])}

    def bar(r, key, fmt, x, y, span):
        soft = (key != "acc" and r["prov"]) or (key == "usd" and r["estimate"])
        clipped = r[key] > tops[key] * 1.0001
        width = max(1.5, min(r[key] / tops[key], 1.0) * span)
        cls = "m-outline" if soft else (("m-written" if r["written"] else "m-own") if r["open"] else "m-bar")
        note = " est." if key == "usd" and r["estimate"] else (" prov." if soft else "")
        return (f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="9.5" class="{cls}"/>' + (_t(x + width - 3, y + 8.5, "›", "end", "m-lab") if clipped else "")
                + _t(x + width + 5, y + 8.5, fmt(r[key]) + note, "start"))

    if wide:
        g, y, col, span = [], 16, lambda j: 132 + j * 170, 96
        g += [_t(col(j), 12, TESTS[t], "start", "m-head") for j, t in enumerate(tests)]
        for key, title, fmt in MEASURES:
            g.append(_t(4, y + 18, title, "start", "m-tick"))
            y += 24
            for name in names:
                own = name.startswith("Qwen")
                g.append(_t(124, y + 8.5, LABEL.get(name, name), "end", "m-lab m-strong" if own and not name.endswith("written") else ("m-lab m-ink" if own else "m-lab")))
                g += [bar(next(r for r in rows if r["name"] == name and r["test"] == t), key, fmt, col(j), y, span) for j, t in enumerate(tests)]
                y += 14
            y += 6
        return f'<svg viewBox="-30 0 670 {y}" role="img" aria-label="Accuracy, seconds and dollars for every system, by question type">{"".join(g)}</svg>'  # 30 px more on the left: the longest bold row name
    out = []
    for t in tests:
        g, y = [_t(4, 13, TESTS[t], "start", "m-head")], 22
        for key, title, fmt in MEASURES:
            g.append(_t(4, y + 10, title, "start", "m-tick"))
            y += 16
            for name in names:
                g.append(_t(122, y + 8.5, LABEL.get(name, name), "end", "m-lab m-strong" if name.startswith("Qwen") and not name.endswith("written") else ("m-lab m-ink" if name.startswith("Qwen") else "m-lab")))  # 122: room for the longest name at phone width
                g.append(bar(next(r for r in rows if r["name"] == name and r["test"] == t), key, fmt, 128, y, 150))
                y += 14
            y += 8
        out.append(f'<svg viewBox="-16 0 376 {y}" role="img" aria-label="Accuracy, seconds and dollars for every system, {TESTS[t]}">{"".join(g)}</svg>')
    return "".join(out)


def method_diagram(wide):
    """Image and typed question -> one forward pass of a frozen open model -> logits of the allowed answers -> probabilities."""
    steps = [("image + typed question", "yes/no, pick one, or rate"), ("one forward pass", "frozen open model, local"),
             ("answer-position logits", "allowed answers only"), ("probabilities", "nothing generated")]
    g = []
    if wide:
        bw, bh, gap = 146, 58, 18
        for i, (a, b) in enumerate(steps):
            x = 2 + i * (bw + gap)
            g.append(f'<rect x="{x}" y="6" width="{bw}" height="{bh}" class="m-box"/>' + _t(x + bw / 2, 30, a, "middle", "m-lab m-strong") + _t(x + bw / 2, 48, b, "middle", "m-tick"))
            if i < len(steps) - 1:
                g.append(f'<line x1="{x + bw + 3}" x2="{x + bw + gap - 7}" y1="35" y2="35" class="m-arrow"/><polygon points="{x + bw + gap - 3},35 {x + bw + gap - 10},31 {x + bw + gap - 10},39" class="m-own"/>')
        return f'<svg viewBox="0 0 640 70" role="img" aria-label="How a question is answered: image and typed question, one forward pass, logits of the allowed answers, probabilities">{"".join(g)}</svg>'
    bw, bh, gap = 300, 48, 20
    for i, (a, b) in enumerate(steps):
        y = 4 + i * (bh + gap)
        g.append(f'<rect x="30" y="{y}" width="{bw}" height="{bh}" class="m-box"/>' + _t(180, y + 20, a, "middle", "m-lab m-strong") + _t(180, y + 37, b, "middle", "m-tick"))
        if i < len(steps) - 1:
            g.append(f'<line x1="180" x2="180" y1="{y + bh + 3}" y2="{y + bh + gap - 7}" class="m-arrow"/><polygon points="180,{y + bh + gap - 3} 176,{y + bh + gap - 10} 184,{y + bh + gap - 10}" class="m-own"/>')
    return f'<svg viewBox="0 0 360 {4 + len(steps) * (bh + gap)}" role="img" aria-label="How a question is answered">{"".join(g)}</svg>'


def both_widths(wide, narrow):
    return f'<div class="only-wide">{wide}</div><div class="only-narrow">{narrow}</div>'


CSS = """
figure svg{width:100%;height:auto;display:block}
.only-narrow{display:none}@media (max-width:560px){.only-wide{display:none}.only-narrow{display:block}}
svg .m-tick{font-size:11px;fill:var(--muted)}svg .m-lab{font-size:12px;fill:var(--muted)}svg .m-strong{fill:var(--ink);font-weight:600}svg .m-ink{fill:var(--ink)}svg .m-written{fill:var(--ink);opacity:.42}svg .m-head{font-size:12.5px;fill:var(--ink);font-weight:600}
svg .m-grid{stroke:var(--rule);stroke-width:1}svg .m-spoke{stroke:var(--rule);stroke-width:1.2}svg .m-whisk{stroke:var(--muted);stroke-width:1}
svg .m-box{fill:var(--paper);stroke:var(--ink);stroke-width:1.1}svg .m-arrow{stroke:var(--ink);stroke-width:1.2}svg .m-bar{fill:var(--muted)}svg .m-outline{fill:var(--paper);stroke:var(--muted);stroke-width:1;stroke-dasharray:2 2}svg .m-own{fill:var(--ink)}svg .m-host{fill:var(--paper);stroke:var(--ink);stroke-width:1.6}svg .m-prov{fill:var(--paper);stroke:var(--ink);stroke-width:1.4;stroke-dasharray:2 2}
svg .m-s-own{fill:var(--muted)}svg .m-s-host{fill:var(--paper);stroke:var(--muted);stroke-width:1.1}svg .m-s-prov{fill:var(--paper);stroke:var(--muted);stroke-width:1.1;stroke-dasharray:1.6 1.6}
.key svg{width:11px;height:11px;vertical-align:-1px;margin:0 3px 0 8px;display:inline}
"""
KEY = ('<span class="key"><svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="2.8" class="m-s-host"/></svg>yes/no<svg viewBox="0 0 12 12"><rect x="3.4" y="3.4" width="5.2" height="5.2" class="m-s-host"/></svg>pick-one'
       '<svg viewBox="0 0 12 12"><polygon points="6,2.6 9.2,8.4 2.8,8.4" class="m-s-host"/></svg>rating</span>')

"""Three candidate charts for the comparison matrix (results/lab/matrix.json), as one phone-friendly page of static SVG,
so the owner can pick one. Cells without a clean timing use contended measurements and are drawn dashed ("provisional").

uv run python tools/make_chart_options.py   ->  site/chart_options.html
"""
import json
import math
import pathlib
import statistics

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "results/lab/matrix.json").read_text())
TESTS = ["yesno", "choice", "rating"]
NAMES = {"yesno": "yes/no", "choice": "pick-one", "rating": "rating"}
GPU = statistics.mean(json.loads((ROOT / "results/lab/cost_model.json").read_text())["assumptions"]["cloud_gpu_usd_per_hour"])
json_ms = statistics.median(r["latency_ms"] for r in read_jsonl(ROOT / "lab/runs/lab_jsondigits.jsonl"))
PROVISIONAL = {"Qwen3-VL-4B, written": {"choice": 3.75, "rating": 2.40}, "Qwen3-VL-4B, read (Glance)": {"choice": 1.40, "rating": json_ms / 1000}}  # GPU was shared
SHORT = {"Gemini 3.1 Pro": "G", "Claude Opus 5": "O", "GPT-5.6": "P", "Qwen3-VL-4B, written": "written", "Qwen3-VL-4B, read (Glance)": "read"}
rows = []
for name, s in m["systems"].items():
    for t in TESTS:
        sec, prov = s["seconds"][t]["value"], False
        if sec is None:
            sec, prov = PROVISIONAL[name][t], True
        usd = s["usd_per_1000"][t]["value"]
        usd = statistics.mean(usd) if usd else sec / 3600 * 1000 * GPU
        rows.append({"name": name, "k": SHORT[name], "open": name.startswith("Qwen"), "written": name.endswith("written"), "test": t, "acc": s["accuracy"][t]["accuracy"],
                     "ci": s["accuracy"][t]["ci95"], "sec": sec, "usd": usd, "prov": prov})
lin = lambda v, lo, hi, a, b: a + (v - lo) / (hi - lo) * (b - a)  # noqa: E731
log = lambda v, lo, hi, a, b: a + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (b - a)  # noqa: E731
money = lambda v: f"${v:.2f}" if v < 1 else f"${v:.0f}"  # noqa: E731


def mark(r, x, y):
    if not r["open"]:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" class="host"/>'
    cls = "prov" if r["prov"] else "own"
    return (f'<rect x="{x - 4.4:.1f}" y="{y - 4.4:.1f}" width="8.8" height="8.8" class="{cls}"/>' if r["written"] else f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="{cls}"/>')


def t(x, y, s, anchor="middle", cls="tick"):
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{s}</text>'


def option_a():
    out = []
    for test in TESTS:
        lo, hi = (0.5, 0.75) if test == "rating" else (0.8, 1.0)
        x0, x1, y0, y1 = 44, 340, 26, 176
        g = [t(x0, 14, NAMES[test], "start", "head")]
        for j in range(3):
            v = lo + (hi - lo) * j / 2
            y = lin(v, lo, hi, y1, y0)
            g.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>' + t(x0 - 6, y + 4, f"{v:.2f}", "end"))
        for v in (0.1, 1, 10):
            x = log(v, 0.04, 30, x0, x1)
            g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" class="grid"/>' + t(x, y1 + 15, money(v)))
        for r in [r for r in rows if r["test"] == test]:
            x, y = log(r["usd"], 0.04, 30, x0, x1), lin(r["acc"], lo, hi, y1, y0)
            g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{lin(r["ci"][0], lo, hi, y1, y0):.1f}" y2="{lin(min(r["ci"][1], hi), lo, hi, y1, y0):.1f}" class="whisk"/>' + mark(r, x, y)
                     + t(x + 8, y + 4, r["k"], "start", "lab own" if r["open"] else "lab"))
        out.append(f'<svg viewBox="0 0 360 200" role="img" aria-label="accuracy against cost, {NAMES[test]}">{"".join(g)}</svg>')
    return "".join(out) + '<p class="axis">Horizontal: US dollars per 1,000 answers, log scale. Vertical: accuracy with 95% interval.</p>'


def strips(pairs_only=False):
    panels = [("accuracy", "acc", 0.5, 1.0, False, (0.5, 0.75, 1.0), lambda v: f"{v:.2f}"), ("seconds per answer", "sec", 0.2, 6, True, (0.3, 1, 3), lambda v: f"{v:g} s"),
              ("US dollars per 1,000 answers", "usd", 0.04, 30, True, (0.1, 1, 10), money)]
    out = []
    for title, key, lo, hi, is_log, ticks, fmt in panels:
        x0, x1 = 66, 344
        sc = (lambda v: log(v, lo, hi, x0, x1)) if is_log else (lambda v: lin(v, lo, hi, x0, x1))
        g = [t(x0, 14, title, "start", "head")]
        for v in ticks:
            g.append(f'<line x1="{sc(v):.1f}" x2="{sc(v):.1f}" y1="24" y2="150" class="grid"/>' + t(sc(v), 166, fmt(v)))
        for i, test in enumerate(TESTS):
            y = 48 + i * 42
            g.append(t(x0 - 10, y + 4, NAMES[test], "end", "lab own") + f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" class="grid"/>')
            these = [r for r in rows if r["test"] == test and (not pairs_only or r["k"] in ("G", "read"))]
            if pairs_only:
                a, b = (next(r for r in these if r["k"] == k) for k in ("G", "read"))
                gap = (f"{(b['acc'] - a['acc']) * 100:+.1f} points" if key == "acc" else f"{a[key] / b[key]:.1f}x faster" if key == "sec" else f"{a[key] / b[key]:.0f}x cheaper")
                g.append(f'<line x1="{sc(a[key]):.1f}" x2="{sc(b[key]):.1f}" y1="{y}" y2="{y}" class="link"/>' + t((sc(a[key]) + sc(b[key])) / 2, y - 10, gap, "middle", "lab own"))
            for r in these:
                if key == "acc" and not pairs_only:
                    g.append(f'<line x1="{sc(r["ci"][0]):.1f}" x2="{sc(min(r["ci"][1], hi)):.1f}" y1="{y}" y2="{y}" class="whisk"/>')
                g.append(mark(r, sc(r[key]), y) + ("" if r["open"] or pairs_only else t(sc(r[key]), y - 9, r["k"])))
        out.append(f'<svg viewBox="0 0 360 176" role="img" aria-label="{title}">{"".join(g)}</svg>')
    return "".join(out)


LEGEND = ('<p class="legend"><svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.6" class="own"/></svg>open 4B model, read (Glance) '
          '<svg viewBox="0 0 12 12"><rect x="1.6" y="1.6" width="8.8" height="8.8" class="own"/></svg>open 4B model, written '
          '<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.2" class="host"/></svg>hosted: G Gemini, O Opus, P GPT '
          '<svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.2" class="prov"/></svg>provisional timing, clean run tonight</p>')
PAGE = f"""
<title>Glance Chart Options</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{--paper:#ffffff;--ink:#14171a;--muted:#56616b;--rule:#d5dbe1;--link:#1a4480}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#0f1214;--ink:#e4e8eb;--muted:#98a3ad;--rule:#2b3239;--link:#8fb4e8}}}}
:root[data-theme="dark"]{{--paper:#0f1214;--ink:#e4e8eb;--muted:#98a3ad;--rule:#2b3239;--link:#8fb4e8}}
body{{background:var(--paper);color:var(--ink);font:400 1.05rem/1.6 "STIX Two Text",Georgia,serif;padding-inline:18px;padding-block:2rem 3rem}}
main{{max-width:35rem;margin-inline:auto;display:flex;flex-direction:column;gap:2.4rem}}
section{{display:flex;flex-direction:column;gap:.6rem}}h1,h2,p{{margin:0}}
h1{{font-size:1.7rem;font-weight:600;line-height:1.2}}h2{{font-size:1.2rem;font-weight:600;border-top:1.5px solid var(--ink);padding-top:.6rem}}
.note,.legend,.axis{{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.84rem;line-height:1.5;color:var(--muted)}}
.legend svg{{width:12px;height:12px;vertical-align:-1px;margin:0 3px 0 10px}}.legend svg:first-child{{margin-left:0}}
svg{{width:100%;height:auto;display:block}}svg text{{font-family:"IBM Plex Sans",Arial,sans-serif}}
.tick{{font-size:10.5px;fill:var(--muted)}}.lab{{font-size:11px;fill:var(--muted)}}.lab.own{{fill:var(--ink);font-weight:600}}.head{{font-size:12px;fill:var(--ink);font-weight:600}}
.grid{{stroke:var(--rule);stroke-width:1}}.whisk{{stroke:var(--muted);stroke-width:1}}.link{{stroke:var(--ink);stroke-width:2;opacity:.35}}
.host{{fill:var(--paper);stroke:var(--ink);stroke-width:1.4}}.own{{fill:var(--ink)}}.prov{{fill:var(--paper);stroke:var(--ink);stroke-width:1.4;stroke-dasharray:2 2}}
</style>
<main>
<section><h1>Three ways to draw the comparison</h1>
<p class="note">Same data in each: three hosted frontier models, the open 4B model writing, the open 4B model read with Glance; three tests; accuracy, seconds, dollars. Zero-shot, same items in every row.</p>{LEGEND}</section>
<section><h2>A. Accuracy against cost</h2><p class="note">Up and left is better. One panel per test. Speed is not shown.</p>{option_a()}</section>
<section><h2>B. Three strips</h2><p class="note">The matrix redrawn: one strip per measure, one row per test, every system a mark on a shared scale.</p>{strips()}</section>
<section><h2>C. Headline gaps</h2><p class="note">Only two systems: the open model read, against the most accurate hosted model (Gemini).</p>{strips(pairs_only=True)}</section>
</main>
"""
(ROOT / "site/chart_options.html").write_text(PAGE)
print("wrote site/chart_options.html", len(PAGE) // 1024, "KB")

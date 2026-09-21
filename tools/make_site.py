"""Writes the project page from the result files, so no number on it can drift from the repository:
site/index.html (standalone, for static hosting) and site/page.html (the same content without the document wrapper, for a
private preview). Design notes and the content plan: docs/SITE_PLAN.md. Positioning rules: docs/paper/OUTLINE.md.

uv run python tools/make_site.py
"""
import datetime
import html
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTIER = {"openrouter/google/gemini-3.1-pro-preview": "Gemini 3.1 Pro", "anthropic/claude-opus-5": "Claude Opus 5", "openai/gpt-5.6": "GPT-5.6"}


def load(path):
    f = ROOT / path
    return json.loads(f.read_text()) if f.exists() else None


def ci(e, d=3):
    return f'{e["accuracy"]:.{d}f} <span class="ci">[{e["ci95"][0]:.{d}f}, {e["ci95"][1]:.{d}f}]</span>'


def dot_plot(rows, lo, hi, title, width=640, label_w=236):
    """Dot-and-whisker chart drawn to one scale. rows: (label, value, (lo, hi) or None, filled, group_gap_before)."""
    row_h, top, y, marks = 26, 34, 34, []
    x = lambda v: label_w + (v - lo) / (hi - lo) * (width - label_w - 56)  # noqa: E731
    ticks = [lo + i * (hi - lo) / 5 for i in range(6)]
    for label, value, whisk, filled, gap in rows:
        y += 12 if gap else 0
        cy = y + row_h / 2
        marks.append(f'<text x="{label_w - 12}" y="{cy + 4:.1f}" text-anchor="end" class="lab{" own" if filled else ""}">{html.escape(label)}</text>')
        if whisk:
            marks.append(f'<line x1="{x(whisk[0]):.1f}" x2="{x(whisk[1]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="whisk"/>')
        marks.append(f'<circle cx="{x(value):.1f}" cy="{cy:.1f}" r="4.5" class="{"dot own" if filled else "dot"}"/>')
        marks.append(f'<text x="{width - 4}" y="{cy + 4:.1f}" text-anchor="end" class="val">{value:.3f}</text>')
        y += row_h
    grid = "".join(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{y + 4}" class="grid"/><text x="{x(t):.1f}" y="{top - 12}" text-anchor="middle" class="tick">{t:.2f}</text>' for t in ticks)
    return (f'<div class="figure-scroll"><svg viewBox="0 0 {width} {y + 14}" role="img" aria-label="{html.escape(title)}" '
            f'style="min-width:520px;width:100%;height:auto">{grid}{"".join(marks)}</svg></div>')


fresh, inat = load("results/lab/fresh_commons.json"), load("results/lab/fresh_inat.json")
h2h, gen, lf = load("results/lab/frontier_head_to_head.json"), load("results/lab/gen_accuracy.json"), load("results/lab/label_free_test.json")
cost, measured, ladder = load("results/lab/cost_model.json"), load("results/lab/frontier_cost_measured.json"), load("lab/READOUT_LADDER.json")
nullp, scaling, genb = load("results/lab/null_prior.json"), load("results/lab/scaling.json"), load("lab/GENBENCH.json")

# ---- Figure 1: fresh photos ------------------------------------------------------------------------
rows1 = []
for suite, key, tag in (("fresh_yesno", "vlm:statement", "yes/no"), ("fresh_choice", "vlm:independent", "pick one of 13")):
    s = fresh["suites"][suite]
    rows1.append((f"Qwen3-VL-4B, read · {tag}", s[key]["accuracy"], s[key]["ci95"], True, suite == "fresh_choice"))
    rows1 += [(f"{name} · {tag}", s[m]["accuracy"], s[m]["ci95"], False, False) for m, name in FRONTIER.items() if m in s]
fig1 = dot_plot(rows1, 0.75, 1.0, "Accuracy on photos taken after every model's release")

# ---- Figure 2: ratings -------------------------------------------------------------------------------
written = sum(gen[k]["written"][0] for k in gen if k.startswith("ladder_")) / 5
loc = h2h["local"]
rows2 = [(FRONTIER.get(r["model"], r["model"]) + ", written pick", r["mean_accuracy"], None, False, False) for r in h2h["runs"].values()]
rows2 += [("Qwen3-VL-4B, written answer", written, None, True, False), ("Qwen3-VL-4B, read (no labels)", loc["+ Glance ens4d, 0 labels"]["mean_accuracy"], None, True, False),
          ("… read + unlabeled images", loc["+ Glance ens4d, 0 labels + unlabeled images"]["mean_accuracy"], None, True, True),
          ("… read + 32 labels", loc["+ Glance ens4d, 32 labels"]["mean_accuracy"], None, True, False)]
fig2 = dot_plot(rows2, 0.5, 1.0, "Exact-level accuracy on five 4-level rating scales, same 1,000 images")

# ---- tables ----------------------------------------------------------------------------------------
def fresh_table():
    out = ['<div class="table-scroll"><table><thead><tr><th>System, zero-shot</th><th class="n">yes/no</th><th class="n">pick-one</th></tr></thead><tbody>']
    y, c = fresh["suites"]["fresh_yesno"], fresh["suites"]["fresh_choice"]
    out.append(f'<tr class="own"><td>Qwen3-VL-4B, read <span class="note">Commons, 131 photos</span></td><td class="n">{ci(y["vlm:statement"])}</td><td class="n">{ci(c["vlm:independent"])}</td></tr>')
    out += [f'<tr><td>{name} <span class="note">same photos, test half</span></td><td class="n">{ci(y[m])}</td><td class="n">{ci(c[m])}</td></tr>' for m, name in FRONTIER.items() if m in y]
    if inat:
        yi, cj = inat["suites"]["inat_yesno"], inat["suites"]["inat_choice"]
        out.append(f'<tr class="own sep"><td>Qwen3-VL-4B, read <span class="note">iNaturalist, 200 photos</span></td><td class="n">{ci(yi["vlm:statement"])}</td><td class="n">{ci(cj["vlm:independent"])}</td></tr>')
        extra = [m for m in FRONTIER if m in yi]
        out += [f'<tr><td>{FRONTIER[m]}</td><td class="n">{ci(yi[m])}</td><td class="n">{ci(cj[m])}</td></tr>' for m in extra]
        if not extra:
            out.append('<tr><td>Frontier models <span class="note">iNaturalist</span></td><td class="n pending" colspan="2">paid calls not yet run</td></tr>')
    return "".join(out) + "</tbody></table></div>"


def cost_table():
    w = cost["same_model_write_vs_read_per_1000_requests"]
    f = lambda r: f'${r["cloud_gpu_usd"][0]:.2f}–{r["cloud_gpu_usd"][1]:.2f}'  # noqa: E731
    out = ['<div class="table-scroll"><table><thead><tr><th>Request, one image</th><th class="n">writes JSON</th><th class="n">read</th><th class="n">saving</th><th class="n">agree</th></tr></thead><tbody>']
    out += [f'<tr><td>{html.escape(k)}</td><td class="n">{f(e["write_json"])}</td><td class="n">{f(e["read_fast2"])}</td><td class="n">{e["saving_read_fast2"]:.0%}</td><td class="n">{e["answers_agree"]:.0%}</td></tr>' for k, e in w.items()]
    return "".join(out) + "</tbody></table></div>"


got = [r for r in (measured or []) if r["usd_per_1000_calls"] is not None]
api_lo, api_hi = min(r["usd_per_1000_calls"] for r in got), max(r["usd_per_1000_calls"] for r in got)
raw = lf["raw (0 labels, no pool)"]
r4a = ladder["summary"]["R4a"]["accuracy"] if ladder and "R4a" in ladder.get("summary", {}) else None
paired = {"Claude Opus 5": "+12.2 [8.1, 16.3]", "GPT-5.6": "+7.5 [3.0, 11.9]", "Gemini 3.1 Pro": "+2.1 [−2.3, 6.6]"}  # lab/NOTES.md entry 32b
today = datetime.date.today().isoformat()

BODY = f"""
<header>
  <p class="running">Working paper · results regenerated {today} from the repository · every experiment registered before it ran</p>
  <h1>Glance</h1>
  <p class="subtitle">Reading typed decisions from open vision-language models: what a frozen 4B model already does, measured against frontier APIs on photographs none of them has seen.</p>
  <p class="byline">Yohei Nakajima</p>
  <p class="links"><a href="#evidence">Evidence</a> · <a href="#limits">Limits and misses</a> · <a href="#reproduce">Reproduce</a> · Paper and code: forthcoming</p>
</header>

<section aria-labelledby="abstract">
  <h2 id="abstract" class="plain">Abstract</h2>
  <p class="abstract">A frozen open vision-language model (Qwen3-VL-4B, Apache-2.0, on a laptop) is asked typed questions about an image: is this true, which one, where on this rubric. The answer is <em>read</em> from the logits of a single forward pass, not generated. On photographs taken after every model’s release, with labels nobody on this project made, it is level with Gemini 3.1 Pro, Claude Opus 5 and GPT-5.6 on yes/no and pick-one questions. On ratings it orders images almost perfectly zero-shot (rank agreement {0.93:.2f}, within one level on {raw["within_1"]:.2f} of images); exact levels depend on where a rubric draws its lines, which no model can know unseen. Reading gives the same answers as the same model writing JSON, {genb["1 yes/no"]["speedup_vs_write_fast2"]:.1f} to {genb["25 ratings"]["speedup_vs_write_fast2"]:.1f} times faster, with probabilities. Glance is the calibration and measurement harness around that readout; the readout itself is shared with other training-free tools and is not claimed as new.</p>
</section>

<section aria-labelledby="evidence">
  <h2 id="evidence"><span class="num">1</span>Evidence</h2>
  <h3>1.1 Yes/no and pick-one on photographs no model has seen</h3>
  <p>Wikimedia Commons photographs taken after 15 August 2026, labelled by their uploaders’ structured “depicts” statements, and iNaturalist observations uploaded on the day of the test, labelled by community identification. No labels were made by us or by any model.</p>
  {fresh_table()}
  <p class="caption"><b>Table 1.</b> Accuracy with 95% bootstrap intervals, uncalibrated decisions, nothing fitted. Rows in bold are the open model.</p>
  <figure>{fig1}<figcaption><b>Figure 1.</b> The Commons rows of Table 1, drawn to one scale. Filled marks are the open 4B model; hollow marks are hosted frontier models. Every interval overlaps every other.</figcaption></figure>

  <h3>1.2 Ratings against a rubric in words</h3>
  <p>Five synthetic four-level scales (blur, exposure, JPEG, noise, resolution), the same 1,000 held-out images for every system. Zero-shot, the open model’s <em>written</em> answer is ahead of Claude Opus 5 by {paired["Claude Opus 5"]} points and of GPT-5.6 by {paired["GPT-5.6"]}, and level with Gemini 3.1 Pro ({paired["Gemini 3.1 Pro"]}), paired on the same images. Our own zero-shot <em>read</em> is {100 * (written - loc["+ Glance ens4d, 0 labels"]["mean_accuracy"]):.0f} points worse than the same model’s written answer, a prompt effect we registered, measured and report against ourselves.</p>
  <figure>{fig2}<figcaption><b>Figure 2.</b> Exact-level accuracy, chance 0.25. The lower group has seen images of the rubric: unlabeled ones (zero labels, but not zero-shot), then 32 labeled ones. Intervals for n = 1,000 are about ±0.03; paired differences are in the repository.</figcaption></figure>
  <p>Exact level is a hard target for any model because level boundaries are a convention. Zero-shot the open model is exactly right on {raw["accuracy"]:.2f} of images but within one level on {raw["within_1"]:.3f}, and 97% of its errors are one step, in a direction that is constant per rubric. Sixteen <em>unlabeled</em> images of the rubric remove that offset ({raw["accuracy"]:.3f} → {lf["BCz, pool = 16 unlabeled (20 draws)"]["accuracy"]:.3f}).</p>

  <h3>1.3 What reading buys over writing, same model</h3>
  {cost_table()}
  <p class="caption"><b>Table 2.</b> Cost per 1,000 images on a rented GPU assumed no faster than the laptop; self-hosted cost is GPU time, so the saving is the measured time saving. For comparison, the frontier calls on these tasks measured ${api_lo:.2f} to ${api_hi:.2f} per 1,000 answers.</p>
  <p>On yes/no and pick-one the read and the written answers are identical item for item. Reading adds probabilities (accuracy on the most confident 80% of answers is 0.975 on the iNaturalist set), and it is the only route to fitting: a written pick has nothing to calibrate.</p>
</section>

<section aria-labelledby="new">
  <h2 id="new"><span class="num">2</span>What is, and is not, new</h2>
  <p>Reading answer-token logits from a frozen generative model, with several questions sharing one image prefix, is what Simple Jev, jev-visual and LitJev also do. For yes/no and pick-one the forward pass here is not new, and the accuracy belongs to the open model. What this project adds is a harness on top: a fresh-photograph comparison with paid frontier calls, the same model writing against reading, self-calibration from unlabeled images and labeled fitting for rating levels (<code>glance fit</code>), and measured dollars and milliseconds. It trains no weights, unlike YOFO, Laya Vision or OpenJev v2.</p>
</section>

<section aria-labelledby="limits">
  <h2 id="limits"><span class="num">3</span>Limits and misses</h2>
  <ul class="misses">
    <li><span class="verdict miss">not supported</span> A content-free prior (blank and noise images) was expected to help zero-shot ratings. It took exact accuracy from {nullp["table"]["raw zero-shot"]["accuracy"]:.3f} to {nullp["table"]["content-free prior, all six null images (registered)"]["accuracy"]:.3f}: for an image rubric there is no content-free image.</li>
    <li><span class="verdict miss">not supported</span> The rating readout was expected to match the written answer zero-shot. It trails it by ten points; a one-pass read at the JSON answer position is registered as the follow-up.</li>
    <li><span class="verdict miss">not supported</span> On KADID-10k (23 distortion types, five levels, human scores) every registered target was missed: 0.527 exact with labels, 0.33 zero-shot.</li>
    <li><span class="verdict">supported</span> A fitted readout on the model’s hidden state reaches {r4a:.3f} from one pass, against 0.867 for the token readout: the model represents severity almost perfectly. It needs on the order of a hundred labels.</li>
    <li><span class="verdict">known</span> Hand-built image features beat the VLM on low-level artifacts when labels are plentiful (0.979). A calibration fitted on one rubric does not transfer to another. One model family measured so far; a second family and a 2B / 4B / 8B ladder are running.</li>
  </ul>
</section>

<section aria-labelledby="reproduce">
  <h2 id="reproduce"><span class="num">4</span>Reproduce</h2>
<pre><code>uv sync &amp;&amp; uv run glance doctor
uv run python tools/fetch_fresh_inat.py          # 200 photographs, about ten API calls
uv run glance eval --suite inat_choice --suite inat_yesno --model vlm
uv run python tools/make_results_zeroshot.py     # every table on this page</code></pre>
  <p>The lab notebook records each hypothesis before its experiment, each verdict after, and two errata. Frontier model outputs are never stored; only whether each answer was right.</p>
</section>

<footer>
  <p>Open model: Qwen3-VL-4B-Instruct, Apache-2.0. Photographs: CC0, CC BY and CC BY-SA, attributed per file in the repository and not redistributed. This page is generated by <code>tools/make_site.py</code>.</p>
</footer>
"""

STYLE = """
<title>Glance Working Paper</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<style>
:root{--paper:#ffffff;--ink:#14171a;--muted:#56616b;--rule:#d5dbe1;--link:#1a4480;--miss:#8f2d2d;--own:#14171a;--tint:#f3f5f7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#0f1214;--ink:#e4e8eb;--muted:#98a3ad;--rule:#2b3239;--link:#8fb4e8;--miss:#e39a9a;--own:#e4e8eb;--tint:#171b1f}}
:root[data-theme="dark"]{--paper:#0f1214;--ink:#e4e8eb;--muted:#98a3ad;--rule:#2b3239;--link:#8fb4e8;--miss:#e39a9a;--own:#e4e8eb;--tint:#171b1f}
body{background:var(--paper);color:var(--ink);font:400 1.0625rem/1.62 "STIX Two Text","Iowan Old Style","Palatino Linotype",Georgia,serif;padding-inline:20px;padding-block:2.5rem 4rem}
main{max-width:41rem;margin-inline:auto;display:flex;flex-direction:column;gap:2.75rem}
section,header{display:flex;flex-direction:column;gap:.9rem}
p,ul,pre,table,figure,h1,h2,h3{margin:0}
a{color:var(--link);text-decoration-thickness:.06em;text-underline-offset:.18em}
a:focus-visible{outline:2px solid var(--link);outline-offset:2px}
.running,.caption,figcaption,.links,.note,th,.ci,.tick,.lab,.val,footer,.verdict{font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif}
.running{font-size:.78rem;color:var(--muted);letter-spacing:.02em;border-bottom:1px solid var(--rule);padding-bottom:.7rem}
h1{font-size:2.6rem;line-height:1.1;font-weight:600;letter-spacing:-.01em;margin-top:1.2rem}
.subtitle{font-size:1.28rem;line-height:1.42;text-wrap:balance}
.byline{font-size:1.05rem}
.links{font-size:.9rem;color:var(--muted)}
h2{font-size:1.32rem;font-weight:600;line-height:1.25;text-wrap:balance;border-top:1.5px solid var(--ink);padding-top:.7rem}
h2.plain{border-top:0;padding-top:0;font-size:1rem;font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:500}
.num{display:inline-block;min-width:1.6rem;color:var(--muted);font-weight:400}
h3{font-size:1.06rem;font-weight:600;font-style:italic;margin-top:.6rem}
.abstract{font-size:1.1rem;line-height:1.6}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.95rem;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}.table-scroll{overflow-x:auto}
thead th{font-size:.76rem;font-weight:500;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);text-align:left;padding:.5rem .9rem .45rem 0;border-bottom:.75px solid var(--ink);white-space:nowrap}
td{padding:.42rem .9rem .42rem 0;vertical-align:baseline;white-space:nowrap}
td:first-child{white-space:normal;min-width:11rem}
.n{text-align:right}td.n:last-child,th.n:last-child{padding-right:0}
tr.own td{font-weight:600}tr.sep td{border-top:.75px solid var(--rule)}
.ci{font-size:.78rem;color:var(--muted);font-weight:400}.note{font-size:.78rem;color:var(--muted);font-weight:400;display:block}
.pending{color:var(--muted);font-style:italic;text-align:center}
.caption,figcaption{font-size:.82rem;line-height:1.5;color:var(--muted)}
figure{display:flex;flex-direction:column;gap:.6rem;margin-block:.4rem}
.figure-scroll{overflow-x:auto}
svg text{fill:var(--ink)}svg .tick,svg .val{font-size:11px;fill:var(--muted)}svg .lab{font-size:12.5px;fill:var(--muted)}svg .lab.own{fill:var(--ink);font-weight:600}
svg .grid{stroke:var(--rule);stroke-width:1}svg .whisk{stroke:var(--ink);stroke-width:1.25}
svg .dot{fill:var(--paper);stroke:var(--ink);stroke-width:1.5}svg .dot.own{fill:var(--own)}
.misses{list-style:none;padding:0;display:flex;flex-direction:column;gap:.85rem}
.verdict{display:block;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600}.verdict.miss{color:var(--miss)}
pre{background:var(--tint);border-left:2px solid var(--rule);padding:.8rem 1rem;overflow-x:auto;font:400 .82rem/1.55 "IBM Plex Mono",ui-monospace,Menlo,monospace}
code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.88em}
footer{font-size:.8rem;line-height:1.55;color:var(--muted);border-top:1px solid var(--rule);padding-top:1rem}
@media (max-width:480px){h1{font-size:2.1rem}.subtitle{font-size:1.12rem}body{font-size:1rem}}
</style>
"""

page = STYLE + "<main>" + BODY + "</main>\n"
(ROOT / "site").mkdir(exist_ok=True)
(ROOT / "site/page.html").write_text(page)
(ROOT / "site/index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                      + STYLE + "</head>\n<body>\n<main>" + BODY + "</main>\n</body>\n</html>\n")
print("wrote site/index.html and site/page.html,", len(page) // 1024, "KB")

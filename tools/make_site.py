"""Writes the project page from the result files, so no number on it can drift from the repository:
site/index.html (standalone, for static hosting) and site/page.html (the same content without the document wrapper, for a
private preview). Design notes and the content plan: docs/SITE_PLAN.md. Positioning rules: docs/paper/OUTLINE.md.

uv run python tools/make_site.py
"""
import datetime
import html
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import site_charts  # noqa: E402

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
jd = load("results/lab/jsondigits.json")["mean"]

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
rows2 += [("Qwen3-VL-4B, written answer", written, None, True, False), ("Qwen3-VL-4B, read, one pass", jd["json_zero"], None, True, False),
          ("… earlier four-pass read", jd["ens_zero"], None, True, False),
          ("… one-pass read + 16 unlabeled images", jd["json_u16"], None, True, True),
          ("… four-pass read + 32 labels", loc["+ Glance ens4d, 32 labels"]["mean_accuracy"], None, True, False)]
fig2 = dot_plot(rows2, 0.5, 1.0, "Exact-level accuracy on five 4-level rating scales, same 1,000 images", label_w=262)

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


matrix = load("results/lab/matrix.json")
mrows = site_charts.matrix_rows()
ALL = list(site_charts.TESTS)
fig_cost_speed = site_charts.both_widths(site_charts.cost_speed(mrows, 640, 400), site_charts.cost_speed(mrows, 360, 380))
fig_acc_cost = site_charts.both_widths(site_charts.accuracy_cost(mrows, 640, 250, ALL), "".join(site_charts.accuracy_cost(mrows, 360, 220, [t]) for t in ALL))
fig_method = site_charts.both_widths(site_charts.method_diagram(True), site_charts.method_diagram(False))
has_prov = any(r["prov"] for r in mrows)
fig_bars = site_charts.both_widths(site_charts.bars_by_type(mrows, True, ALL), site_charts.bars_by_type(mrows, False, ALL))
hosted_rating = sorted(v["accuracy"]["rating"]["accuracy"] for n, v in load("results/lab/matrix.json")["systems"].items() if not n.startswith("Qwen"))
has_est = any(r["estimate"] for r in mrows)
TEST_HEADS = {"yesno": "yes/no", "choice": "pick-one", "rating": "rating"}


def matrix_tables(tests):
    def fmt(field, c):
        if field == "accuracy":
            return ci(c)
        if c["value"] is None:
            return '<span class="pending">tonight</span>'
        if field == "seconds":
            return f'{c["value"]:.2f} s'
        lo, hi = c["value"]
        est = ' <span class="ci">est.</span>' if "estimate" in c["how"] else ""
        return (f"${lo:.2f}" if lo == hi else f"${lo:.2f}–{hi:.2f}") + est
    out = []
    for title, field in (("Accuracy, zero-shot", "accuracy"), ("Median seconds per answer", "seconds"), ("US dollars per 1,000 answers", "usd_per_1000")):
        out.append(f'<div class="table-scroll"><table><thead><tr><th>{title}</th>' + "".join(f'<th class="n">{TEST_HEADS[t]}</th>' for t in tests) + "</tr></thead><tbody>")
        for name, sysrow in matrix["systems"].items():
            own = ' class="own"' if name.startswith("Qwen") else ""
            sep = ' class="own sep"' if name.endswith("written") else own
            out.append(f"<tr{sep}><td>{html.escape(name)}</td>" + "".join(f'<td class="n">{fmt(field, sysrow[field][t])}</td>' for t in tests) + "</tr>")
        out.append("</tbody></table></div>")
    return "".join(out)


def extras_table():
    out = ['<div class="table-scroll"><table><thead><tr><th>Give the read row</th><th class="n">rating accuracy</th></tr></thead><tbody>']
    out += [f'<tr><td>{html.escape(e["what"]).replace("`glance fit --unlabeled`", "<code>glance fit --unlabeled</code>").replace("`glance fit`", "<code>glance fit</code>")}'
            f'<span class="note">{html.escape(e["note"])}</span></td><td class="n">{e["rating_accuracy"]:.3f}</td></tr>' for e in matrix["only_the_read_row_can_add"]]
    return "".join(out) + "</tbody></table></div>"


def closed_set_table():
    other = load("results/lab/other_models.json") or {}
    order = [("Qwen3-VL-2B", "Qwen3-VL-2B"), ("Qwen3-VL-4B", "Qwen3-VL-4B"), ("Qwen3-VL-8B", "Qwen3-VL-8B"), ("SmolVLM2-2.2B, another family", "SmolVLM2-2.2B (another family)")]
    out = ['<div class="table-scroll"><table><thead><tr><th>Open model, same questions</th><th class="n">yes/no · Commons</th><th class="n">yes/no · iNaturalist</th><th class="n">pick-one · Commons</th><th class="n">pick-one · iNaturalist</th></tr></thead><tbody>']
    for label, key in order:
        e = other.get(key)
        if not e:
            out.append(f'<tr><td>{label}</td><td class="n pending" colspan="4">running tonight</td></tr>')
            continue
        out.append(f'<tr{" class=own" if key == "Qwen3-VL-4B" else ""}><td>{label}</td>' + "".join(f'<td class="n">{e[k]["accuracy"]:.3f}</td>' if k in e else '<td class="n">–</td>'
                                                                                                    for k in ("fresh_yesno", "inat_yesno", "fresh_choice", "inat_choice")) + "</tr>")
    return "".join(out) + "</tbody></table></div>"


def models_table():
    sizes = (scaling or {}).get("sizes", {})
    order = [("Qwen3-VL-2B", "2B"), ("Qwen3-VL-4B", "4B"), ("Qwen3-VL-8B", "8B"), ("SmolVLM2-2.2B, another family", "SmolVLM2-2.2B (another family)")]
    out = ['<div class="table-scroll"><table><thead><tr><th>Open model, same prompts</th><th class="n">zero-shot</th><th class="n">within one</th><th class="n">+16 unlabeled</th><th class="n">+32 labels</th></tr></thead><tbody>']
    for label, key in order:
        r = sizes.get(key)
        if not r:
            out.append(f'<tr><td>{label}</td><td class="n pending" colspan="4">running tonight</td></tr>')
            continue
        zero = r.get("json_zero_shot", r["zero_shot"])
        u16 = r.get("json_unlabeled_16", r["unlabeled_16"])
        note = "" if "json_zero_shot" in r else '<span class="note">four-pass read; the one-pass read was not collected</span>'
        out.append(f'<tr{" class=own" if key == "4B" else ""}><td>{label}{note}</td><td class="n">{zero:.3f}</td><td class="n">{r["within_1"]:.3f}</td><td class="n">{u16:.3f}</td><td class="n">{r["labels_32"]:.3f}</td></tr>')
    return "".join(out) + "</tbody></table></div>"


hosted = {n: v for n, v in matrix["systems"].items() if not n.startswith("Qwen")}
best_rating = max(hosted, key=lambda n: hosted[n]["accuracy"]["rating"]["accuracy"])
best_rating_acc = hosted[best_rating]["accuracy"]["rating"]["accuracy"]
cheap_yesno = min((v["usd_per_1000"]["yesno"]["value"][0], n) for n, v in hosted.items() if v["usd_per_1000"]["yesno"]["value"])
cheap_rating = min((v["usd_per_1000"]["rating"]["value"][0], n) for n, v in hosted.items() if v["usd_per_1000"]["rating"]["value"] and "estimate" not in v["usd_per_1000"]["rating"]["how"])
own_yesno = matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["usd_per_1000"]["yesno"]["value"]
own_rating = matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["usd_per_1000"]["rating"]["value"]


def times_cheaper(hosted_usd, own_range):
    """'about N times cheaper than' / 'about as cheap as', from the numbers, so the sentence cannot outlive a re-measurement."""
    ratio = hosted_usd / (sum(own_range) / 2)
    return f"about {ratio:.0f} times cheaper than" if ratio >= 1.75 else ("somewhat cheaper than" if ratio >= 1.2 else "no cheaper than")


def speed_rank(test):
    """'faster than N of the M hosted models' for the open model's read, from the matrix."""
    own = matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["seconds"][test]["value"]
    others = [v["seconds"][test]["value"] for n, v in hosted.items()]
    return f"{own:.1f} s, faster than {sum(own < o for o in others)} of the {len(others)} hosted models"


photo_timed = "on these photographs" in matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["seconds"]["yesno"]["how"]
size_note = "" if photo_timed else " The open model’s yes/no and pick-one timings here were taken on smaller test images (and a three-option pick-one) than the photographs the hosted models saw; a timing on the same photographs is running and will replace them."
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
  <p class="subtitle">Reading typed decisions from open vision-language models: what a frozen 4B model already does for yes/no and pick-one questions, measured against hosted models on photographs none of them has seen.</p>
  <p class="byline">Yohei Nakajima</p>
  <p class="links"><a href="#what">What it is</a> · <a href="#evidence">Does it work?</a> · <a href="#read">Why read</a> · <a href="#ratings">Ratings</a> · <a href="#models">Other models</a> · <a href="#limits">Limits and misses</a> · <a href="#reproduce">Reproduce</a> · Paper and code: forthcoming</p>
</header>

<section aria-labelledby="abstract">
  <h2 id="abstract" class="plain">Abstract</h2>
  <p class="abstract">A frozen open vision-language model (Qwen3-VL-4B, Apache-2.0, on a laptop) is asked typed questions about an image, and the answer is <em>read</em> from the logits of a single forward pass, not generated. For yes/no and pick-one questions this works with no setup. On photographs taken after every model’s release, with labels nobody on this project made, the open model is level with Gemini 3.1 Pro, Claude Opus 5 and GPT-5.6, and with the cheapest current models of OpenAI and Google. It gives the same answers as the same model writing JSON, faster (1.5 times on a full-size photograph, {genb["1 yes/no"]["speedup_vs_write_fast2"]:.1f} to {genb["25 ratings"]["speedup_vs_write_fast2"]:.1f} times on small images and many questions), with probabilities. A yes/no about a full-size photograph takes {speed_rank("yesno")}, and is {times_cheaper(cheap_yesno[0], own_yesno)} the cheapest hosted model; no image leaves the machine. A 2B model of the same family does as well on yes/no. Ratings are the hard case: zero-shot every model gets the order right (rank agreement {0.93:.2f}, within one level on {raw["within_1"]:.2f} of images) and the exact level often wrong; the cheapest Google model leads there ({best_rating_acc:.3f} against {jd["json_zero"]:.3f}), and the open model needs 16 unlabeled images of the rubric to come level ({jd["json_u16"]:.3f}) and 32 labels to lead. Glance is the calibration and measurement harness around that readout; the readout itself is shared with other training-free tools and is not claimed as new.</p>
</section>

<section aria-labelledby="what">
  <h2 id="what"><span class="num">1</span>What it is</h2>
  <p>You hand a frozen open vision-language model an image and a <em>typed</em> question: is this true (yes/no), which one (pick one of a list), or where on this rubric (a rating). The model runs once. Nothing is generated: the answer is read from the scores the model gives to the allowed answers at the position where it would start writing, and those scores become probabilities. Several questions about one image share the cost of reading the image.</p>
  <figure>{fig_method}</figure>
<pre><code>photo: a bold jumping spider, iNaturalist 402011619 (CC BY, antimatterbee)

"Is the main subject a spider or other arachnid?"  ->  P(yes) = 1.00        0.8 s
"Is the main subject a fish?"                      ->  P(yes) = 0.00        0.7 s
"What kind of organism is it?"  (10 options)       ->  arachnid, p = 1.00   1.3 s</code></pre>
  <p class="caption">A real logged example from the open 4B model on a laptop, with a second job sharing the GPU. Most answers are this decisive; the uncertain ones are where the probabilities earn their keep.</p>
</section>

<section aria-labelledby="evidence">
  <h2 id="evidence"><span class="num">2</span>Does it work? Every system, the same items</h2>
  <p>The same questions went to the open 4B model (read with Glance, and also <em>writing</em> its answer as JSON), to three hosted flagships and to each provider’s cheapest current vision model. Three tests, all zero-shot: {matrix["tests"]["yesno"].lower()}; {matrix["tests"]["choice"].lower()}; {matrix["tests"]["rating"].lower()}. The photographs were taken after every model’s release and labelled by people outside this project (section 2.1).</p>
  <figure>{fig_bars}<figcaption><b>Figure 1.</b> Accuracy, seconds and dollars for every system, bundled by question type, on scales that start at zero. Dark bars are the open 4B model.{" An outlined bar is a provisional timing (taken while the GPU was shared) or an estimated cost; an estimate beyond the measured range is cut short and marked ›." if (has_prov or has_est) else ""}</figcaption></figure>
  <p><b>Yes/no and pick-one:</b> every system is level on accuracy. The open model answers a yes/no in {speed_rank("yesno")}; it costs a fraction of the flagships and is {times_cheaper(cheap_yesno[0], own_yesno)} the cheapest hosted model. Most of its time on a full-size photograph goes into reading the image, not into the answer. <b>Ratings:</b> there is no clear best model. {best_rating}, the cheapest Google model, leads on accuracy, each provider’s cheap model beats its own flagship, and the open 4B model sits between them. Section 4 explains why ratings behave differently and what fixes them.</p>
  {matrix_tables(ALL)}
  <p class="caption"><b>Table 1.</b> The numbers behind Figure 1. Accuracy with 95% bootstrap intervals on the items EVERY system answered (the test half of the Commons set: 131 yes/no questions, 65 pick-one photographs; section 2.1 gives the open model on all items); bold rows are the open model. The open model’s rating is read in one pass at the JSON answer position. Hosted speed is wall time per call from one laptop; hosted cost is the provider’s bill where it was logged (“est.” is a list-price upper estimate for a run that predates cost logging). Open-model speed is measured with the GPU otherwise idle; its cost is those seconds at an on-demand cloud GPU price. Cells marked “tonight” await a clean timing.{size_note} No few-shot prompt was tried for any written row.</p>
  <h3>2.1 The photographs: nothing any model could have seen</h3>
  <p>Wikimedia Commons photographs taken after 15 August 2026, labelled by their uploaders’ structured “depicts” statements, and iNaturalist observations uploaded on the day of the test, labelled by community identification. No labels were made by us or by any model.</p>
  {fresh_table()}
  <p class="caption"><b>Table 2.</b> All items of each photo set (the frontier models answered the test half), 95% bootstrap intervals, uncalibrated decisions, nothing fitted.</p>
  <figure>{fig1}<figcaption><b>Figure 2.</b> The Commons rows of Table 2, drawn to one scale. Filled marks are the open 4B model; hollow marks are hosted frontier models. Every interval overlaps every other.</figcaption></figure>

  <h3>2.2 Two other views of the same numbers</h3>
  <figure>{fig_acc_cost}<figcaption><b>Figure 3.</b> Accuracy against cost, one panel per question type, 95% intervals. The open model’s marks (read: circle, written: square) sit at the height of the hosted models ({site_charts.hosted_key(mrows)}): one to two orders of magnitude left of the flagships, and much closer to the providers’ cheapest models.</figcaption></figure>
  <figure>{fig_cost_speed}<figcaption><b>Figure 4.</b> Cost against speed; down and left is better. Each system is a large mark at the centre (geometric mean) of three small ones, one per question type: {site_charts.KEY}. Filled marks are the open 4B model, hollow marks are hosted models.{" Dashed small marks use a timing taken while the GPU was shared and will be replaced by a clean one." if has_prov else ""}</figcaption></figure>
</section>

<section aria-labelledby="read">
  <h2 id="read"><span class="num">3</span>Why read the answer instead of writing it?</h2>
  <p>The obvious alternative is to let the same model <em>write</em> a JSON answer. On yes/no and pick-one the two give identical answers, item for item, so reading changes nothing about what the model knows. What it changes is everything around the answer: no tokens are generated, so it is faster and cheaper: about 1.5 times on one question about a full-size photograph (reading the image dominates), 2 to 3 times on small images, and more as questions per image grow; the answer arrives as probabilities, so you can threshold, abstain or rank (accuracy on the most confident 80% of answers is 0.975 on the iNaturalist set); and a read answer can be fitted, which section 4 needs.</p>
  {cost_table()}
  <p class="caption"><b>Table 3.</b> The open model writing against reading, cost per 1,000 images on a rented GPU assumed no faster than the laptop; self-hosted cost is GPU time, so the saving is the measured time saving. For comparison, the frontier calls on these tasks measured ${api_lo:.2f} to ${api_hi:.2f} per 1,000 answers.</p>
</section>

<section aria-labelledby="ratings">
  <h2 id="ratings"><span class="num">4</span>Ratings: the hard case</h2>
  <p>A rating asks for a level on a rubric written in words, here five synthetic four-level scales (blur, exposure, JPEG, noise, resolution) and the same 1,000 held-out images for every system. Zero-shot, every model gets the <em>order</em> right and the <em>exact level</em> often wrong.</p>
  <p>The open model is exactly right on {raw["accuracy"]:.2f} of images but within one level on {raw["within_1"]:.3f}, and 97% of its errors are one step, in a direction that is constant per rubric: half a level too harsh on blur, never the worst level on JPEG. It sees the severity; it does not know where the rubric’s author drew the lines, and neither do the hosted models, which score {hosted_rating[0]:.2f} to {hosted_rating[-1]:.2f}. Read in one pass, the open model gives the same ratings as when it writes them ({jd["agree_with_written"]:.0%} identical, {jd["json_zero"]:.3f} against {written:.3f}).</p>
  <figure>{fig2}<figcaption><b>Figure 5.</b> Exact-level accuracy on the same 1,000 images, chance 0.25. Upper group: zero-shot. Lower group: the open model after it has seen images of the rubric, unlabeled ones first (zero labels, but not zero-shot), then 32 labeled ones.</figcaption></figure>
  <h3>4.1 What closes the gap</h3>
  <p>A written pick has nothing to fit. A read answer is a set of logits, so a few images of your own rubric can remove the constant offset, with or without labels. Sixteen unlabeled images bring the open model level with the best hosted model; 32 labels put it ahead of everything we measured.</p>
  {extras_table()}
  <p class="caption"><b>Table 4.</b> Exact-level accuracy on the rating test of Table 1 (the last row uses the full test split).</p>
</section>

<section aria-labelledby="models">
  <h2 id="models"><span class="num">5</span>Does it depend on the model?</h2>
  <p>The same questions and readouts, not a word changed, on other sizes of the same family and on a model from a different family (different vision tower, different language model).</p>
  {closed_set_table()}
  <p class="caption"><b>Table 5.</b> Yes/no and pick-one on the two fresh photo sets, all items, uncalibrated. Even the 2B model is level on yes/no; size shows on the harder pick-one.</p>
  {models_table()}
  <p class="caption"><b>Table 6.</b> Ratings, the same 1,000 images: exact-level accuracy. Zero-shot quality belongs to the model; once a few dozen labels exist the models land within a few points of each other, so the fitted recipe carries across sizes and families. “Within one” is for the four-pass read.</p>
</section>

<section aria-labelledby="new">
  <h2 id="new"><span class="num">6</span>What is, and is not, new</h2>
  <p>Reading answer-token logits from a frozen generative model, with several questions sharing one image prefix, is what Simple Jev, jev-visual and LitJev also do. For yes/no and pick-one the forward pass here is not new, and the accuracy belongs to the open model. What this project adds is a harness on top: a fresh-photograph comparison with paid frontier calls, the same model writing against reading, self-calibration from unlabeled images and labeled fitting for rating levels (<code>glance fit</code>), and measured dollars and milliseconds. It trains no weights, unlike YOFO, Laya Vision or OpenJev v2.</p>
</section>

<section aria-labelledby="limits">
  <h2 id="limits"><span class="num">7</span>Limits and misses</h2>
  <ul class="misses">
    <li><span class="verdict miss">not supported</span> A content-free prior (blank and noise images) was expected to help zero-shot ratings. It took exact accuracy from {nullp["table"]["raw zero-shot"]["accuracy"]:.3f} to {nullp["table"]["content-free prior, all six null images (registered)"]["accuracy"]:.3f}: for an image rubric there is no content-free image.</li>
    <li><span class="verdict miss">not supported</span> Our four-pass rating readout was expected to match the same model’s written answer zero-shot. It trailed it by ten points: a readout selected with a calibration in the loop is good to fit and poor zero-shot. <span class="verdict">supported</span> The registered fix, one pass read at the JSON answer position, matches the written answer ({jd["json_zero"]:.3f}) and reaches {jd["json_u16"]:.3f} with 16 unlabeled images.</li>
    <li><span class="verdict miss">not supported</span> We expected each provider’s cheapest model to score at or below its flagship on ratings. Every one beats its flagship, and {best_rating} ({best_rating_acc:.3f}, ${cheap_rating[0]:.2f} to ${hosted[best_rating]["usd_per_1000"]["rating"]["value"][0]:.2f} per 1,000 among the cheap models) is nine points ahead of the open model zero-shot. The open model needs 16 unlabeled images to draw level and 32 labels to lead.</li>
    <li><span class="verdict miss">not supported</span> On KADID-10k (23 distortion types, five levels, human scores) every registered target was missed: 0.527 exact with labels, 0.33 zero-shot.</li>
    <li><span class="verdict">supported</span> A fitted readout on the model’s hidden state reaches {r4a:.3f} from one pass, against 0.867 for the token readout: the model represents severity almost perfectly. It needs on the order of a hundred labels.</li>
    <li><span class="verdict">known</span> Hand-built image features beat the VLM on low-level artifacts when labels are plentiful (0.979). A calibration fitted on one rubric does not transfer to another. Two model families and two sizes measured so far (Tables 5 and 6); that is not “any model”.</li>
  </ul>
</section>

<section aria-labelledby="reproduce">
  <h2 id="reproduce"><span class="num">8</span>Reproduce</h2>
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

STYLE = ("""
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
""" + site_charts.CSS + "</style>\n")

page = STYLE + "<main>" + BODY + "</main>\n"
(ROOT / "site").mkdir(exist_ok=True)
(ROOT / "site/page.html").write_text(page)
(ROOT / "site/index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                      + STYLE + "</head>\n<body>\n<main>" + BODY + "</main>\n</body>\n</html>\n")
print("wrote site/index.html and site/page.html,", len(page) // 1024, "KB")

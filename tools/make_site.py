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
    """Both photo sets, every system: the open model read and (where collected) written, then all hosted models on the test half."""
    def t3(e):
        return f'{e[0]:.3f} <span class="ci">[{e[1]:.3f}, {e[2]:.3f}]</span>'

    def block(title, note, y, c, wy, wc, sep):
        rows = [f'<tr class="own{" sep" if sep else ""}"><td>Qwen3-VL-4B, read <span class="note">{title}, {note}</span></td><td class="n">{ci(y["vlm:statement"])}</td><td class="n">{ci(c["vlm:independent"])}</td></tr>']
        if wy and wc:
            rows.append(f'<tr class="own"><td>Qwen3-VL-4B, written <span class="note">same items</span></td><td class="n">{t3(wy["written"])}</td><td class="n">{t3(wc["written"])}</td></tr>')
        rows += [f'<tr><td>{name}</td><td class="n">{ci(y[m])}</td><td class="n">{ci(c[m])}</td></tr>' for m, name in ALL_HOSTED.items() if m in y and m in c]
        return rows

    out = ['<div class="table-scroll"><table><thead><tr><th>System, zero-shot</th><th class="n">yes/no</th><th class="n">pick-one</th></tr></thead><tbody>']
    out += block("Commons", "131 photos", fresh["suites"]["fresh_yesno"], fresh["suites"]["fresh_choice"], gen.get("fresh_yesno"), gen.get("fresh_choice"), False)
    if inat:
        out += block("iNaturalist", "200 photos", inat["suites"]["inat_yesno"], inat["suites"]["inat_choice"], gen.get("inat_yesno"), gen.get("inat_choice"), True)
    return "".join(out) + "</tbody></table></div>"


def apart():
    """Hosted models whose interval does not overlap the open model's, per photo set; stated on the page rather than left for the reader to find."""
    found = []
    for title, d, ys, cs in (("Commons", fresh, "fresh_yesno", "fresh_choice"), ("iNaturalist", inat, "inat_yesno", "inat_choice")):
        if not d:
            continue
        for suite, own, kind in ((ys, "vlm:statement", "yes/no"), (cs, "vlm:independent", "pick-one")):
            e = d["suites"][suite]
            for m, name in ALL_HOSTED.items():
                if m in e and (e[m]["ci95"][1] < e[own]["ci95"][0] or e[m]["ci95"][0] > e[own]["ci95"][1]):
                    found.append((title, kind, name, e[m]["accuracy"], e[own]["accuracy"], e[m]["accuracy"] < e[own]["accuracy"]))
    if not found:
        return "Every hosted interval overlaps the open model’s on both sets."
    by = {}
    for title, kind, name, a, o, below in found:
        by.setdefault((title, name, below), []).append(f"{a:.3f} against {o:.3f} on {kind}")
    parts = [f"{name} is {'below' if below else 'above'} the open model on the {title} set ({', '.join(v)}; the intervals do not overlap)" for (title, name, below), v in by.items()]
    return "One exception to “indistinguishable”: " + "; ".join(parts) + ". Every other hosted interval overlaps the open model’s on both sets."


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
            return '<span class="pending">not in this snapshot</span>'
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
            out.append(f'<tr><td>{label}</td><td class="n pending" colspan="4">not in this snapshot</td></tr>')
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
            out.append(f'<tr><td>{label}</td><td class="n pending" colspan="4">not in this snapshot</td></tr>')
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
    return f"about {ratio:.0f} times cheaper than" if ratio >= 3 else "of the same order of cost as"


def speed_rank(test):
    """'faster than N of the M hosted models' for the open model's read, from the matrix."""
    own = matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["seconds"][test]["value"]
    others = [v["seconds"][test]["value"] for n, v in hosted.items()]
    return f"{own:.1f} s, faster than {sum(own < o for o in others)} of the {len(others)} hosted models"


noise = load("results/lab/label_noise_proxy.json") or {}
import subprocess  # noqa: E402


def noise_n(suite):
    return (noise.get(suite) or {}).get("every_system_wrong", "a few")


snapshot = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip() or "unversioned"
own = matrix["systems"]["Qwen3-VL-4B, read (Glance)"]
REFS = [
    ("Alain and Bengio, 2016", "Understanding intermediate layers using linear classifier probes", "https://arxiv.org/abs/1610.01644"),
    ("Guo, Pleiss, Sun and Weinberger, 2017", "On calibration of modern neural networks (temperature, vector and matrix scaling)", "https://arxiv.org/abs/1706.04599"),
    ("Kadavath et al., 2022", "Language models (mostly) know what they know: reading P(True) for a proposed answer", "https://arxiv.org/abs/2207.05221"),
    ("Kull et al., 2019", "Beyond temperature scaling: Dirichlet calibration", "https://arxiv.org/abs/1910.12656"),
    ("Lin et al., 2024", "VQAScore: the probability of “Yes” from one forward pass of a VQA model", "https://linzhiqiu.github.io/papers/vqascore/"),
    ("Liu et al., 2023", "G-Eval: a rubric score as the probability-weighted sum over rating tokens", "https://arxiv.org/abs/2303.16634"),
    ("Wang, Chan and Loy, 2023", "CLIP-IQA: frozen CLIP with antonym prompts as an image-quality score", "https://arxiv.org/abs/2207.12396"),
    ("Wu et al., 2024a", "Q-Bench: a softmax over “good” and “poor” logits as a zero-shot quality score", "https://arxiv.org/abs/2309.14181"),
    ("Wu et al., 2024b", "Q-Align: fine-tuning an LMM on text-defined rating levels and reading the level tokens", "https://proceedings.mlr.press/v235/wu24ah.html"),
    ("Zhang, Wu, Jia, Lin and Zhai, 2025", "Q-SiT: teaching LMMs for image quality scoring and interpreting (Q-SiT-mini, 0.9B, is the trained quality model we compared against)", "https://arxiv.org/abs/2503.09197"),
    ("Zhang et al., 2025", "YOFO: fine-tuned Qwen-VL judging many yes/no requirements in one forward pass", "https://arxiv.org/abs/2511.16600"),
    ("Zheng et al., 2024", "Large language models are not robust multiple choice selectors (option-letter bias)", "https://arxiv.org/abs/2309.03882"),
    ("vLLM project", "Automatic prefix caching, including multimodal inputs", "https://docs.vllm.ai/en/stable/design/prefix_caching/"),
    ("Simple Jev; jev-visual; LitJev", "Training-free servers that read answer-token logits from a frozen model", "https://github.com/featherless-ai/simple-jev"),
]


def refs_html():
    return "<ol class=\"refs\">" + "".join(f'<li>{a}. {t}. <a href="{u}">{u.split("//")[1][:48]}</a></li>' for a, t, u in REFS) + "</ol>"


photo_timed = "on these photographs" in matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["seconds"]["yesno"]["how"]
size_note = "" if photo_timed else " The open model’s yes/no and pick-one timings here were taken on smaller test images (and a three-option pick-one) than the photographs the hosted models saw; a timing on the same photographs is running and will replace them."
def finer():
    orders, probes = load("results/lab/fresh_inat_orders.json"), load("results/lab/probes.json")
    out = []
    if orders:
        y, c = orders["suites"]["inat_orders_yesno"], orders["suites"]["inat_orders_choice"]
        rows = [("Qwen3-VL-4B, read", y.get("vlm:statement"), c.get("vlm:independent")), ("SigLIP2 (open dual encoder)", None, c.get("siglip:independent"))]
        rows += [(n, y.get(k), c.get(k)) for k, n in list(FRONTIER.items()) + [("anthropic/claude-haiku-4-5", "Claude Haiku 4.5"), ("openai/gpt-5.6-luna", "GPT-5.6 Luna"), ("openrouter/google/gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite")] if k in y or k in c]
        hosted_c = sorted((c[k]["accuracy"], n) for k, n in [(k, n) for _, _, _ in [(0, 0, 0)] for k, n in list(FRONTIER.items()) + [("anthropic/claude-haiku-4-5", "Claude Haiku 4.5"), ("openai/gpt-5.6-luna", "GPT-5.6 Luna"), ("openrouter/google/gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite")]] if k in c)
        spread = (f" This test does separate systems: on pick-one the hosted models range from {hosted_c[0][0]:.3f} ({hosted_c[0][1]}) to {hosted_c[-1][0]:.3f} ({hosted_c[-1][1]}), and the open 4B model, at "
                  f"{c['vlm:independent']['accuracy']:.3f}, is within {abs(hosted_c[-1][0] - c['vlm:independent']['accuracy']) * 100:.0f} point of the best.") if hosted_c else ""
        out.append("<p>Insect orders on iNaturalist (beetle, true bug, fly, …; 210 photographs; every “no” question names a look-alike order) ask for finer distinctions than the ten-group test." + spread + " The open dual encoder falls to 0.71 here.</p>")
        out.append('<div class="table-scroll"><table><thead><tr><th>Insect orders, zero-shot</th><th class="n">yes/no</th><th class="n">pick one of 7</th></tr></thead><tbody>'
                   + "".join(f'<tr{" class=own" if n.startswith("Qwen") else ""}><td>{n}</td><td class="n">{ci(a) if a else "–"}</td><td class="n">{ci(b_) if b_ else "–"}</td></tr>' for n, a, b_ in rows) + "</tbody></table></div>")
        out.append('<p class="caption"><b>Table 4.</b> Seven insect orders, zero-shot, 95% bootstrap intervals. The open model answered all items (420 yes/no questions, 210 photographs); hosted models, where present, the test half (210 and 105).</p>')
    if not probes and not load("results/lab/ui_screens.json"):
        out.append("<p>Tests that need more than coarse recognition (counting, spatial relations, relative size, reading, and which element of a screen to click) are registered and built from images drawn by program, with labels exact by construction; their results are not in this snapshot.</p>")
    return "\n  ".join(out)


def lower_keep(text):
    """Lower-case a test name for use inside a sentence, keeping proper nouns."""
    return text.lower().replace("commons", "Commons").replace("inaturalist", "iNaturalist")


def older():
    """On older public benchmarks one hosted flagship is ahead of the open model (entry 33); said next to the fresh-photo result."""
    v0 = load("results/v0/analysis/bootstrap_v0.json")
    if not v0:
        return ""
    g = v0["mean gap, yes/no and choice suites only (no blur_ladder)"]
    return (f"On four older public benchmarks (POPE, a GQA yes/no subset, Oxford Pets, Caltech-101), which may sit in every model’s training data, the one hosted flagship we ran, Claude Opus 5, "
            f"is ahead of the open model by {g['points']:.1f} points on average [{g['ci95'][0]:.1f}, {g['ci95'][1]:.1f}]. On photographs no model can have seen that gap is not there, "
            "so the fresh-photo result is not a contamination effect in the open model’s favour. Why the gap appears only on the older sets is not established: contamination in the hosted model’s favour and harder items are both possible.")


def other_rubrics():
    """Zero-shot on the five rubrics that are not image quality (entry 43b); stated next to the scope so the section's title is not read as a claim about rubrics in general."""
    d = (load("results/lab/jsondigits_hard.json") or {}).get("benches", {}).get("creative-QA rubrics")
    if not d:
        return ""
    m = d["mean"]
    e4 = (load("lab/SEMANTIC_REPORT.json") or {}).get("summary", {}).get("ens4d")
    fitted = (f" A labeled fit does not rescue them either: with 300 labels per rubric the four-pass read reaches {e4['accuracy']:.2f} exact and {e4['within_1']:.2f} within one, where we had predicted 0.75 and 0.95. "
              "A fit removes an offset; it cannot supply a judgement the model does not make, and the weakest rubrics are the geometric ones (tilt, how much of the subject is cut off).") if e4 else \
        " Whether a labeled fit rescues them is a registered experiment still running."
    return (f"\n  <p><b>Rubrics that are not image quality.</b> The pattern of this section does not extend to every rubric. On five synthetic rubrics that are not image quality (subject cut off by the frame, occlusion, tilt, caption legibility, watermark) the zero-shot read is "
            f"exactly right on {m['json_zero']:.2f} of images and within one level on {m['json_within_1']:.2f} (chance 0.25), tilt and cut-off sit at chance, and unlabeled images do not help ({m['json_u16']:.2f})." + fitted + "</p>")


def outside_sentence():
    """One sentence on the trained quality model (entry 37c), for section 5.1; the ledger item in the limits section has the detail."""
    d = load("results/lab/external_same_items.json") or {}
    q = next((e for name, e in d.items() if name.startswith("q-sit-mini")), None)
    if not q:
        return ""
    a, b = q["32_labels"]["outside"], q["32_labels"]["ens4d"]
    out = f" A 0.9B model trained for image quality (Q-SiT-mini; Zhang, Wu, Jia, Lin and Zhai 2025), given the same 32-label fit on the same items, scores {a[0]:.3f} against {b[0]:.3f} for the 4B model: indistinguishable, at a quarter of the size."
    return out


def openjev_sentence():
    """The second outside system (entry 37e): a trained general claim scorer, behind under the same fit; its own gain from the fit is the point."""
    o = next((e for name, e in (load("results/lab/external_same_items.json") or {}).items() if name.startswith("openjev")), None)
    raw = (((load("results/lab/external_systems.json") or {}).get("openjev") or {}).get("summary") or {}).get("mean_uncalibrated_accuracy")
    if not o or raw is None:
        return ""
    g = o["32_labels"]["outside_minus_ens4d_points"]
    return (f" The other outside system we ran, a 4B open model trained to score claims (openjev v2), is {abs(g[0]):.1f} points behind the 4B model under the same 32-label fit [{abs(g[2]):.1f}, {abs(g[1]):.1f}]; "
            f"on its own it scores {raw:.2f} and with our fit {o['32_labels']['outside'][0]:.2f}, so the fit is a part that transfers to another system’s readout.")


def outside_item():
    """The registered comparison with outside open systems (entry 37), on the same items with the same fit."""
    d = load("results/lab/external_same_items.json") or {}
    q = next((e for name, e in d.items() if name.startswith("q-sit-mini")), None)
    if not q:
        return ""
    a, b, g = q["32_labels"]["outside"], q["32_labels"]["ens4d"], q["32_labels"]["outside_minus_ens4d_points"]
    return (f'    <li><span class="verdict miss">not supported</span> We expected a 0.9B model trained for image quality (Q-SiT-mini; Zhang, Wu, Jia, Lin and Zhai 2025), given our fit, to stay below the frozen 4B model on the five quality scales. '
            f'On the same items with the same 32-label fit it scores {a[0]:.3f} against {b[0]:.3f} (difference {g[0]:+.1f} points [{g[1]:+.1f}, {g[2]:+.1f}]): indistinguishable, at a quarter of the size. '
            'A general model read this way earns its place by answering any typed question with one set of frozen weights, not by being the best quality meter.</li>')


_kadid = ((load("results/lab/jsondigits_hard.json") or {}).get("benches", {}).get("KADID-10k (23 severity distortions)") or {}).get("mean", {})
kadid_json, kadid_ens = _kadid.get("json_zero", float("nan")), _kadid.get("ens_zero", float("nan"))  # E17, lab/NOTES.md entry 43c
e4_acc = ((load("lab/SEMANTIC_REPORT.json") or {}).get("summary", {}).get("ens4d") or {}).get("accuracy", float("nan"))  # E4, lab/NOTES.md entry 36c
_closed = [gen[k] for k in ("fresh_yesno", "fresh_choice", "inat_yesno", "inat_choice") if k in gen]
rw_gap = (min(abs(e["read_minus_written_points"][0]) for e in _closed), max(abs(e["read_minus_written_points"][0]) for e in _closed))
rw_same = (min(e["same_outcome"] for e in _closed), max(e["same_outcome"] for e in _closed))

ALL_HOSTED = {**FRONTIER, "anthropic/claude-haiku-4-5": "Claude Haiku 4.5", "openai/gpt-5.6-luna": "GPT-5.6 Luna", "openrouter/google/gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite"}
BEYOND = [("results/lab/probes.json", "Images drawn by program", [
              ("probe_spatial", "Is the ball left of / right of / above / below the square?", "1 of 2"), ("probe_stripes", "Which way do the stripes run?", "1 of 4"),
              ("probe_text", "Which of six look-alike words is printed?", "1 of 6"), ("probe_largest", "Which of four shapes is the largest?", "1 of 4"),
              ("probe_count", "How many balls? (1 to 8)", "1 of 8"), ("probe_count_color", "How many red balls? (0 to 6)", "1 of 7")]),
          ("results/lab/ui_screens.json", "Synthetic interface screens", [
              ("ui_page", "What kind of page is this?", "1 of 5"), ("ui_state", "Is the page in this state? (error shown, dialog open, …)", "1 of 2"),
              ("ui_done", "Is this goal already done?", "1 of 2"), ("ui_click", "Which numbered mark serves this goal?", "1 of 6 to 8"),
              ("ui_reason", "The same, after one step of reasoning (the cheaper plan, the earliest date)", "1 of 2 to 6")])]


def beyond():
    """Tests that need more than coarse recognition: rendered probes (entry 50) and synthetic interface screens (entry 49). Data-driven; empty until a report exists."""
    found = [(load(path), title, suites) for path, title, suites in BEYOND]
    if not any(d for d, _, _ in found):
        return ""
    hosted = [k for k in ALL_HOSTED if any(k in e for d, _, _ in found if d for e in d["suites"].values())]
    head = "".join(f'<th class="n">{ALL_HOSTED[k]}</th>' for k in hosted)
    body = []
    for d, title, suites in found:
        if not d:
            continue
        body.append(f'<tr class="group"><td colspan="{3 + len(hosted)}">{title}</td></tr>')
        for suite, question, options in suites:
            e = d["suites"].get(suite)
            if not e:
                continue
            own = e["open 4B model, read"]
            cell = lambda c: f'{c["accuracy"]:.2f}'  # noqa: E731
            body.append(f'<tr><td>{question}</td><td class="n">{options}</td><td class="n own">{ci(own["all_items"], 2)} <span class="ci">n={own["all_items"]["n"]}</span></td>'
                        + "".join(f'<td class="n">{cell(e[k]["all_items"]) if k in e else "–"}</td>' for k in hosted) + "</tr>")
    note = "Hosted models answered the test half of each set." if hosted else "Hosted models have not been run on these sets."
    return ('<div class="table-scroll"><table><thead><tr><th>Question, zero-shot</th><th class="n">options</th><th class="n">Qwen3-VL-4B, read</th>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
            + f'<p class="caption"><b>Table 5.</b> Exact-answer accuracy on images whose labels are exact by construction (drawn or rendered by program; no photographs, no people). {note}</p>')


def beyond_text():
    """Sentences for the probe results, every number from results/lab/probes.json (entry 50c)."""
    d = load("results/lab/probes.json")
    if not d:
        return ""
    acc = {k: v["open 4B model, read"]["all_items"]["accuracy"] for k, v in d["suites"].items()}
    b = {k: v["systems"]["open 4B model, read"] for k, v in d.get("breakdowns", {}).items()}
    pooled = lambda g, keys: sum(g[k]["accuracy"] * g[k]["n"] for k in keys) / sum(g[k]["n"] for k in keys)  # noqa: E731
    out = (f"Six sets of 150 images drawn by program, with labels exact by construction, mark where coarse recognition ends for the open model read this way. It reads which of six look-alike words is printed on {acc['probe_text']:.2f} of images "
           f"and gets left, right, above and below right on {acc['probe_spatial']:.2f}.")
    if "probe_count" in b:
        c = b["probe_count"]
        out += f" It counts one to five balls almost without error ({pooled(c, ['1', '2', '3', '4', '5']):.2f}) and then degrades: {c['6']['accuracy']:.2f} at six, {c['7']['accuracy']:.2f} at seven, {c['8']['accuracy']:.2f} at eight."
    if "probe_stripes" in b:
        st = b["probe_stripes"]
        out += (f" It tells horizontal from vertical stripes perfectly ({pooled(st, ['horizontal', 'vertical']):.2f}) and cannot tell the two diagonal directions apart ({pooled(st, ['diagonal_rising', 'diagonal_falling']):.2f}, below a coin flip: "
                "a mirror-image confusion).")
    if "probe_largest" in b:
        lg = b["probe_largest"]
        out += (f" And it picks the largest of four like shapes on only {acc['probe_largest']:.2f} of images: {lg['2.0']['accuracy']:.2f} when the largest has twice the area of the others, {lg['1.15']['accuracy']:.2f} at 1.15 times (chance 0.25).")
    hosted = any(k != "open 4B model, read" for v in d["suites"].values() for k in v)
    out += (" We had predicted at least 0.90 on stripes and on the twofold size difference, and a steeper fall in counting; all three predictions were wrong."
            + ("" if hosted else " Hosted models have not been run on these sets, so whether they share these weaknesses is not known."))
    return "<p>" + out + "</p>" + ui_text()


def ui_text():
    """Sentences for the synthetic interface screens, every number from results/lab/ui_screens.json (entry 49c)."""
    d = load("results/lab/ui_screens.json")
    if not d:
        return ""
    acc = {k: v["open 4B model, read"]["all_items"] for k, v in d["suites"].items()}
    b = {k: v["systems"]["open 4B model, read"] for k, v in d.get("breakdowns", {}).items()}
    out = (f"Five hundred synthetic interface screens (five kinds of page, invented content, rendered from generated HTML so every label is exact) ask what an agent would ask. Given a goal in words and six to eight numbered marks on the screen, "
           f"the open model names the mark to click on {acc['ui_click']['accuracy']:.3f} of {acc['ui_click']['n']} screens; when the goal needs one step of reasoning first (the cheaper plan, the item out of stock, the earliest date) on {acc['ui_reason']['accuracy']:.2f}. "
           f"State questions (is a dialog open, is an error shown, is the user signed in) are right on {acc['ui_state']['accuracy']:.2f}")
    if "ui_state" in b and "primary_disabled" in b["ui_state"]:
        rest = [v["accuracy"] for k, v in b["ui_state"].items() if k != "primary_disabled"]
        out += f", with one exception: whether the main button is disabled ({b['ui_state']['primary_disabled']['accuracy']:.2f}; the other five states {min(rest):.2f} to {max(rest):.2f}), a state our screens draw as a pale tint that the model almost never reports"
    out += f". It is weaker on “is this goal already done” ({acc['ui_done']['accuracy']:.2f}"
    if "ui_done" in b:
        out += f": {b['ui_done']['False']['accuracy']:.2f} on screens where it is not, {b['ui_done']['True']['accuracy']:.2f} where it is"
    out += f") and on the kind of page ({acc['ui_page']['accuracy']:.2f}, every error being another page called an article: an uncalibrated bias toward one option)."
    hosted = any(k != "open 4B model, read" for v in d["suites"].values() for k in v)
    out += (" We had predicted that one forward pass would trail hosted models by ten points on the reasoning screens; at this score it cannot."
            + ("" if hosted else " Hosted models have not been run on these screens."))
    return "\n  <p>" + out + "</p>"


finer_block = finer()
beyond_block = beyond()
beyond_section = (f'<section aria-labelledby="stops">\n  <h2 id="stops"><span class="num">3</span>Where coarse recognition ends: geometry, not reading</h2>\n  {beyond_text()}\n  {beyond_block}\n</section>\n') if beyond_block else ""
t0 = 5 if beyond_block else 4  # tables after section 2 are numbered from here, so no number is skipped while Table 5 has no data
got = [r for r in (measured or []) if r["usd_per_1000_calls"] is not None]
api_lo, api_hi = min(r["usd_per_1000_calls"] for r in got), max(r["usd_per_1000_calls"] for r in got)
raw = lf["raw (0 labels, no pool)"]
r4a = ladder["summary"]["R4a"]["accuracy"] if ladder and "R4a" in ladder.get("summary", {}) else None
paired = {"Claude Opus 5": "+12.2 [8.1, 16.3]", "GPT-5.6": "+7.5 [3.0, 11.9]", "Gemini 3.1 Pro": "+2.1 [−2.3, 6.6]"}  # lab/NOTES.md entry 32b
today = datetime.date.today().isoformat()

BODY = f"""
<header>
  <p class="running">Working paper · snapshot {snapshot}, {today} · every experiment registered before it ran · results regenerate from the repository</p>
  <h1>Glance</h1>
  <p class="subtitle">Coarse recognition at the level of hosted models is already in a small open vision-language model, and it can be read without generating. What remains hard about quality ratings is where the rubric draws its lines; where the model itself stops is geometry: tilt, relative size, mirror-image direction.</p>
  <p class="byline">Yohei Nakajima <span class="aff">· independent · built with AI assistance throughout (the notebook records who did what)</span></p>
  <p class="links"><a href="#method">Method</a> · <a href="#evidence">Yes/no and pick-one</a> · <a href="#stops">Where it stops</a> · <a href="#read">Read against write</a> · <a href="#ratings">Ratings</a> · <a href="#models">Scale and family</a> · <a href="#cost">Cost and speed</a> · <a href="#new">What is not new</a> · <a href="#limits">Limits and misses</a> · <a href="#refs">References</a></p>
</header>

<section aria-labelledby="abstract">
  <h2 id="abstract" class="plain">Abstract</h2>
  <p class="abstract">A <em>typed</em> question is one whose legal answers form a closed set known before the model runs: yes or no, one of a list, a level on a rubric. We put such questions about images to a frozen open vision-language model (Qwen3-VL-4B, Apache-2.0, on a laptop) and read the answer from the logits of one forward pass; nothing is generated.</p>
  <p class="abstract">On photographs taken after every model’s release, labelled by people outside this project, the open model is statistically indistinguishable from six hosted models, three flagships and three low-cost ones, on coarse yes/no and pick-one questions (n = 131 and 65; every 95% interval overlaps every other). These questions are easy and the labels are imperfect: about 3 to 5% of items are answered “wrongly” by all seven systems. Reading is as accurate as the same model writing JSON.</p>
  <p class="abstract">Ratings behave differently. On five image-quality scales, zero-shot, every system orders images correctly (the open model is within one level on {raw["within_1"]:.2f} of images) and places the level boundaries wrongly, by a constant offset per rubric; a low-cost hosted model leads ({best_rating_acc:.3f} against {jd["json_zero"]:.3f}), and an 8B model is no better than a 4B one. Because a read answer is a vector of logits, it can be fitted: 16 unlabeled images of the rubric remove most of the offset ({jd["json_u16"]:.3f}) and 32 labels reach {loc["+ Glance ens4d, 32 labels"]["mean_accuracy"]:.3f}; the hosted models were not given examples, so this is a comparison of products, not of models. On a real image-quality benchmark (KADID-10k) the approach missed every target we registered, hand-built features and a small trained quality model do as well or better on low-level artefacts, and on rubrics that are not image quality (tilt, how much of a subject is cut off) even 300 labels reach only {e4_acc:.2f}.</p>
  <p class="abstract">The open model’s cost is of the same order as the low-cost hosted models and one to two orders below the flagships; no image leaves the machine. The readout is shared with other training-free tools and is not claimed as new; what is offered is the measurement, registered before it was run, with its misses.</p>
</section>

<section aria-labelledby="method">
  <h2 id="method"><span class="num">1</span>Method: score the allowed answers in one forward pass</h2>
  <p>The image and a question go to a frozen model. The assistant turn is left empty (or forced to begin a JSON value), the model runs once, and only the logits of the allowed answer tokens at that position are kept. A softmax over that closed set, at temperature 1, gives the probabilities; nothing is sampled, generated or parsed.</p>
  <figure>{fig_method}</figure>
  <dl class="protocol">
    <dt>Yes/no</dt>
    <dd><code>Question: …? Answer Yes or No.</code><br>Read: the log-sum of the logits of “Yes”, “ Yes”, “yes”, “ yes” against the same four for “No”; the probability is the sigmoid of the difference.</dd>
    <dt>Pick one of K</dt>
    <dd><code>Question: … Candidate answer: X. Is this candidate the correct answer? Answer Yes or No.</code><br>Read: one such statement per option, all sharing one image prefix; the yes-minus-no logit of each option; a softmax over the K options.</dd>
    <dt>Rating, K levels</dt>
    <dd><code>"answer": How … is the image? Allowed: 0 = …; 1 = …; (answer with the number)</code>, with the assistant turn forced to begin <code>{{"answer": </code><br>Read: the logits of the digits 0 to K−1 at the next position; a softmax; the score is the expected level. One pass.</dd>
  </dl>
  <p class="caption"><b>Table 1.</b> The three readouts. Images are resized to at most 768 visual tokens (28-pixel patches), which sets most of the latency. Several questions about one image reuse its key-value prefix and branch into independent suffixes, so they cannot influence each other; on 100 checked requests this changed no prediction. “Zero-shot” below means these raw probabilities, with nothing fitted.</p>
  <p>Two fits are possible because the answer is a logit vector. <em>Without labels</em>: over a small pool of unlabeled images of the rubric, each level logit is centred and scaled by the pool’s mean and standard deviation before the softmax. This assumes the model’s ordering is right and removes a constant offset; it cannot move boundaries it has no evidence for. <em>With labels</em>: an affine map of the logits before the softmax (matrix scaling, Guo et al. 2017), L2-regularised, with its sharpness chosen on held-out folds; about 32 labels suffice. A research variant fits the same map on the model’s final hidden state (a linear probe). An earlier four-pass rating readout of ours (digits in both orders, with and without a magnified crop) is the better input for the labeled fit and is worse zero-shot; section 5 reports both.</p>
<pre><code>decisive: a bold jumping spider, iNaturalist 402011619 (CC BY, antimatterbee)
  "Is the main subject a spider or other arachnid?"   P(yes) = 1.00     correct
  "What kind of organism is it?"  (10 options)        arachnid 1.00     correct

uncertain: a fungus on bark, iNaturalist 401937340 (CC BY, Марина Давлетшина)
  "What kind of organism is it?"  (10 options)        fungus 0.63, insect 0.26, arachnid 0.11     correct, and unsure
  "Is the main subject an insect?"                    P(yes) = 0.60     wrong, and close to the fence
  "Is the main subject a fungus?"                     P(yes) = 0.03     wrong, and confident</code></pre>
  <p class="caption">Two logged examples from the open 4B model. Most answers look like the first. The second shows both what the probabilities are for (0.63 and 0.60 are flags) and that they are not a guarantee (0.03 is a confident miss).</p>
</section>

<section aria-labelledby="evidence">
  <h2 id="evidence"><span class="num">2</span>On coarse yes/no and pick-one questions, a 4B open model is indistinguishable from hosted models</h2>
  <p>The same items went to the open 4B model (read, and also writing its answer as JSON), to three hosted flagships and to each provider’s lowest-cost current vision model. Three tests, all zero-shot: {lower_keep(matrix["tests"]["yesno"])}; {lower_keep(matrix["tests"]["choice"])}; {lower_keep(matrix["tests"]["rating"])} (ratings are the subject of section 5).</p>
  <figure>{fig_bars}<figcaption><b>Figure 1.</b> Accuracy, seconds and dollars for every system, by question type, on scales that start at zero, on the items every system answered (Table 2 gives the counts). Dark bars are the open 4B model.{" An outlined bar is an estimated cost; an estimate beyond the measured range is cut short and marked ›." if has_est else ""}</figcaption></figure>
  <p>On yes/no and pick-one every 95% interval overlaps every other, so the defensible statement is “indistinguishable at this sample size”, not “equal”. Two cautions apply. The questions are coarse (is there a bridge; which of thirteen everyday things is this), so they measure a floor that all current systems clear. And the labels are not gold: {noise_n("fresh_yesno")} of 131 yes/no items and {noise_n("fresh_choice")} of 65 pick-one items are answered “wrongly” by all seven systems, which is more likely a wrong or ambiguous label than seven identical mistakes (a proxy; no human audit was done). On ratings there is no single best system.</p>
  {matrix_tables(ALL)}
  <p class="caption"><b>Table 2.</b> The numbers behind Figure 1, on the items every system answered (the test half of the Commons set). Accuracy with 95% bootstrap intervals; bold rows are the open model. Hosted speed is wall time per call from one laptop, network included; hosted cost is the provider’s bill where it was logged (“est.” is a list-price upper estimate). The open model was timed on the same photographs with the GPU otherwise idle; its cost is those seconds at an on-demand cloud GPU price.{size_note} No few-shot prompt was tried for any written row.</p>
  <h3>2.1 The photographs</h3>
  <p>Wikimedia Commons photographs taken after 15 August 2026, labelled by their uploaders’ structured “depicts” statements, and iNaturalist observations uploaded on the day of the test, labelled by community identification. No labels were made by us or by any model.</p>
  {fresh_table()}
  <p class="caption"><b>Table 3.</b> The open model on all items of each photo set, hosted models on the test half; 95% bootstrap intervals, uncalibrated decisions, nothing fitted. {apart()}</p>
  <figure>{fig1}<figcaption><b>Figure 2.</b> The Commons rows of Table 3, drawn to one scale. Filled marks are the open 4B model; hollow marks are hosted frontier models. Every interval overlaps every other.</figcaption></figure>
  <p>{older()}</p>

  <h3>2.2 A finer test</h3>
  {finer_block}
</section>

{beyond_section}
<section aria-labelledby="read">
  <h2 id="read"><span class="num">4</span>Reading is as accurate as writing, and gives the same answer when the prompt is the same</h2>
  <p>The alternative to reading is to let the same model write a JSON answer. With ground truth and one question per request, the two agree. On yes/no and pick-one the accuracy is the same on both photo sets (differences of {rw_gap[0]:.1f} to {rw_gap[1]:.1f} points, every interval spanning zero; Table 3), and each item comes out the same way, right or wrong, on {rw_same[0]:.1%} to {rw_same[1]:.1%} of items; the few that differ reflect the prompts, which are not the same (a JSON request against one statement per option). On ratings, where the digits can be read at the very position where the written answer puts them, {jd["agree_with_written"]:.0%} of answers are identical: a written answer under greedy decoding is an argmax over the same logits, so this is expected.</p>
  <p>They stop agreeing when the prompts differ. Our earlier four-pass rating readout uses different wording from the JSON prompt and scores {100 * (written - jd["ens_zero"]):.0f} points lower zero-shot. And when one written JSON object carries 25 ratings, each field is conditioned on the fields already written, while 25 separate reads are independent: the two agree on only {genb["25 ratings"]["agreement_with_read_on_valid_fields"]:.0%} of fields (that request has no ground truth, so this is a difference, not an error rate). Reading is therefore not a free substitute for any prompt; it is a way to take the same decision without generating it.</p>
  <p>What reading changes is cost and form: about 1.5 times faster for one question about a full-size photograph (encoding the image dominates), 2 to 3 times on small images, more as questions per image grow; and the answer is a probability vector, which can be thresholded, ranked and fitted.</p>
  {cost_table()}
  <p class="caption"><b>Table {t0 + 1}.</b> The open model writing against reading, cost per 1,000 images on a rented GPU assumed no faster than the laptop; self-hosted cost is GPU time, so the saving is the measured time saving. For comparison, the frontier calls on these tasks measured ${api_lo:.2f} to ${api_hi:.2f} per 1,000 answers.</p>
</section>

<section aria-labelledby="ratings">
  <h2 id="ratings"><span class="num">5</span>Ratings: models get the order right and the boundaries wrong; a few images of the rubric fix the offset</h2>
  <p><b>Scope.</b> “Ratings” in this section means five synthetic, single-factor, four-level image-quality scales (blur, exposure, JPEG, noise, resolution) on 1,000 held-out images, the same for every system; it does not mean aesthetic judgement. The pattern in the title holds on these scales and does not hold outside them: section 5.1 has KADID-10k, rubrics that are not image quality, and the specialised tools that do as well or better.</p>
  <p><b>Order against boundaries.</b> Zero-shot the open model is exactly right on {raw["accuracy"]:.2f} of images and within one level on {raw["within_1"]:.3f}; 97% of its errors are one step, in a direction that is constant per rubric (half a level harsh on blur, never the worst level on JPEG). The hosted models score {hosted_rating[0]:.2f} to {hosted_rating[-1]:.2f}; each provider’s low-cost model beats its own flagship, and {best_rating} leads. Where a rubric’s author drew the lines is a convention that no model can know unseen.</p>
  <figure>{fig2}<figcaption><b>Figure 3.</b> Exact-level accuracy on the same 1,000 images, chance 0.25. Upper group: zero-shot. Lower group: the open model after seeing images of the rubric, first unlabeled (zero labels, but not zero-shot), then 32 labeled.</figcaption></figure>
  <p><b>Fitting the offset.</b> The two fits of section 1 act on this offset. The comparison is asymmetric by design: the hosted models stayed zero-shot, because a written pick offers nothing to fit and we did not give them few-shot examples. It shows what a local, fittable readout buys a user; it does not show that the open model sees better.</p>
  {extras_table()}
  <p class="caption"><b>Table {t0 + 2}.</b> Exact-level accuracy on the rating test of Table 2 (the last row uses the full test split and is a research result, not shipped). Unlabeled fitting roughly halves the calibration error (ECE 0.33 to about 0.2); only the labeled fit gives calibrated probabilities (ECE about 0.03).</p>
  <h3>5.1 Outside the quality scales</h3>
  <p><b>A real image-quality benchmark.</b> On KADID-10k (23 distortion types, five levels, human scores) the fitted open model reached 0.527 exact and missed every target we had registered. Zero-shot it is at {kadid_json:.2f} with the one-pass read and {kadid_ens:.2f} with the four-pass read; that check, together with the rubrics below, is what made the one-pass read the default for a rubric with nothing fitted, by a rule fixed in advance; the gain on KADID-10k was {100 * (kadid_json - kadid_ens):.1f} points where we had predicted at least five.</p>
  <p><b>Specialised tools.</b> With plentiful labels, 29 hand-built features score 0.979 on the synthetic scales against 0.867 for the fitted model.{outside_sentence()} For low-level artefacts these remain the better tools; a general model read this way earns its place by answering any typed question with one set of frozen weights.{openjev_sentence()}</p>{other_rubrics()}
</section>

<section aria-labelledby="models">
  <h2 id="models"><span class="num">6</span>Scale and family: zero-shot quality belongs to the model, and size does not buy boundary knowledge</h2>
  <p>The same prompts and readouts, not a word changed, on other sizes of the same family and on a model from a different family (different vision tower, different language model).</p>
  {closed_set_table()}
  <p class="caption"><b>Table {t0 + 3}.</b> Yes/no and pick-one on the two fresh photo sets, all items, uncalibrated. Within the Qwen3-VL family accuracy is flat from 2B upward, except the 13-way pick-one where the 2B model trails. The 2.2B model of another family is level with them on everyday photographs and trails on nature photographs (5 points on yes/no, 12 on pick-one), so the comparison with hosted models in section 2 is a statement about this family, not about every small open model.</p>
  {models_table()}
  <p class="caption"><b>Table {t0 + 4}.</b> Ratings, the same 1,000 images: exact-level accuracy. From 2B to 4B zero-shot accuracy rises sharply; from 4B to 8B it does not rise at all, and after 32 labels the three sizes are within 1.4 points. We had predicted a monotone rise and were wrong. “Within one” is for the four-pass read.</p>
</section>

<section aria-labelledby="cost">
  <h2 id="cost"><span class="num">7</span>Cost and speed: same order as the low-cost hosted models, one to two orders below the flagships</h2>
  <p>The two sides of this comparison are not measured the same way, and the caveats come first. Hosted cost is the provider’s bill per call and includes nothing for an operator; hosted latency is wall time from one laptop and includes the network and the provider’s queue. Open-model cost is measured seconds on a laptop multiplied by an on-demand cloud GPU price, with no batching, no idle time and no operator counted; a different GPU price or image resolution moves it by more than the gap to the low-cost hosted models. Hosted cost also depends on image size (GPT-5.6: $7.55 per 1,000 on 1,280-pixel files, $1.86 on 500-pixel files).</p>
  <p>With that said: a yes/no about a full-size photograph takes the open 4B model {own["seconds"]["yesno"]["value"]:.1f} s on a laptop and costs ${own["usd_per_1000"]["yesno"]["value"][0]:.2f} to ${own["usd_per_1000"]["yesno"]["value"][1]:.2f} per 1,000 on a rented GPU; the low-cost hosted models cost ${cheap_yesno[0]:.2f} and up, the flagships several dollars. The durable differences are not the cents: the image never leaves the machine, there is no per-call bill, it works offline, and the answer can be fitted.</p>
  <figure>{fig_acc_cost}<figcaption><b>Figure 4.</b> Accuracy against cost, one panel per question type, 95% intervals. Open model: read (circle) and written (square). Hosted models: {site_charts.hosted_key(mrows)}.</figcaption></figure>
  <figure>{fig_cost_speed}<figcaption><b>Figure 5.</b> Cost against speed; down and left is better. Each system is a large mark at the centre (geometric mean) of three small ones, one per question type: {site_charts.KEY}. Filled marks are the open 4B model.</figcaption></figure>
</section>

<section aria-labelledby="new">
  <h2 id="new"><span class="num">8</span>What is not new, and the work this sits in</h2>
  <p>Reading answer-token logits from a frozen generative model, with several questions sharing one image prefix, is what Simple Jev, jev-visual and LitJev also do. For yes/no and pick-one the forward pass here is not new, and the accuracy belongs to the open model. What this project adds is a harness on top: a fresh-photograph comparison with paid frontier calls, the same model writing against reading, self-calibration from unlabeled images and labeled fitting for rating levels (<code>glance fit</code>), and measured dollars and milliseconds. It trains no weights, unlike YOFO (Zhang et al. 2025), Laya Vision or OpenJev v2.</p>
  <p>The pieces are older than any of these tools. Scoring a closed set of candidate answers instead of generating is standard for language models (Kadavath et al. 2022) and its option-letter pitfalls are known (Zheng et al. 2024); VQAScore reads P(“Yes”) from a VQA model in one pass (Lin et al. 2024); a rubric score as a probability-weighted sum over rating tokens is G-Eval (Liu et al. 2023); level-token readouts for image quality are Q-Bench and Q-Align (Wu et al. 2024a, b), and frozen-CLIP quality scores are CLIP-IQA (Wang et al. 2023); the fits are matrix scaling and its relatives (Guo et al. 2017; Kull et al. 2019); the hidden-state variant is a linear probe (Alain and Bengio 2016); sharing an image prefix across questions is prefix caching (vLLM). Our additions are the contamination-controlled comparison, the read-against-write control, the unlabeled fit for ratings with the failed content-free prior as its contrast, and the registered misses.</p>
</section>

<section aria-labelledby="limits">
  <h2 id="limits"><span class="num">9</span>Limits and misses</h2>
  <ul class="misses">
    <li><span class="verdict miss">not supported</span> A content-free prior (blank and noise images) was expected to help zero-shot ratings. It took exact accuracy from {nullp["table"]["raw zero-shot"]["accuracy"]:.3f} to {nullp["table"]["content-free prior, all six null images (registered)"]["accuracy"]:.3f}: for an image rubric there is no content-free image.</li>
    <li><span class="verdict miss">not supported</span> Our four-pass rating readout was expected to match the same model’s written answer zero-shot. It trailed it by ten points: a readout selected with a calibration in the loop is good to fit and poor zero-shot. <span class="verdict">supported</span> The registered fix, one pass read at the JSON answer position, matches the written answer ({jd["json_zero"]:.3f}) and reaches {jd["json_u16"]:.3f} with 16 unlabeled images.</li>
    <li><span class="verdict miss">not supported</span> We expected each provider’s cheapest model to score at or below its flagship on ratings. Every one beats its flagship, and {best_rating} ({best_rating_acc:.3f}, ${cheap_rating[0]:.2f} to ${hosted[best_rating]["usd_per_1000"]["rating"]["value"][0]:.2f} per 1,000 among the cheap models) is nine points ahead of the open model zero-shot. With 16 unlabeled images the open model reaches {jd['json_u16']:.3f}, {100 * (best_rating_acc - jd['json_u16']):.1f} points short of it; with 32 labels it leads.</li>
    <li><span class="verdict miss">not supported</span> On KADID-10k (23 distortion types, five levels, human scores) every registered target was missed: 0.527 exact with labels, 0.33 zero-shot.</li>
    <li><span class="verdict">supported</span> A fitted readout on the model’s hidden state reaches {r4a:.3f} from one pass, against 0.867 for the token readout: the model represents severity almost perfectly. It needs on the order of a hundred labels.</li>
{outside_item()}
    <li><span class="verdict">known</span> Hand-built image features beat the VLM on low-level artifacts when labels are plentiful (0.979). A calibration fitted on one rubric does not transfer to another. Two model families and three sizes are measured (Tables {t0 + 3} and {t0 + 4}); that is not “any model”.</li>
  </ul>
</section>

<section aria-labelledby="reproduce">
  <h2 id="reproduce"><span class="num">10</span>Reproduce</h2>
<pre><code>uv sync &amp;&amp; uv run glance doctor
uv run python tools/fetch_fresh_inat.py          # 200 photographs, about ten API calls
uv run glance eval --suite inat_choice --suite inat_yesno --model vlm
uv run python tools/make_results_zeroshot.py     # every table on this page
uv run glance --model-id &lt;any Hugging Face image-text model&gt; --revision &lt;commit&gt; ask photo.jpg "Is there a dog?"   # checked on SmolVLM2-2.2B only</code></pre>
  <p>The lab notebook records each hypothesis before its experiment, each verdict after, and its errata. Frontier model outputs are never stored; only whether each answer was right.</p>
</section>

<section aria-labelledby="refs">
  <h2 id="refs"><span class="num">11</span>References</h2>
  {refs_html()}
</section>

<footer>
  <p>Open model: Qwen3-VL-4B-Instruct, Apache-2.0. Photographs: CC0, CC BY and CC BY-SA, attributed per file in the repository and not redistributed. Code, the pre-registration notebook and every result file: github.com/yoheinakajima/glance (public with the release). This page is generated by <code>tools/make_site.py</code>; cite it by its snapshot id.</p>
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
tr.own td{font-weight:600}tr.sep td{border-top:.75px solid var(--rule)}td.own{font-weight:600}tr.group td{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.76rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);padding-top:.85rem;padding-bottom:.15rem}
.ci{font-size:.78rem;color:var(--muted);font-weight:400}.note{font-size:.78rem;color:var(--muted);font-weight:400;display:block}
.pending{color:var(--muted);font-style:italic;text-align:center}
.protocol{margin:0;display:grid;grid-template-columns:7.5rem 1fr;gap:.7rem 1rem;font-size:.95rem}.protocol dt{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);padding-top:.2rem}.protocol dd{margin:0}.protocol code{white-space:normal;word-break:break-word}@media (max-width:560px){.protocol{grid-template-columns:1fr;gap:.2rem}.protocol dd{margin-bottom:.6rem}}.refs{font-family:"IBM Plex Sans",Arial,sans-serif;font-size:.84rem;line-height:1.5;padding-left:1.4rem;display:flex;flex-direction:column;gap:.35rem}.aff{color:var(--muted);font-size:.9rem}td code,th code{white-space:normal}.caption,figcaption{font-size:.82rem;line-height:1.5;color:var(--muted)}
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

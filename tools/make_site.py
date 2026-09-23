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
    rows1.append((f"Qwen3-VL-4B + Glance · {tag}", s[key]["accuracy"], s[key]["ci95"], True, suite == "fresh_choice"))
    rows1 += [(f"{name} · {tag}", s[m]["accuracy"], s[m]["ci95"], False, False) for m, name in FRONTIER.items() if m in s]
fig1 = dot_plot(rows1, 0.75, 1.0, "Accuracy on photos taken after every model's release")

# ---- Figure 2: ratings -------------------------------------------------------------------------------
written = sum(gen[k]["written"][0] for k in gen if k.startswith("ladder_")) / 5
loc = h2h["local"]
rows2 = [(FRONTIER.get(r["model"], r["model"]) + ", written pick", r["mean_accuracy"], None, False, False) for r in h2h["runs"].values()]
rows2 += [("Qwen3-VL-4B, written answer", written, None, True, False), ("Qwen3-VL-4B + Glance, one-pass read", jd["json_zero"], None, True, False),
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
        rows = [f'<tr class="own{" sep" if sep else ""}"><td>Qwen3-VL-4B + Glance <span class="note">{title}, {note}</span></td><td class="n">{ci(y["vlm:statement"])}</td><td class="n">{ci(c["vlm:independent"])}</td></tr>']
        if wy and wc:
            rows.append(f'<tr class="own"><td>Qwen3-VL-4B, writing JSON <span class="note">same items, no Glance</span></td><td class="n">{t3(wy["written"])}</td><td class="n">{t3(wc["written"])}</td></tr>')
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
            out.append(f"<tr{sep}><td>{html.escape(SHORT.get(name, name))}</td>" + "".join(f'<td class="n">{fmt(field, sysrow[field][t])}</td>' for t in tests) + "</tr>")
        out.append("</tbody></table></div>")
    return "".join(out)


def extras_table():
    out = ['<div class="table-scroll"><table><thead><tr><th>Give the read row</th><th class="n">rating accuracy</th></tr></thead><tbody>']
    out += [f'<tr><td>{html.escape(e["what"]).replace("`glance fit --unlabeled`", "<code>glance fit --unlabeled</code>").replace("`glance fit`", "<code>glance fit</code>")}'
            f' <span class="note">{html.escape(e["note"])}</span></td><td class="n">{e["rating_accuracy"]:.3f}</td></tr>' for e in matrix["only_the_read_row_can_add"]]
    return "".join(out) + "</tbody></table></div>"


def closed_set_table():
    other = load("results/lab/other_models.json") or {}
    order = [("Qwen3-VL-2B", "Qwen3-VL-2B"), ("Qwen3-VL-4B", "Qwen3-VL-4B"), ("Qwen3-VL-8B", "Qwen3-VL-8B"), ("SmolVLM2-2.2B, another family", "SmolVLM2-2.2B (another family)")]
    out = ['<div class="table-scroll"><table><thead><tr><th>Open model, same questions</th><th class="n">yes/no · Commons</th><th class="n">yes/no · iNaturalist</th><th class="n">yes/no · insect orders</th><th class="n">pick-one · Commons</th><th class="n">pick-one · iNaturalist</th><th class="n">pick-one · insect orders</th></tr></thead><tbody>']
    for label, key in order:
        e = other.get(key)
        if not e:
            out.append(f'<tr><td>{label}</td><td class="n pending" colspan="4">not in this snapshot</td></tr>')
            continue
        out.append(f'<tr{" class=own" if key == "Qwen3-VL-4B" else ""}><td>{label}</td>' + "".join(f'<td class="n">{e[k]["accuracy"]:.3f}</td>' if k in e else '<td class="n">–</td>'
                                                                                                    for k in ("fresh_yesno", "inat_yesno", "inat_orders_yesno", "fresh_choice", "inat_choice", "inat_orders_choice")) + "</tr>")
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
    ("Agrahri (TrueStandard), 2026", "Is Jev really 193x faster? We measured 1.7x and 100x: independent timings of hosted Jev against fast classifiers and against a thinking-model workflow", "https://truestandard.ai/blog/is-jev-really-193x-faster"),
    ("dorarep, 2026", "Jev against small LLMs (in Japanese): latency as the number of questions per request grows from 1 to 100", "https://zenn.dev/dorarep/articles/8f1efbf10e3e8c"),
    ("Goedecke, 2026", "Jev means structured output is interesting again: one constrained token against written structured output on a small open model", "https://seangoedecke.com/jev-means-structured-output-is-interesting-again/"),
    ("TypeSafe, 2026", "Introducing System One models and Jev (the vendor’s launch post; its figures are the vendor’s own)", "https://typesafe.ai/blog/introducing-system-one-models-and-jev"),
    ("Simple Jev", "Any Hugging Face model as a classifier, by reading next-token logits (training-free)", "https://github.com/featherless-ai/simple-jev"),
    ("jev-visual", "The same readout on a Qwen vision-language model, with a shared multimodal prefix (training-free)", "https://github.com/hr98w/jev-visual"),
    ("LitJev", "A Jev-like typed-decision endpoint on an off-the-shelf Qwen model (training-free)", "https://github.com/zhengxuyu/litjev"),
]


def refs_html():
    people = sorted((r for r in REFS if r[0][0].isalpha() and "," in r[0]), key=lambda r: r[0].lower())  # author-year entries, alphabetical
    rest = [r for r in REFS if r not in people]  # projects and tools, in the order given
    return "<ol class=\"refs\">" + "".join(f'<li>{a}. {t}. <a href="{u}">{u.split("//")[1][:48]}</a></li>' for a, t, u in people + rest) + "</ol>"


photo_timed = "on these photographs" in matrix["systems"]["Qwen3-VL-4B, read (Glance)"]["seconds"]["yesno"]["how"]
size_note = "" if photo_timed else " The open model’s yes/no and pick-one timings here were taken on smaller test images (and a three-option pick-one) than the photographs the hosted models saw; a timing on the same photographs is running and will replace them."
def finer():
    orders, probes = load("results/lab/fresh_inat_orders.json"), load("results/lab/probes.json")
    out = []
    if orders:
        y, c = orders["suites"]["inat_orders_yesno"], orders["suites"]["inat_orders_choice"]
        rows = [("Qwen3-VL-4B + Glance", y.get("vlm:statement"), c.get("vlm:independent")), ("SigLIP2 (open dual encoder)", None, c.get("siglip:independent"))]
        rows += [(n, y.get(k), c.get(k)) for k, n in list(FRONTIER.items()) + [("anthropic/claude-haiku-4-5", "Claude Haiku 4.5"), ("openai/gpt-5.6-luna", "GPT-5.6 Luna"), ("openrouter/google/gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite")] if k in y or k in c]
        hosted_c = sorted((c[k]["accuracy"], n) for k, n in [(k, n) for _, _, _ in [(0, 0, 0)] for k, n in list(FRONTIER.items()) + [("anthropic/claude-haiku-4-5", "Claude Haiku 4.5"), ("openai/gpt-5.6-luna", "GPT-5.6 Luna"), ("openrouter/google/gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite")]] if k in c)
        spread = (f" This test does separate systems: on pick-one the hosted models range from {hosted_c[0][0]:.3f} ({hosted_c[0][1]}) to {hosted_c[-1][0]:.3f} ({hosted_c[-1][1]}), and the open 4B model, at "
                  f"{c['vlm:independent']['accuracy']:.3f}, is within {abs(hosted_c[-1][0] - c['vlm:independent']['accuracy']) * 100:.0f} point of the best.") if hosted_c else ""
        out.append("<p>Insect orders on iNaturalist (beetle, true bug, fly, …; 210 photographs; every “no” question names a look-alike order) ask for finer distinctions than the ten-group test." + spread + " The open dual encoder falls to 0.71 here.</p>")
        out.append('<div class="table-scroll"><table><thead><tr><th>Insect orders, zero-shot</th><th class="n">yes/no</th><th class="n">pick one of 7</th></tr></thead><tbody>'
                   + "".join(f'<tr{" class=own" if n.startswith("Qwen") else ""}><td>{n}</td><td class="n">{ci(a) if a else "–"}</td><td class="n">{ci(b_) if b_ else "–"}</td></tr>' for n, a, b_ in rows) + "</tbody></table></div>")
        out.append('<p class="caption"><b>Table 5.</b> Seven insect orders, zero-shot, 95% bootstrap intervals. The open model answered all items (420 yes/no questions, 210 photographs); hosted models, where present, the test half (210 and 105).</p>')
    if not probes and not load("results/lab/ui_screens.json"):
        out.append("<p>Tests that need more than coarse recognition (counting, spatial relations, relative size, reading, and which element of a screen to click) are registered and built from images drawn by program, with labels exact by construction; their results are not in this snapshot.</p>")
    return "\n  ".join(out)


def lower_keep(text):
    """Lower-case a test name for use inside a sentence, keeping proper nouns."""
    return text.lower().replace("commons", "Commons").replace("inaturalist", "iNaturalist")


_om = load("results/lab/other_models.json") or {}
_so, _qo = _om.get("SmolVLM2-2.2B (another family)", {}), _om.get("Qwen3-VL-4B", {})
smol_orders = (f" and falls well behind on the insect orders ({_so['inat_orders_yesno']['accuracy']:.2f} and {_so['inat_orders_choice']['accuracy']:.2f} against {_qo['inat_orders_yesno']['accuracy']:.2f} and {_qo['inat_orders_choice']['accuracy']:.2f})"
               if "inat_orders_choice" in _so and "inat_orders_choice" in _qo else "")
gap_2b = ((POOLED_EARLY := load("results/lab/pooled_photos.json"))["kinds"]["yesno"]["systems"].get("Qwen3-VL-2B, read") or {}).get("open_4b_minus_this_points", [float("nan")])[0]
_open_read = [n for n in matrix["systems"] if n.startswith("Qwen") and not n.endswith("written")]
_sizes = [n.split("-")[2].split(",")[0] for n in _open_read]  # "2B", "4B", "8B"
_pairs = [z for z in _sizes if f"Qwen3-VL-{z}, written" in matrix["systems"]]
open_rows_note = ("Qwen3-VL at " + ", ".join(_sizes[:-1]) + " and " + _sizes[-1] + " read with Glance; the rows marked “written” are the same model generating a JSON answer without it ("
                  + " and ".join(_pairs) + ")") if len(_sizes) > 1 else "Qwen3-VL-4B read with Glance, and the same model writing JSON without it"
_w2 = matrix["systems"].get("Qwen3-VL-2B, written")
strict_note = (f" A written answer that does not parse or is not an allowed value counts as wrong, for every system; that rule costs the 2B model most of its written score "
               f"({_w2['invalid_share']['yesno']:.0%} of its yes/no answers and {_w2['invalid_share']['rating']:.0%} of its ratings are malformed, typically a missing brace). Scored leniently, "
               f"taking the first allowed answer in its text, it reaches {_w2['lenient_accuracy']['yesno']:.2f}, {_w2['lenient_accuracy']['choice']:.2f} and {_w2['lenient_accuracy']['rating']:.2f}: "
               "reading spares a small model the formatting; it does not make it smarter.") if _w2 else ""
dark_note = ("Black bars are open models on a laptop read with Glance; the lighter bar above each is the same model writing JSON without it (" + open_rows_note.split("(")[-1].rstrip(")") + "; the 8B model’s written row was not collected)."
             + strict_note) if len(_sizes) > 1 else "Dark bars are the open 4B model."
basis_note = ("yes/no and pick-one accuracy pooled over the three fresh photo sets, Table 3; seconds and dollars as measured on the Commons photographs, because hosted cost depends on image size"
              if matrix.get("accuracy_basis", "").startswith("three") else "the test half of the Commons set, the first of three photo sets; Table 3 pools all three")
POOLED = load("results/lab/pooled_photos.json")
SHORT = {"Qwen3-VL-4B, read": "Qwen3-VL-4B + Glance", "Qwen3-VL-4B, written": "Qwen3-VL-4B, writing JSON", "Qwen3-VL-2B, written": "Qwen3-VL-2B, writing JSON", "Qwen3-VL-2B, read": "Qwen3-VL-2B + Glance", "Qwen3-VL-8B, read": "Qwen3-VL-8B + Glance",
         "Qwen3-VL-4B, read (Glance)": "Qwen3-VL-4B + Glance", "Qwen3-VL-2B, read (Glance)": "Qwen3-VL-2B + Glance", "Qwen3-VL-8B, read (Glance)": "Qwen3-VL-8B + Glance"}  # display names


def pooled_groups(kind):
    """Hosted models by the reading of the paired difference (entries 54 and 62): behind / no difference detected / ahead. An interval that
    spans zero is not equivalence; the equivalence-style statement is `within_3_points` (the whole interval inside +-3)."""
    e = POOLED["kinds"][kind]["systems"]
    g = {"open model behind": [], "no difference detected": [], "open model ahead": []}
    for name, row in e.items():
        if "verdict" in row:
            g[row["verdict"]].append((name, row["open_minus_this_points"]))
    return g


def _names(items, with_gap=False):
    parts = [f"{n} ({abs(d[0]):.1f} points [{min(abs(d[1]), abs(d[2])):.1f}, {max(abs(d[1]), abs(d[2])):.1f}])" if with_gap else n for n, d in items]
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if parts else "none"


def pooled_sentence(kind, label):
    g, e = pooled_groups(kind), POOLED["kinds"][kind]["systems"]
    own = e["Qwen3-VL-4B, read"]["pooled"]

    def join(parts):
        return ", ".join(parts[:-1]) + (" or " if len(parts) > 1 else "") + parts[-1] if parts else ""

    out = f"On {label} ({POOLED['kinds'][kind]['n_total']} items) the open model scores {own[0]:.3f} [{own[1]:.3f}, {own[2]:.3f}]."
    same = g["no difference detected"]
    if same:
        within = [n for n, _ in same if e[n].get("within_3_points")]
        out += " We detect no difference from " + join([f"{n} ({d[0]:+.1f} points [{d[1]:+.1f}, {d[2]:+.1f}])" for n, d in same])
        out += ("; " + ("all of these intervals lie" if len(within) == len(same) else f"the intervals for {_names([(n, None) for n in within])} lie") + " within three points, the closest this sample comes to showing equivalence." if within else ".")
    rest = []
    if g["open model behind"]:
        rest.append(f"behind {_names(g['open model behind'], True)}")
    if g["open model ahead"]:
        rest.append(f"ahead of {_names(g['open model ahead'], True)}")
    if rest:
        out += " It is " + " and ".join(rest) + "."
    return out

def orders_clause():
    """One clause for the abstract on the hardest photo set: look-alike insect orders, where the hosted models separate."""
    o = load("results/lab/fresh_inat_orders.json")
    if not o:
        return ""
    c = o["suites"]["inat_orders_choice"]
    hosted = [c[k]["accuracy"] for k in ALL_HOSTED if k in c]
    if not hosted:
        return ""
    return (f" On the hardest of the three sets, look-alike insect orders, the hosted models spread over {100 * (max(hosted) - min(hosted)):.0f} points on pick-one "
            f"({min(hosted):.2f} to {max(hosted):.2f}) and the open model scores {c['vlm:independent']['accuracy']:.2f}.")


def abstract_photos():
    """The abstract's sentence on photographs, from the pooled analysis (entry 55)."""
    y, c = POOLED["kinds"]["yesno"], POOLED["kinds"]["choice"]
    own_y, own_c = y["systems"]["Qwen3-VL-4B, read"]["pooled"][0], c["systems"]["Qwen3-VL-4B, read"]["pooled"][0]
    hosted = lambda e: {n: r["pooled"][0] for n, r in e["systems"].items() if "verdict" in r}  # noqa: E731
    best_y, best_c = max(hosted(y).items(), key=lambda kv: kv[1]), max(hosted(c).items(), key=lambda kv: kv[1])
    gy, gc = pooled_groups("yesno"), pooled_groups("choice")
    level_both = [n for n, _ in gy["no difference detected"] if n in {m for m, _ in gc["no difference detected"]}]
    ahead_both = [n for n, _ in gy["open model ahead"] if n in {m for m, _ in gc["open model ahead"]}]
    return (f"On photographs taken after every model’s release, labelled by people outside this project (three sets, {y['n_total']} yes/no questions and {c['n_total']} pick-one photographs put to every system), "
            f"the open model is level with the best hosted models on pick-one ({own_c:.3f} against {best_c[1]:.3f} for {best_c[0]}) and about two points behind the best on yes/no ({own_y:.3f} against {best_y[1]:.3f} for {best_y[0]}; "
            f"the paired difference excludes zero for both Gemini models). We detect no difference from {_names([(n, None) for n in level_both])} on either, and it is ahead of {_names([(n, None) for n in ahead_both])} on both.{orders_clause()}")


def pooled_table():
    def row(name, e):
        def cell(kind):
            r = e[kind].get(name)
            if not r:
                return '<td class="n">–</td><td class="n">–</td>'
            d = r.get("open_minus_this_points")
            return (f'<td class="n">{r["pooled"][0]:.3f} <span class="ci">[{r["pooled"][1]:.3f}, {r["pooled"][2]:.3f}]</span></td>'
                    + (f'<td class="n">{d[0]:+.1f} <span class="ci">[{d[1]:+.1f}, {d[2]:+.1f}]</span></td>' if d else '<td class="n">–</td>'))
        return f'<tr{" class=own" if name.startswith("Qwen") else ""}><td>{SHORT.get(name, name)}</td>{cell("yesno")}{cell("choice")}</tr>'
    e = {k: POOLED["kinds"][k]["systems"] for k in ("yesno", "choice")}
    names = list(dict.fromkeys(list(e["yesno"]) + list(e["choice"])))
    n = {k: POOLED["kinds"][k]["n_total"] for k in e}
    return ('<div class="table-scroll"><table><thead><tr><th>Three photo sets pooled, zero-shot</th>'
            f'<th class="n">yes/no, n={n["yesno"]}</th><th class="n">open minus this, points</th><th class="n">pick-one, n={n["choice"]}</th><th class="n">open minus this, points</th></tr></thead><tbody>'
            + "".join(row(name, e) for name in names) + "</tbody></table></div>")


def speed_context():
    """Independent timings of hosted Jev, a text-only product of the same family, beside our own ratios (notebook entry 56). The
    outside numbers were read at their sources on 2026-09-21; ours are computed from the matrix. Nothing here is our measurement of Jev."""
    sec = {n: v["seconds"]["yesno"]["value"] for n, v in matrix["systems"].items()}
    own, wrote = sec["Qwen3-VL-4B, read (Glance)"], sec["Qwen3-VL-4B, written"]
    flash, haiku = sec["Gemini 3.1 Flash-Lite"], sec["Claude Haiku 4.5"]
    versus_haiku = f"slower than Claude Haiku 4.5 ({own:.2f} s against {haiku:.2f} s)" if haiku < own else f"{haiku / own:.1f} times against Claude Haiku 4.5"
    return ("<p><b>Speed, in context.</b> Hosted Jev (TypeSafe 2026) is the trained product of this family; it takes text, not images. Its launch material quotes 40 to 200 times faster than frontier language models. "
            "Independent timings show what that figure is made of. TrueStandard (Agrahri 2026) timed one three-way classification of a support ticket at 477 ms of server time against 790 ms for Gemini 3.1 Flash Lite and 928 ms "
            "for Claude Haiku 4.5, 1.7 and 1.9 times, and reached 100 times only when one call replaced six sequential calls to a thinking model: “the multiple is a property of the comparison, not of the model”. "
            "Goedecke (2026) measured 2 to 3 times from having a small open model emit one constrained token instead of written structured output, and dorarep (2026) found that going from one question to a hundred per request "
            "cost Jev 1.5 times the latency where generating models paid 6 to 28 times. "
            f"Our ratios sit in the same modest band, with the image as a fixed cost that text systems do not pay: {wrote / own:.1f} times against the same model writing JSON on a full-size photograph, 2.4 to 6.1 times on small images "
            f"as questions per image grow, {flash / own:.1f} times against Gemini 3.1 Flash-Lite, and {versus_haiku}. We did not run a sequential thinking-model workflow and claim nothing about one. "
            "None of the outside figures is our measurement, and raw milliseconds do not transfer between a hosted text model and a 4B vision model on a laptop.</p>")


def pip_live():
    """Is `glance-vlm` on PyPI? Checked when the page is built (3 s), so the install line is never ahead of the truth; without a
    network the last recorded answer is used."""
    import urllib.error
    import urllib.request
    state = ROOT / "results/lab/pypi_status.json"
    last = (json.loads(state.read_text()) if state.exists() else {}).get("glance-vlm on PyPI", False)
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/glance-vlm/json", timeout=3) as r:
            live = r.status == 200
    except urllib.error.HTTPError as e:
        live = False if e.code == 404 else last
    except Exception:
        return last
    if live != last or not state.exists():
        state.write_text(json.dumps({"glance-vlm on PyPI": live, "changed": datetime.date.today().isoformat()}) + "\n")
    return live


def regimes():
    """The whole result in three rows (a reviewer's suggestion, adopted 2026-09-21): what kind of question, what was measured, what to do.
    Every number comes from a result file."""
    y, c = POOLED["kinds"]["yesno"]["systems"], POOLED["kinds"]["choice"]["systems"]
    best = lambda e: max(r["pooled"][0] for r in e.values() if "verdict" in r)  # noqa: E731
    own_y, own_c = y["Qwen3-VL-4B, read"]["pooled"][0], c["Qwen3-VL-4B, read"]["pooled"][0]
    wr_y, wr_c = y["Qwen3-VL-4B, written"]["pooled"][0], c["Qwen3-VL-4B, written"]["pooled"][0]
    pr = load("results/lab/probes.json") or {}
    acc = {k: v["open 4B model, read"]["all_items"]["accuracy"] for k, v in pr.get("suites", {}).items()}
    hosted_perfect = pr.get("paired", {}).get("probe_largest", {}).get("best_hosted_accuracy")
    ctl = (load("results/lab/probe_controls.json") or {}).get("sets", {})
    rows = [("Recognition", "yes/no and pick-one about what is in a photograph",
             f"{own_y:.3f} and {own_c:.3f} on fresh photographs, against {best(y):.3f} and {best(c):.3f} for the best hosted model; the same model writing its answer scores {wr_y:.3f} and {wr_c:.3f}",
             "Read it. One forward pass of a small open model, nothing generated."),
            ("Ratings on a rubric", "a level on an image-quality scale",
             f"zero-shot the order is right (within one level on {raw['within_1']:.2f} of images) and the exact level often wrong ({jd['json_zero']:.3f}); {jd['json_u16']:.3f} after the model has seen 16 unlabeled images of the rubric",
             "Show it a few images of the rubric first. Outside image quality this does not work (section 5.1)."),
            ("Geometry", "relative size, line direction, tilt",
             (f"{acc.get('probe_largest', float('nan')):.2f} on the largest of four shapes and {acc.get('probe_stripes', float('nan')):.2f} on stripe direction, where the best hosted models score {hosted_perfect:.2f}"
              if hosted_perfect else "weak on relative size and line direction")
             + (f"; the direction is in its hidden state ({ctl['probe_stripes']['hidden_state_probe_cv'][0]:.2f} by a linear probe) and the relative size is not" if ctl else ""),
             "Do not rely on this readout. Use a larger hosted model.")]
    body = "".join(f'<tr><td><b>{a}</b><span class="note">{b}</span></td><td>{c_}</td><td>{d}</td></tr>' for a, b, c_, d in rows)
    return ('<section aria-labelledby="regimes" class="regimes"><h2 id="regimes" class="plain">The result in three regimes</h2>'
            '<div class="table-scroll"><table><thead><tr><th>What you ask</th><th>What we measured, Qwen3-VL-4B + Glance</th><th>What to do</th></tr></thead><tbody>'
            + body + "</tbody></table></div></section>\n")


def use_it():
    """A panel that belongs to the WEBSITE, not to the paper: what the repository lets a reader do, with the commands. The paper
    version drops this block (it is the only element with class `site-only`)."""
    repo = "https://github.com/yoheinakajima/glance"
    demo, replicate, space = "https://glancevlm.replit.app", "https://replicate.com/untapped/glance-qwen3-vl-4b", "https://huggingface.co/spaces/yoheinakajima/glance-qwen3-vl-4b-demo"
    live = pip_live()
    install = ("# install: Python 3.11; about 9 GB of open weights download on first use\npip install glance-vlm" if live else
               f"# install: Python 3.11 and uv; about 9 GB of open weights download on first use\ngit clone {repo} && cd glance && uv sync")
    run = "" if live else "uv run "
    return f"""<aside class="useit site-only" aria-labelledby="useit">
  <div class="useit-top"><h2 id="useit" class="plain">Use it</h2><a class="gh" href="{repo}">github.com/yoheinakajima/glance&nbsp;→</a></div>
  <p>Glance is also a tool you can run: the readout and the fits measured on this page, packaged. One open model on your own machine, typed answers with probabilities, no image leaves it, no per-call bill. Apache-2.0.</p>
  <p class="tryit"><b>Speed lab</b>: twenty-one measured latency experiments, including an experimental 8-bit MLX scorer that lowered fresh-frame p50 by 25–28% on the fixed suite. <a href="speed/">Read the report and run it locally&nbsp;→</a><span class="note">The speed result is on one Apple M5 machine and a small quality guardrail; the report keeps the rejected experiments and exact scope beside the headline.</span></p>
  <p class="tryit"><b>Try it in your browser</b>, no install: <a href="{demo}">live demo&nbsp;→</a> <a href="{replicate}">Replicate&nbsp;→</a> <a href="{space}">Hugging Face Space&nbsp;→</a><span class="note">The same readout on Qwen3-VL-4B, hosted: yes/no and pick-one. These run in the cloud, so your image is sent to them and the first answer can take a minute while a machine starts; the tool below runs on your own machine.</span></p>
<pre><code>{install}

# yes/no: a probability
{run}glance ask photo.jpg "Is there a dog?"
# pick one: a distribution over your options
{run}glance ask photo.jpg "What is it?" --options dog cat car
# a rating, with its distribution
{run}glance ask photo.jpg "How blurry is it?" --levels Sharp Soft Blurry
# fit your own rubric from a few dozen images
{run}glance fit --rubric rubric.json --data labels/</code></pre>
  <ul>
    <li><b>Ask</b> from the command line, from Python (<code>from glance import Glance</code>) or over a local HTTP server (<code>glance serve</code>); several questions about one image share the cost of reading it.</li>
    <li><b>Fit your own rubric</b> from 16 unlabeled or about 32 labeled images. A fit is a few hundred numbers; the model stays frozen. Section 5 says where this works and where it does not.</li>
    <li><b>Swap the model</b>: the adapter is model-generic (<code>--model-id</code>, any Hugging Face image-text model with a chat template); Qwen3-VL and SmolVLM2 are the families checked so far, and accuracy depends on the model (section 6).</li>
    <li><b>Check us</b>: every table and figure here regenerates from result files in the repository, which also holds the notebook of registered experiments and their misses.</li>
    <li><b>Hand it to an agent</b>: <a href="{repo}/blob/main/AGENTS.md">AGENTS.md</a> is a one-page operating guide, and <a href="llms.txt">llms.txt</a> summarises the project and its limits.</li>
  </ul>
  <p class="useit-foot">This panel belongs to the website. The paper begins below.</p>
</aside>
"""


def size_speed():
    """Section 6: seconds and dollars by model size, writing against reading (results/lab/size_speed.json; all idle-GPU timings)."""
    d = load("results/lab/size_speed.json")
    if not d or len(d["sizes"]) < 2:
        return ""
    sizes = list(d["sizes"])
    tasks = list(d["sizes"][sizes[0]])
    rows = []
    for task in tasks:
        for i, size in enumerate(sizes):
            c = d["sizes"][size][task]
            rows.append(f'<tr{" class=sep" if i == 0 and task != tasks[0] else ""}><td>{task if i == 0 else ""}</td><td class="n">{size}</td><td class="n">{c["write_s"]:.2f} s</td><td class="n">{c["read_s"]:.2f} s</td>'
                        f'<td class="n">{c["write_over_read"]:.1f}×</td><td class="n">${c["read_usd_per_1000"][0]:.2f}–{c["read_usd_per_1000"][1]:.2f}</td></tr>')
    yn, rt = tasks[0], tasks[-1]
    ratio = lambda task: ", ".join(f"{d['sizes'][z][task]['write_over_read']:.1f}" for z in sizes)  # noqa: E731
    read_s = ", ".join(f"{d['sizes'][z][yn]['read_s']:.2f}" for z in sizes)
    text = (f"<p><b>Speed and cost by size.</b> Timed the same way at every size (one laptop, GPU otherwise idle), a read yes/no about a full-size photograph takes {read_s} s at {', '.join(sizes[:-1])} and {sizes[-1]}: "
            f"each doubling of the model roughly doubles the time and the cost, and above 4B it buys no accuracy (Tables {{t0 + 3}} and {{t0 + 4}}). Does reading save more as the model grows? It depends on what dominates. "
            f"On a full-size photograph the image has to be encoded either way, so writing costs {ratio(yn)} times a read at the three sizes: a steady saving of about a third, not a growing one. "
            f"On small images, where the answer tokens are most of the work, the saving grows with size ({ratio(rt)} times for a rating), because every generated token costs a full pass of a larger model while a read stays one pass.</p>")
    table = ('<div class="table-scroll"><table><thead><tr><th>Seconds per answer, idle GPU</th><th class="n">size</th><th class="n">written</th><th class="n">read</th><th class="n">written / read</th><th class="n">read, $ per 1,000</th></tr></thead><tbody>'
             + "".join(rows) + "</tbody></table></div>")
    cap = (f'<p class="caption"><b>Table {{t0 + 5}}.</b> Qwen3-VL writing a JSON answer against the same model read with Glance. Yes/no and pick-one on the full-size Commons photographs, ratings on 448-pixel lab images (one-pass read). '
           f'Cost is the measured seconds at a rented-GPU price of ${d["gpu_usd_per_hour"][0]:.2f} to ${d["gpu_usd_per_hour"][1]:.2f} per hour, the same for every size; a larger model may need a dearer GPU, which is not modelled. '
           'In a 40-image benchmark the 2B model’s written answers were short and often invalid, which flatters its writing time.</p>')
    return (text + table + cap).replace("{t0 + 3}", str(t0 + 3)).replace("{t0 + 4}", str(t0 + 4)).replace("{t0 + 5}", str(t0 + 5))


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
    return ('<div class="table-scroll"><table><thead><tr><th>Question, zero-shot</th><th class="n">options</th><th class="n">Qwen3-VL-4B + Glance</th>' + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
            + f'<p class="caption"><b>Table 6.</b> Exact-answer accuracy on images whose labels are exact by construction (drawn or rendered by program; no photographs, no people). {note}</p>')


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
    return "<p>" + out + "</p>" + probes_hosted_text(d) + ui_text()


def controls_text():
    """E26 (entries 61, 61b): token read, written answer and a linear probe on the hidden state, same images; from results/lab/probe_controls.json."""
    c = (load("results/lab/probe_controls.json") or {}).get("sets")
    if not c:
        return "Whether the open model’s failures sit in this readout or in the model is the subject of a registered control that is running."
    s, g = c["probe_stripes"], c["probe_largest"]
    return (f"Are the open model’s two large failures in the readout or in the model? A registered control says they differ in kind. On stripe direction the token read scores {s['token_read'][0]:.2f} and the same model "
            f"writing its answer {s['written_strict'][0]:.2f}, yet a linear probe on its final hidden state, cross-validated on the same 150 images, reaches {s['hidden_state_probe_cv'][0]:.2f}: the model separates the four directions, "
            f"the two diagonals included, and no answer token gets it out. On the largest shape all three routes fail alike ({g['token_read'][0]:.2f}, {g['written_strict'][0]:.2f}, {g['hidden_state_probe_cv'][0]:.2f}): "
            "at this size and image resolution the model does not make that judgement. The probe sees labels, so it shows that the information is present, not that it can be read without examples.")


def probes_hosted_text(d):
    """The hosted half of the drawn probes (entry 50d), by the rule of entry 54: per set, same items, best hosted minus open, no composite."""
    pr = d.get("paired") or {}
    if not pr:
        return ""
    same = lambda suite, name: d["suites"][suite][name]["same_items"]["accuracy"]  # noqa: E731
    hosted = [k for k in ALL_HOSTED if all(k in d["suites"][s] for s in pr)]
    perfect = [ALL_HOSTED[k] for k in hosted if all(same(s, k) >= 0.9995 for s in pr)]
    gap = lambda s: f"{pr[s]['best_minus_open_points'][0]:.0f} points [{pr[s]['best_minus_open_points'][1]:.0f}, {pr[s]['best_minus_open_points'][2]:.0f}]"  # noqa: E731
    b = d.get("breakdowns", {}).get("probe_stripes", {}).get("systems", {})
    diag = {ALL_HOSTED[k]: (v["diagonal_rising"]["accuracy"] + v["diagonal_falling"]["accuracy"]) / 2 for k, v in b.items() if k in ALL_HOSTED}
    confused = [n for n, v in diag.items() if v < 0.8]
    out = (f"These are not limits of vision-language models. On the same items {_names([(n, None) for n in perfect])} answer every question of all six sets correctly. "
           f"Paired on the items every system answered, the best hosted model is ahead of the open model by {gap('probe_largest')} on the largest shape and {gap('probe_stripes')} on stripe direction, "
           f"and by {gap('probe_count')} on counting and {gap('probe_spatial')} on position; on the look-alike words every system is perfect. "
           + (f"The confusion of the two diagonals is shared by {_names([(n, None) for n in confused])} and by no other hosted model. " if confused else "")
           + "The low-cost hosted models fail where the open model fails: on the largest shape at the smallest ratio, and when counting seven or eight. "
           "We had predicted that most hosted models would share the diagonal confusion and that counting would be weak for every system; both predictions were wrong. "
           "One difference in the setup should be kept in mind: a hosted model writes its answer and may reason before it, while the open model gets one forward pass. "
           + controls_text())
    return "\n  <p>" + out + "</p>"


def ui_text():
    """Sentences for the synthetic interface screens (entries 49c, 49d): registered all-item numbers from results/lab/ui_screens.json, the
    test's two defects and the disabled-button diagnostic from results/lab/ui_defects.json."""
    d, x = load("results/lab/ui_screens.json"), load("results/lab/ui_defects.json")
    if not d:
        return ""
    acc = {k: v["open 4B model, read"]["all_items"] for k, v in d["suites"].items()}
    pr = d.get("paired") or {}
    out = (f"Five hundred synthetic interface screens (five kinds of page, invented content, rendered from generated HTML so every label is exact) ask what an agent would ask. Given a goal in words and six to eight numbered marks on the screen, "
           f"the open model names the mark to click on {acc['ui_click']['accuracy']:.3f} of {acc['ui_click']['n']} screens")
    if pr:
        gap = lambda s: f"{pr[s]['best_minus_open_points'][0]:.0f} points [{pr[s]['best_minus_open_points'][1]:.0f}, {pr[s]['best_minus_open_points'][2]:.0f}]"  # noqa: E731
        out += (f", and so does every hosted model (0.99 to 1.00): finding the element that serves a goal does not separate systems. One step of reasoning first (the cheaper plan, the item out of stock, the earliest date) does: "
                f"the open model scores {pr['ui_reason']['open_accuracy']:.2f} on the shared screens and the best hosted model {pr['ui_reason']['best_hosted_accuracy']:.2f}, {gap('ui_reason')} ahead, as we had predicted. "
                f"On state questions (is a dialog open, is an error shown, is the user signed in) the best hosted model is {gap('ui_state')} ahead.")
    else:
        out += f"; when the goal needs one step of reasoning first on {acc['ui_reason']['accuracy']:.2f}. State questions are right on {acc['ui_state']['accuracy']:.2f}."
    if x:
        det = x["disabled_button_detected"]["rate"]
        hosted_det = {k: v for k, v in det.items() if not k.startswith("Qwen")}
        low = [k for k, v in hosted_det.items() if v < 0.8]
        p, a = x["page_type"], x["already_done"]
        out += (f" The hosted results also exposed three weaknesses of our test, which we report rather than remove. A disabled main button, which our screens draw as a pale tint, is reported by the open model on {det['Qwen3-VL-4B + Glance']:.2f} of the screens that have it "
                f"and by {len(low)} of the {len(hosted_det)} hosted models on fewer than 0.80 (only {max(hosted_det, key=hosted_det.get)} sees it every time). "
                f"On page type every hosted model scores {p['all_items']['accuracy_same_items']['Claude Opus 5']:.3f} and the open model {p['all_items']['accuracy_same_items']['Qwen3-VL-4B + Glance']:.3f}, because on the "
                f"{p['screens_with_a_dialog_over_the_page']} of {p['of_screens']} screens where a dialog is open its backdrop hides the page; on the others every system is perfect, the open model included. "
                f"And for one goal (“go to the next page of results”) the after-screen does not show that anything was done; without it the open model is at {a['without_that_goal']['accuracy_same_items']['Qwen3-VL-4B + Glance']:.2f} "
                f"on “is this goal already done” and the hosted models at {min(v for k, v in a['without_that_goal']['accuracy_same_items'].items() if not k.startswith('Qwen')):.2f} to {max(v for k, v in a['without_that_goal']['accuracy_same_items'].items() if not k.startswith('Qwen')):.2f}. "
                "Table 6 keeps the registered numbers on all items.")
    return "\n  <p>" + out + "</p>"

finer_block = finer()
beyond_block = beyond()
beyond_section = (f'<section aria-labelledby="stops">\n  <h2 id="stops"><span class="num">3</span>Where the small open model falls behind: geometric judgements</h2>\n  {beyond_text()}\n  {beyond_block}\n</section>\n') if beyond_block else ""
t0 = 6 if beyond_block else 5  # tables after section 2 are numbered from here, so no number is skipped while Table 5 has no data
got = [r for r in (measured or []) if r["usd_per_1000_calls"] is not None]
api_lo, api_hi = min(r["usd_per_1000_calls"] for r in got), max(r["usd_per_1000_calls"] for r in got)
raw = lf["raw (0 labels, no pool)"]
r4a = ladder["summary"]["R4a"]["accuracy"] if ladder and "R4a" in ladder.get("summary", {}) else None
paired = {"Claude Opus 5": "+12.2 [8.1, 16.3]", "GPT-5.6": "+7.5 [3.0, 11.9]", "Gemini 3.1 Pro": "+2.1 [−2.3, 6.6]"}  # lab/NOTES.md entry 32b
today = datetime.date.today().isoformat()
version = next(line.split('"')[1] for line in (ROOT / "pyproject.toml").read_text().splitlines() if line.startswith("version"))

BODY = f"""
<header>
  <p class="running">glance-vlm · working paper · v{version} · snapshot {snapshot}, {today} · every experiment registered before it ran · results regenerate from the repository</p>
  <h1>Glance</h1>
  <p class="papertitle">Reading typed visual judgements from a frozen open vision-language model</p>
  <p class="subtitle">Coarse recognition close to the best hosted models (level on pick-one, two points behind on yes/no) is already in a small open vision-language model, and it can be read without generating. What remains hard about quality ratings is where the rubric draws its lines. On geometric judgements (relative size, line direction) the best hosted models are perfect and the small open model, read this way, is far behind.</p>
  <p class="byline">Yohei Nakajima <span class="aff">· independent · built with AI assistance throughout (the notebook records who did what)</span></p>
  <p class="links"><a href="#regimes">Summary</a> · <a href="#useit">Use it</a> · <a href="#method">Method</a> · <a href="#evidence">Yes/no and pick-one</a> · <a href="#stops">Where it stops</a> · <a href="#read">Read against write</a> · <a href="#ratings">Ratings</a> · <a href="#models">Scale and family</a> · <a href="#cost">Cost and speed</a> · <a href="#new">What is not new</a> · <a href="#limits">Limits and misses</a> · <a href="#refs">References</a></p>
</header>

{regimes()}
{use_it()}
<section aria-labelledby="abstract">
  <h2 id="abstract" class="plain">Abstract</h2>
  <p class="abstract">A <em>typed</em> question is one whose legal answers form a closed set known before the model runs: yes or no, one of a list, a level on a rubric. We put such questions about images to a frozen open vision-language model (Qwen3-VL-4B, Apache-2.0, on a laptop) and read the answer from the logits of one forward pass; nothing is generated.</p>
  <p class="abstract">{abstract_photos()} These questions are easy and the labels are imperfect: about 3 to 5% of items are answered “wrongly” by all seven systems. Reading is as accurate as the same model writing JSON.</p>
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
  <h2 id="evidence"><span class="num">2</span>Coarse yes/no and pick-one: level with the best hosted models on pick-one, two points behind on yes/no</h2>
  <p>The same items went to the open 4B model (read, and also writing its answer as JSON), to three hosted flagships and to each provider’s lowest-cost current vision model. Three tests, all zero-shot: {lower_keep(matrix["tests"]["yesno"])}; {lower_keep(matrix["tests"]["choice"])}; {lower_keep(matrix["tests"]["rating"])} (ratings are the subject of section 5).</p>
  <figure>{fig_bars}<figcaption><b>Figure 1.</b> Accuracy, seconds and dollars for every system, by question type, on scales that start at zero, on the items every system answered (Table 2 gives the counts). {dark_note}{" An outlined bar is an estimated cost; an estimate beyond the measured range is cut short and marked ›." if has_est else ""}</figcaption></figure>
  <p>Within any one photo set every 95% interval overlaps every other, which says as much about sample size as about the systems. Pooled over the three photo sets and paired on the same items, differences appear. {pooled_sentence("yesno", "yes/no")} {pooled_sentence("choice", "pick-one")} Two cautions apply. The questions are coarse (is there a bridge; which of thirteen everyday things is this), so they measure a floor that all current systems clear. And the labels are not gold: on the Commons set, {noise_n("fresh_yesno")} of 131 yes/no items and {noise_n("fresh_choice")} of 65 pick-one items are answered “wrongly” by all seven systems, which is more likely a wrong or ambiguous label than seven identical mistakes (a proxy; no human audit was done). On ratings the leader depends on what is allowed: with nothing fitted a low-cost hosted model leads; once the open model has seen a few images of the rubric it is level or ahead (section 5).</p>
  {matrix_tables(ALL)}
  <p class="caption"><b>Table 2.</b> The numbers behind Figure 1, on the items every system answered ({basis_note}). Accuracy with 95% bootstrap intervals; bold rows are open models ({open_rows_note}). Hosted speed is wall time per call from one laptop, network included; hosted cost is the provider’s bill where it was logged (“est.” is a list-price upper estimate). The open model was timed on the same photographs with the GPU otherwise idle; its cost is those seconds at an on-demand cloud GPU price.{size_note} No few-shot prompt was tried for any written row.</p>
  {pooled_table()}
  <p class="caption"><b>Table 3.</b> The three fresh photo sets pooled (Commons, iNaturalist ten groups, iNaturalist insect orders), on the items the hosted models were asked; a failed call counts as wrong. Differences are paired on the same items, with 95% intervals from a bootstrap stratified by photo set. This pooling was added after the per-set results had been seen; its rule (every set, every system, nothing dropped) was fixed before it was computed.</p>
  <h3>2.1 The photographs</h3>
  <p>Wikimedia Commons photographs taken after 15 August 2026, labelled by their uploaders’ structured “depicts” statements, and iNaturalist observations uploaded on the day of the test, labelled by community identification. No labels were made by us or by any model.</p>
  {fresh_table()}
  <p class="caption"><b>Table 4.</b> The open model on all items of each photo set, hosted models on the test half; 95% bootstrap intervals, uncalibrated decisions, nothing fitted. {apart()}</p>
  <figure>{fig1}<figcaption><b>Figure 2.</b> The Commons rows of Table 4, drawn to one scale. Filled marks are the open 4B model; hollow marks are hosted frontier models. Every interval overlaps every other.</figcaption></figure>
  <p>{older()}</p>

  <h3>2.2 A finer test</h3>
  {finer_block}
</section>

{beyond_section}
<section aria-labelledby="read">
  <h2 id="read"><span class="num">4</span>Reading is as accurate as writing, and gives the same answer when the prompt is the same</h2>
  <p>The alternative to reading is to let the same model write a JSON answer. With ground truth and one question per request, the two agree. On yes/no and pick-one the accuracy is the same on both photo sets (differences of {rw_gap[0]:.1f} to {rw_gap[1]:.1f} points, every interval spanning zero; Table 4), and each item comes out the same way, right or wrong, on {rw_same[0]:.1%} to {rw_same[1]:.1%} of items; the few that differ reflect the prompts, which are not the same (a JSON request against one statement per option). On ratings, where the digits can be read at the very position where the written answer puts them, {jd["agree_with_written"]:.0%} of answers are identical: a written answer under greedy decoding is an argmax over the same logits, so this is expected.</p>
  <p>They stop agreeing when the prompts differ. Our earlier four-pass rating readout uses different wording from the JSON prompt and scores {100 * (written - jd["ens_zero"]):.0f} points lower zero-shot. And when one written JSON object carries 25 ratings, each field is conditioned on the fields already written, while 25 separate reads are independent: the two agree on only {genb["25 ratings"]["agreement_with_read_on_valid_fields"]:.0%} of fields (that request has no ground truth, so this is a difference, not an error rate). Reading is therefore not a free substitute for any prompt; it is a way to take the same decision without generating it.</p>
  <p>What reading changes is cost and form: about 1.5 times faster for one question about a full-size photograph (encoding the image dominates), 2 to 3 times on small images, more as questions per image grow (section 8 sets these ratios beside independent timings of a hosted text product of the same kind); and the answer is a probability vector, which can be thresholded, ranked and fitted.</p>
  {cost_table()}
  <p class="caption"><b>Table {t0 + 1}.</b> The open model writing against reading, cost per 1,000 images on a rented GPU assumed no faster than the laptop; self-hosted cost is GPU time, so the saving is the measured time saving. “Agree” is the share of answer fields on which the written and the read answer are the same. For comparison, the frontier calls on these tasks measured ${api_lo:.2f} to ${api_hi:.2f} per 1,000 answers.</p>
</section>

<section aria-labelledby="ratings">
  <h2 id="ratings"><span class="num">5</span>Ratings: models get the order right and the boundaries wrong; a few images of the rubric fix the offset</h2>
  <p><b>Scope.</b> “Ratings” in this section means five synthetic, single-factor, four-level image-quality scales (blur, exposure, JPEG, noise, resolution) on 1,000 held-out images, the same for every system; it does not mean aesthetic judgement. The pattern in the title holds on these scales and does not hold outside them: section 5.1 has KADID-10k, rubrics that are not image quality, and the specialised tools that do as well or better.</p>
  <p><b>Order against boundaries.</b> Zero-shot the open model is exactly right on {raw["accuracy"]:.2f} of images and within one level on {raw["within_1"]:.3f}; 97% of its errors are one step, in a direction that is constant per rubric (half a level harsh on blur, never the worst level on JPEG). The hosted models score {hosted_rating[0]:.2f} to {hosted_rating[-1]:.2f}; each provider’s low-cost model beats its own flagship, and {best_rating} leads. Where a rubric’s author drew the lines is a convention that no model can know unseen.</p>
  <figure>{fig2}<figcaption><b>Figure 3.</b> Exact-level accuracy on the same 1,000 images, chance 0.25. Upper group: zero-shot. Lower group: the open model after seeing images of the rubric, first unlabeled (zero labels, but not zero-shot), then 32 labeled.</figcaption></figure>
  <p><b>Fitting the offset.</b> The two fits of section 1 act on this offset. The comparison is asymmetric by design: the hosted models stayed zero-shot, because a written pick offers nothing to fit and we did not give them few-shot examples. It shows what a local, fittable readout buys a user; it does not show that the open model sees better.</p>
  {extras_table()}
  <p class="caption"><b>Table {t0 + 2}.</b> Exact-level accuracy on the rating test of Table 2 (the last row uses the full test split and is a research result, not shipped). Unlabeled fitting roughly halves the calibration error (ECE 0.33 to about 0.2); only the labeled fit gives calibrated probabilities (ECE about 0.03). The fitted four-pass number appears on this page in four values that differ by item set and label draw, not by method: 0.857 (here) and 0.853 (Table {t0 + 4}) are two independent sets of twenty random 32-label draws on these 1,000 images; 0.846 (section 5.1) is the 32-label fit on the 1,500 images the outside systems were run on; 0.867 is the fit with 500 labels on the full test split.</p>
  <h3>5.1 Outside the quality scales</h3>
  <p><b>A real image-quality benchmark.</b> On KADID-10k (23 distortion types, five levels, human scores) the fitted open model reached 0.527 exact and missed every target we had registered. Zero-shot it is at {kadid_json:.2f} with the one-pass read and {kadid_ens:.2f} with the four-pass read; that check, together with the rubrics below, is what made the one-pass read the default for a rubric with nothing fitted, by a rule fixed in advance; the gain on KADID-10k was {100 * (kadid_json - kadid_ens):.1f} points where we had predicted at least five.</p>
  <p><b>Specialised tools.</b> With plentiful labels, 29 hand-built features score 0.979 on the synthetic scales against 0.867 for the fitted model.{outside_sentence()} For low-level artefacts these remain the better tools; a general model read this way earns its place by answering any typed question with one set of frozen weights.{openjev_sentence()}</p>{other_rubrics()}
</section>

<section aria-labelledby="models">
  <h2 id="models"><span class="num">6</span>Scale and family: quality belongs to the model, size above 4B buys time and cost but no accuracy</h2>
  <p>The same prompts and readouts, not a word changed, on other sizes of the same family and on a model from a different family (different vision tower, different language model).</p>
  {closed_set_table()}
  <p class="caption"><b>Table {t0 + 3}.</b> Yes/no and pick-one on the three fresh photo sets, all items, uncalibrated. Within the Qwen3-VL family the 4B and 8B models are level everywhere; the 2B model keeps up on the two easier sets and falls behind on the insect orders (pooled and paired, {gap_2b:.1f} points behind the 4B model on yes/no), which is why the headline chart carries all three sizes. The 2.2B model of another family is level with them on everyday photographs, trails on nature photographs (5 points on yes/no, 12 on pick-one){smol_orders}, so the comparison with hosted models in section 2 is a statement about this family, not about every small open model.</p>
  {models_table()}
  <p class="caption"><b>Table {t0 + 4}.</b> Ratings, the same 1,000 images: exact-level accuracy. From 2B to 4B zero-shot accuracy rises sharply; from 4B to 8B it does not rise at all, and after 32 labels the three sizes are within 1.4 points. We had predicted a monotone rise and were wrong. “Within one” is for the four-pass read.</p>
  {size_speed()}
</section>

<section aria-labelledby="cost">
  <h2 id="cost"><span class="num">7</span>Cost and speed: same order as the low-cost hosted models, one to two orders below the flagships</h2>
  <p>The two sides of this comparison are not measured the same way, and the caveats come first. Hosted cost is the provider’s bill per call and includes nothing for an operator; hosted latency is wall time from one laptop and includes the network and the provider’s queue. Open-model cost is measured seconds on a laptop multiplied by an on-demand cloud GPU price, with no batching, no idle time and no operator counted; a different GPU price or image resolution moves it by more than the gap to the low-cost hosted models. Hosted cost also depends on image size (GPT-5.6: $7.55 per 1,000 on 1,280-pixel files, $1.86 on 500-pixel files).</p>
  <p>With that said: a yes/no about a full-size photograph takes the open 4B model {own["seconds"]["yesno"]["value"]:.1f} s on a laptop and costs ${own["usd_per_1000"]["yesno"]["value"][0]:.2f} to ${own["usd_per_1000"]["yesno"]["value"][1]:.2f} per 1,000 on a rented GPU; the low-cost hosted models cost ${cheap_yesno[0]:.2f} and up, the flagships several dollars. The durable differences are not the cents: the image never leaves the machine, there is no per-call bill, it works offline, and the answer can be fitted.</p>
  <figure>{fig_acc_cost}<figcaption><b>Figure 4.</b> Accuracy against cost, one panel per question type, 95% intervals. Filled circles are Qwen3-VL read with Glance, labelled by size; the filled square is the 4B model writing JSON. Hosted models: {site_charts.hosted_key(mrows)}.</figcaption></figure>
  <figure>{fig_cost_speed}<figcaption><b>Figure 5.</b> Cost against speed; down and left is better. Each system is a large mark at the centre (geometric mean) of three small ones, one per question type: {site_charts.KEY}. Filled marks are open models on a laptop.</figcaption></figure>
</section>

<section aria-labelledby="new">
  <h2 id="new"><span class="num">8</span>What is not new, and the work this sits in</h2>
  <p>Reading answer-token logits from a frozen generative model, with several questions sharing one image prefix, is what Simple Jev, jev-visual and LitJev also do. For yes/no and pick-one the forward pass here is not new, and the accuracy belongs to the open model. What this project adds is a harness on top: a fresh-photograph comparison with paid frontier calls, the same model writing against reading, self-calibration from unlabeled images and labeled fitting for rating levels (<code>glance fit</code>), and measured dollars and milliseconds. It trains no weights, unlike YOFO (Zhang et al. 2025), Laya Vision or OpenJev v2.</p>
  {speed_context()}
  <p>The pieces are older than any of these tools. Scoring a closed set of candidate answers instead of generating is standard for language models (Kadavath et al. 2022) and its option-letter pitfalls are known (Zheng et al. 2024); VQAScore reads P(“Yes”) from a VQA model in one pass (Lin et al. 2024); a rubric score as a probability-weighted sum over rating tokens is G-Eval (Liu et al. 2023); level-token readouts for image quality are Q-Bench and Q-Align (Wu et al. 2024a, b), and frozen-CLIP quality scores are CLIP-IQA (Wang et al. 2023); the fits are matrix scaling and its relatives (Guo et al. 2017; Kull et al. 2019); the hidden-state variant is a linear probe (Alain and Bengio 2016); sharing an image prefix across questions is prefix caching (vLLM). Our additions are the contamination-controlled comparison, the read-against-write control, the unlabeled fit for ratings with the failed content-free prior as its contrast, and the registered misses.</p>
</section>

<section aria-labelledby="limits">
  <h2 id="limits"><span class="num">9</span>Limits and misses</h2>
  <ul class="misses">
    <li><span class="verdict miss">not supported</span> A content-free prior (blank and noise images) was expected to help zero-shot ratings. It took exact accuracy from {nullp["table"]["raw zero-shot"]["accuracy"]:.3f} to {nullp["table"]["content-free prior, all six null images (registered)"]["accuracy"]:.3f}: for an image rubric there is no content-free image.</li>
    <li><span class="verdict miss">not supported</span> Our four-pass rating readout was expected to match the same model’s written answer zero-shot. It trailed it by ten points: a readout selected with a calibration in the loop is good to fit and poor zero-shot. <span class="verdict">supported</span> The registered fix, one pass read at the JSON answer position, matches the written answer ({jd["json_zero"]:.3f}) and reaches {jd["json_u16"]:.3f} with 16 unlabeled images.</li>
    <li><span class="verdict miss">not supported</span> We expected each provider’s cheapest model to score at or below its flagship on ratings. Every one beats its flagship, and {best_rating} ({best_rating_acc:.3f}, ${cheap_rating[0]:.2f} to ${hosted[best_rating]["usd_per_1000"]["rating"]["value"][0]:.2f} per 1,000 among the cheap models) is nine points ahead of the open model zero-shot. With 16 unlabeled images the open model reaches {jd['json_u16']:.3f}, {100 * (best_rating_acc - jd['json_u16']):.1f} points short of it; with 32 labels it leads.</li>
    <li><span class="verdict miss">not supported</span> On KADID-10k (23 distortion types, five levels, human scores) every registered target was missed: 0.527 exact with labels; zero-shot 0.33 with the four-pass read and 0.35 with the one-pass read.</li>
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
<title>Glance: reading typed visual judgements from a frozen open vision-language model</title>
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
.papertitle{font-size:1.5rem;line-height:1.25;font-weight:600;margin:.1rem 0 1rem;max-width:34rem}@media (max-width:480px){.papertitle{font-size:1.25rem}}
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
.regimes{margin:1.2rem 0 1.6rem}.regimes td{white-space:normal;vertical-align:top;font-size:.93rem;line-height:1.45;padding-top:.6rem;padding-bottom:.6rem}.regimes td:first-child{min-width:8.5rem}.regimes td:nth-child(2){min-width:17rem}.regimes td:nth-child(3){min-width:12rem}
@media (max-width:620px){.regimes thead{display:none}.regimes table,.regimes tbody,.regimes tr,.regimes td{display:block}.regimes td{min-width:0!important;padding:.15rem 0}.regimes tr{padding:.65rem 0;border-top:.75px solid var(--rule)}.regimes tr:first-child{border-top:0}.regimes td:nth-child(3){font-style:italic}}
.useit{border:1px solid var(--rule);background:var(--tint);padding:1.05rem 1.25rem .95rem;margin:1.4rem 0 2.4rem;font:400 .92rem/1.5 "IBM Plex Sans","Helvetica Neue",Arial,sans-serif}
.useit-top{display:flex;justify-content:space-between;align-items:baseline;gap:.4rem 1rem;flex-wrap:wrap}.useit h2{margin:0;border:0;padding:0}.useit .gh{font-weight:500;white-space:nowrap}
.useit p{margin:.55rem 0}
.useit .tryit a{font-weight:500;white-space:nowrap;margin-right:.9rem}.useit .tryit .note{display:block;margin-top:.2rem;color:var(--muted);font-size:.84rem}.useit pre{background:var(--paper);margin:.75rem 0 .7rem;font-size:.78rem}.useit ul{margin:.3rem 0 0;padding-left:1.1rem}.useit li{margin:.28rem 0}
.useit code{font-size:.86em}.useit li code{white-space:nowrap}.useit-foot{font-size:.76rem;color:var(--muted);margin:.8rem 0 0}
pre{background:var(--tint);border-left:2px solid var(--rule);padding:.8rem 1rem;overflow-x:auto;font:400 .82rem/1.55 "IBM Plex Mono",ui-monospace,Menlo,monospace}
code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.88em}
footer{font-size:.8rem;line-height:1.55;color:var(--muted);border-top:1px solid var(--rule);padding-top:1rem}
@media (max-width:480px){h1{font-size:2.1rem}.subtitle{font-size:1.12rem}body{font-size:1rem}}
""" + site_charts.CSS + "</style>\n")

page = STYLE + "<main>" + BODY + "</main>\n"
(ROOT / "site").mkdir(exist_ok=True)
(ROOT / "site/page.html").write_text(page)
# ---- link previews (messages, social posts): description, Open Graph and a card image; numbers from the pooled analysis -------
_y, _c = POOLED["kinds"]["yesno"]["systems"], POOLED["kinds"]["choice"]["systems"]
_best = lambda e: max(r["pooled"][0] for n, r in e.items() if "verdict" in r)  # noqa: E731
DESCRIPTION = (f"Typed questions about images, answered from one forward pass of a frozen open 4B vision-language model on a laptop. On fresh photographs: pick-one {_c['Qwen3-VL-4B, read']['pooled'][0]:.3f} "
               f"against {_best(_c):.3f} for the best hosted model, yes/no {_y['Qwen3-VL-4B, read']['pooled'][0]:.3f} against {_best(_y):.3f}. Every experiment registered before it ran; misses published.")
META = (f'<meta name="description" content="{html.escape(DESCRIPTION)}">\n<link rel="canonical" href="https://glance.yohei.me/">\n'
        f'<meta property="og:type" content="article">\n<meta property="og:title" content="Glance: reading typed visual judgements from a frozen open vision-language model">\n<meta property="og:description" content="{html.escape(DESCRIPTION)}">\n'
        '<meta property="og:url" content="https://glance.yohei.me/">\n<meta property="og:image" content="https://glance.yohei.me/og.png">\n<meta property="og:image:width" content="1200">\n'
        '<meta property="og:image:height" content="630">\n<meta name="twitter:card" content="summary_large_image">\n')
CARD = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>body{{margin:0;width:1200px;height:630px;background:#fff;color:#14171a;font-family:"STIX Two Text",Georgia,serif}}
.w{{padding:64px 80px}}.k{{font:500 20px "IBM Plex Sans",Arial,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:#56616b}}
h1{{font-size:92px;margin:14px 0 8px;font-weight:600}}.s{{font-size:33px;line-height:1.32;max-width:1010px}}
.r{{display:flex;gap:56px;margin-top:40px;border-top:2px solid #14171a;padding-top:24px}}.r div{{font:400 20px "IBM Plex Sans",Arial,sans-serif;color:#56616b}}
.r b{{display:block;font:600 42px "STIX Two Text",Georgia,serif;color:#14171a;margin-bottom:4px}}.u{{position:absolute;right:80px;bottom:46px;font:500 22px "IBM Plex Sans",Arial,sans-serif;color:#56616b}}</style></head>
<body><div class="w"><div class="k">Working paper · every experiment registered before it ran</div><h1>Glance</h1>
<div class="s">Typed questions about images, read from one forward pass of a frozen open 4B model. Level with the best hosted models on pick-one, two points behind on yes/no; geometry is where it stops.</div>
<div class="r"><div><b>{_c['Qwen3-VL-4B, read']['pooled'][0]:.3f} <span style="font-weight:400;color:#56616b">vs {_best(_c):.3f}</span></b>pick-one, best hosted model</div>
<div><b>{_y['Qwen3-VL-4B, read']['pooled'][0]:.3f} <span style="font-weight:400;color:#56616b">vs {_best(_y):.3f}</span></b>yes/no, best hosted model</div>
<div><b>{own["seconds"]["yesno"]["value"]:.1f} s</b>per yes/no, on a laptop</div></div></div><div class="u">glance.yohei.me</div></body></html>"""
(ROOT / "site/og_card.html").write_text(CARD)  # rendered to site/og.png by tools/make_og.py (needs a local Chrome); not uploaded itself

(ROOT / "site/index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                      + META + STYLE + "</head>\n<body>\n<main>" + BODY + "</main>\n</body>\n</html>\n")
print("wrote site/index.html and site/page.html,", len(page) // 1024, "KB")

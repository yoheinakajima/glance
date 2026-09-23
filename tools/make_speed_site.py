"""Build the Glance Speedlab companion page from a frozen release summary."""

import html
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULT = json.loads((ROOT / "results/speedlab/v0.1.0.json").read_text())
OUT = ROOT / "site/speed"
OUT.mkdir(parents=True, exist_ok=True)
HEAD = RESULT["headline"]
OTHER = RESULT["other_results"]


def esc(value: object) -> str:
    return html.escape(str(value))


PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glance Speedlab — what actually made local camera VLM inference faster</title>
<meta name="description" content="Twenty-one measured experiments on low-latency local camera inference with Glance and an experimental 8-bit MLX scorer.">
<link rel="canonical" href="https://glance.yohei.me/speed/"><meta property="og:title" content="Glance Speedlab">
<meta property="og:description" content="What worked, what did not, and why: 21 local VLM latency experiments on Apple Silicon."><meta property="og:url" content="https://glance.yohei.me/speed/">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<style>
:root{{--paper:#fff;--ink:#14171a;--muted:#5d6770;--rule:#d7dce1;--link:#174680;--tint:#f3f5f7;--good:#17623c;--miss:#8f2d2d}}@media(prefers-color-scheme:dark){{:root{{--paper:#0f1214;--ink:#e5e9ec;--muted:#9ba6ae;--rule:#2c3339;--link:#91b7e9;--tint:#171b1f;--good:#7fd1a4;--miss:#eda2a2}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:400 1.06rem/1.62 "STIX Two Text",Georgia,serif;padding:2.5rem 1.25rem 5rem}}main{{max-width:68rem;margin:auto}}a{{color:var(--link);text-underline-offset:.18em}}a:focus-visible{{outline:2px solid var(--link);outline-offset:3px}}.eyebrow,.nav,.stat span,.card small,th,.note,footer{{font-family:"IBM Plex Sans",Arial,sans-serif}}.eyebrow{{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule);padding-bottom:.75rem}}h1{{font-size:clamp(2.8rem,7vw,5.6rem);line-height:.94;letter-spacing:-.045em;margin:2.5rem 0 1.3rem;max-width:12ch}}.dek{{font-size:clamp(1.25rem,2.4vw,1.65rem);line-height:1.4;max-width:42rem;margin:0 0 1.4rem}}.nav{{font-size:.92rem;color:var(--muted);display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:3rem}}.hero-grid{{display:grid;grid-template-columns:1.2fr .8fr;gap:1.5rem;border-block:1px solid var(--rule);padding:1.5rem 0}}.headline{{font-size:clamp(2.3rem,5vw,4.4rem);line-height:1;letter-spacing:-.04em;margin:0}}.headline strong{{color:var(--good)}}.measure{{margin:.8rem 0 0;max-width:35rem}}.scope{{background:var(--tint);padding:1.1rem 1.25rem;margin:0;font-size:.98rem}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rule);margin:3rem 0}}.stat{{background:var(--paper);padding:1.2rem}}.stat b{{display:block;font-size:1.75rem;line-height:1.15}}.stat span{{display:block;color:var(--muted);font-size:.78rem;line-height:1.35;margin-top:.4rem}}section{{margin-top:4.3rem}}h2{{font-size:1.45rem;border-top:1.5px solid var(--ink);padding-top:.75rem;margin:0 0 1rem}}h3{{font-size:1.08rem;margin:.1rem 0 .45rem}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:.94rem}}th,td{{border-bottom:1px solid var(--rule);padding:.75rem .65rem;text-align:left;vertical-align:top}}th{{font-size:.78rem;color:var(--muted);font-weight:500}}.keep{{color:var(--good)}}.reject{{color:var(--miss)}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem}}.card{{display:block;border:1px solid var(--rule);padding:1.25rem;text-decoration:none;color:var(--ink);min-height:9rem}}.card:hover{{border-color:var(--ink)}}.card small{{color:var(--muted);display:block;margin-top:.45rem}}.arrow{{float:right;color:var(--link)}}pre{{overflow:auto;background:var(--tint);padding:1rem 1.1rem;font:400 .83rem/1.55 "IBM Plex Mono",monospace}}code{{font-family:"IBM Plex Mono",monospace;font-size:.88em}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:2rem}}ul,ol{{padding-left:1.2rem}}li{{margin:.45rem 0}}.note{{font-size:.84rem;color:var(--muted)}}footer{{border-top:1px solid var(--rule);margin-top:4.5rem;padding-top:1rem;color:var(--muted);font-size:.82rem}}@media(max-width:760px){{.hero-grid,.two{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr 1fr}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">Glance Speedlab · technical report · {esc(RESULT['snapshot'])} · {esc(RESULT['date'])}</p>
<h1>What actually made a local camera VLM faster?</h1>
<p class="dek">Twenty-one measured experiments across capture, transport, batching, model size, caching, quantization, token count and decoder depth—with the misses kept beside the wins.</p>
<p class="nav"><a href="../">Glance paper</a><a href="https://github.com/yoheinakajima/glance-speedlab">Source and local demo</a><a href="https://github.com/yoheinakajima/glance-speedlab/blob/main/paper/PAPER.md">Full report</a><a href="#run">Run it</a></p>
<div class="hero-grid"><div><p class="headline"><strong>{esc(HEAD['latency_reduction_pct'])}% lower</strong> fresh-frame latency</p><p class="measure">The accepted 8-bit MLX scorer moved the same nine-statement workload from {esc(HEAD['reference_p50_ms'])} to {esc(HEAD['candidate_p50_ms'])} ms p50 ({esc(HEAD['speedup'])}×). Exact replication measured {esc(HEAD['replication_latency_reduction_pct'])}%.</p></div><p class="scope"><b>Scope matters.</b> This is a fixed-suite result on one {esc(RESULT['hardware'])} machine, not a universal model claim. The candidate matched {esc(HEAD['decision_matches'])}/{esc(HEAD['decision_total'])} decisions with maximum probability drift {esc(HEAD['max_probability_drift'])}. Glance remains the reference backend; MLX is an opt-in Apple Silicon path.</p></div>
<div class="stats" aria-label="Headline findings"><div class="stat"><b>{esc(OTHER['native_batch_speedup'])}×</b><span>one native four-question request vs four cold sequential requests</span></div><div class="stat"><b>{esc(HEAD['speedup'])}×</b><span>8-bit MLX direct scorer vs reference fresh-frame path</span></div><div class="stat"><b>{esc(OTHER['small_model_speedup'])}×</b><span>2B vs 4B, rejected as a global swap at {esc(OTHER['small_model_agreement_pct'])}% agreement</span></div><div class="stat"><b>{esc(OTHER['synthetic_temporal_trigger_reduction_pct'])}%</b><span>fewer synthetic inference triggers; real-camera validation pending</span></div></div>
<section><h2>The result, without the victory lap</h2><div class="table-wrap"><table><thead><tr><th>Intervention</th><th>What happened</th><th>Decision</th></tr></thead><tbody>
<tr><td>Native multi-question request</td><td>{esc(OTHER['native_batch_speedup'])}× faster; exact decisions within tolerance</td><td class="keep">Keep as the default architecture</td></tr>
<tr><td>8-bit MLX direct scoring</td><td>{esc(HEAD['speedup'])}× faster; {esc(HEAD['decision_matches'])}/{esc(HEAD['decision_total'])} decisions; max drift {esc(HEAD['max_probability_drift'])}</td><td class="keep">Ship as an experimental local backend</td></tr>
<tr><td>2B instead of 4B</td><td>{esc(OTHER['small_model_speedup'])}× faster; {esc(OTHER['small_model_agreement_pct'])}% decision agreement</td><td>Fast tier only; do not silently replace 4B</td></tr>
<tr><td>4-bit MLX direct scoring</td><td>Speed held, but maximum probability drift reached 0.361</td><td class="reject">Reject at the declared quality bound</td></tr>
<tr><td>Fixed-shape suffix compilation</td><td>{esc(OTHER['compiled_suffix_improvement_pct'])}% p50 improvement</td><td class="reject">Below materiality; keep eager execution</td></tr>
<tr><td>Uniform vision-token reduction</td><td>Up to 15.6% faster, but a decision changed and drift reached 0.411</td><td class="reject">Reject; investigate learned token selection</td></tr>
<tr><td>Untrained early decoder exits</td><td>6–22% faster; no tested depth passed both gates</td><td class="reject">Reject; a trained head is required</td></tr>
<tr><td>Browser payload work</td><td>Base64 and JSON cost at most {esc(OTHER['transport_p95_ms'])} ms p95 at 320 px</td><td>Measure it, then focus on model compute</td></tr>
</tbody></table></div></section>
<section id="run"><h2>Run it locally</h2><div class="two"><div><h3>Reference path</h3><p>Clone Glance beside Speedlab, prepare the model once, then launch the loopback-only stack.</p><pre><code>git clone https://github.com/yoheinakajima/glance
git clone https://github.com/yoheinakajima/glance-speedlab
cd glance &amp;&amp; uv sync &amp;&amp; uv run glance doctor --json
cd ../glance-speedlab
corepack enable &amp;&amp; pnpm install --frozen-lockfile
pnpm launch</code></pre></div><div><h3>8-bit MLX path</h3><p>On Apple Silicon, install the pinned MLX environment and launch both backends for an A/B comparison.</p><pre><code>uv venv research/runs/e013-venv --python 3.11
uv pip install --python research/runs/e013-venv/bin/python \
  -r requirements-mlx.txt
pnpm launch:mlx</code></pre><p class="note">Tested on Apple M5 with 32 GB unified memory; 16 GB or more is recommended. The pinned 8-bit 2B checkpoint is about 2.7 GB. Smaller-memory machines and other Apple chips were not measured.</p></div></div></section>
<section><h2>What belongs where</h2><div class="two"><div><h3>Glance</h3><p>The stable typed-question protocol, direct probability readout and shared-prefix multi-question path. General improvements need broader correctness evidence before landing here.</p></div><div><h3>Speedlab</h3><p>The live scheduler, backend A/B interface, telemetry, experimental 8-bit MLX scorer, rejected variants and research record. It is the faster-moving proving ground.</p></div></div></section>
<section><h2>Follow the evidence</h2><div class="cards"><a class="card" href="https://github.com/yoheinakajima/glance-speedlab/blob/main/paper/PAPER.md"><span class="arrow">↗</span><b>Technical report</b><small>Method, interpretation, limitations and next hypotheses.</small></a><a class="card" href="https://github.com/yoheinakajima/glance-speedlab/blob/main/paper/ARTIFACTS.md"><span class="arrow">↗</span><b>Claim-to-artifact map</b><small>Every public number linked to aggregate JSON and its experiment record.</small></a><a class="card" href="https://github.com/yoheinakajima/glance-speedlab/tree/main/research/experiments"><span class="arrow">↗</span><b>All 21 experiments</b><small>Pre-registered hypotheses, stopping rules, outcomes and misses.</small></a><a class="card" href="https://github.com/yoheinakajima/glance"><span class="arrow">↗</span><b>Glance</b><small>The stable local typed-decision engine and original working paper.</small></a></div></section>
<section><h2>Next hypotheses</h2><ol><li>A trained intermediate decision head can retain full-depth margins while recovering the 11–22% compute exposed by rejected raw truncation.</li><li>Learned or task-conditioned vision-token selection can recover the measured 54 ms prefix opportunity without uniform-resizing quality loss.</li><li>Model-shaped fused 8-bit Metal kernels can improve the accepted MLX path beyond generic eager execution.</li><li>A calibrated 2B→4B cascade can keep most of the 2B speedup while escalating uncertain or fine-detail frames.</li><li>Temporal reuse can reduce effective compute on labeled camera streams if change detection is task-aware and staleness is bounded.</li></ol></section>
<footer>Snapshot {esc(RESULT['snapshot'])}. This page summarizes a working technical report, not peer-reviewed evidence. Code is MIT licensed; model weights retain their own licenses. No camera frame is included in the public artifact.</footer>
</main></body></html>"""

(OUT / "index.html").write_text(PAGE)
(OUT / "llms.txt").write_text(
    "Glance Speedlab\n================\n\n"
    f"{RESULT['experiments']} measured experiments on low-latency local camera VLM inference.\n"
    f"Headline: 8-bit MLX direct scoring reduced fresh-frame p50 from {HEAD['reference_p50_ms']} ms to {HEAD['candidate_p50_ms']} ms ({HEAD['speedup']}x; {HEAD['latency_reduction_pct']}% lower) on {RESULT['hardware']}.\n"
    f"Guardrail: {HEAD['decision_matches']}/{HEAD['decision_total']} fixed-suite decisions matched; maximum probability drift {HEAD['max_probability_drift']}.\n"
    "Source and full evidence: https://github.com/yoheinakajima/glance-speedlab\n"
)
print("wrote site/speed/index.html and site/speed/llms.txt")

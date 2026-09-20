"""Writes docs/paper/RESULTS_COMPARISONS.md from the comparison result files that exist; missing ones are marked pending.

uv run python tools/make_results_comparisons.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


out = ["# Comparisons with similar systems: accuracy, speed, cost", "",
       "Branch `score-lab`. Plan and registrations: `lab/NOTES.md` entries 20, 23 and 24. In result rows the configuration is "
       "written \"Qwen3-VL-4B + Glance\"; Glance is a readout and calibration recipe, not a model. Every comparison below "
       "uses identical images and labels for all systems in the same table. Sections marked pending fill in as data lands.", ""]

dual = load("results/lab/dual_encoder_vs_vlm_same_items.json")
out += ["## 1. A frozen dual encoder against the VLM readouts (lab scales, identical items)", ""]
if dual:
    scales = dual["scales"]
    names = {"SigLIP2 as shipped": "SigLIP2 base, 256 px, one statement per level (CLIP-IQA-style frozen encoder), as shipped",
             "SigLIP2, matrix": "SigLIP2 + the same matrix calibration", "v0 readout as shipped": "Qwen3-VL-4B, yes/no per level (v0 / P(True)-style), as shipped",
             "v0 readout, matrix": "Qwen3-VL-4B, yes/no per level + matrix calibration", "digits": "Qwen3-VL-4B + Glance `digits` (1 pass)",
             "fast2": "Qwen3-VL-4B + Glance `fast2` (2 passes)", "ens4d": "Qwen3-VL-4B + Glance `ens4d` (4 passes)"}
    out += [f"{dual['n_fit']} calibration labels and {dual['n_test']} test images per scale, the same for every row. SigLIP2 answers in 38 ms per image.", "",
            "| System | " + " | ".join(scales) + " | mean accuracy |", "| --- | " + " | ".join("---" for _ in scales) + " | --- |"]
    for key, label in names.items():
        row = dual["systems"][key]
        out.append(f"| {label} | " + " | ".join(f"{row[s]:.3f}" for s in scales) + f" | **{sum(row[s] for s in scales) / len(scales):.3f}** |")
    out += ["", "Reading: a fitted map on the readout helps a contrastive encoder as well (0.330 to 0.588), so the effect is not specific to the "
            "VLM; and the VLM perceives much more of these attributes than the dual encoder does at 256 px.", ""]
else:
    out += ["Pending.", ""]

out += ["## 2. Other systems' readouts on the same frozen VLM (option letters, rotated letters, two poles)", ""]
letters = load("results/lab/readout_baselines.json")
out += ([letters["markdown"], ""] if letters else ["Pending: registered in `lab/NOTES.md` entry 24, queued on the GPU.", ""])

out += ["## 3. Classical no-reference image-quality features with the same labels", ""]
classical = load("results/lab/classical_baselines.json")
if classical:
    extras = load("lab/EXTRAS_ens4d.json")["learning_curve"]
    vlm_at = lambda n: sum(extras[sc][n]["accuracy"][0] for sc in extras) / len(extras)  # noqa: E731
    summ = {b: {"accuracy": v["summary"]["mean_accuracy_full"], "accuracy_small": v["summary"]["mean_accuracy_small_n"],
                "spearman_dmos": v["summary"].get("mean_spearman_vs_neg_dmos_full")} for b, v in classical["benchmarks"].items()}
    out += ["29 hand-built no-reference features (sharpness, high-frequency energy, noise estimate, 8-px blockiness, luminance and colour statistics, "
            "edge density, ...) with a standardized multinomial logistic regression, fit on the same calibration labels and scored once on the same test "
            "items (`tools/classical_baselines.py`, `results/lab/classical_baselines.md`; feature extraction about 13 ms per image on the CPU).", "",
            "| Lab scales (5 scales, 4 levels, 500 test images each) | about 32 labels per scale | all 500 labels per scale |", "| --- | --- | --- |",
            f"| Classical features + logistic regression | {summ['lab']['accuracy_small']:.3f} | **{summ['lab']['accuracy']:.3f}** |",
            f"| Qwen3-VL-4B + Glance `ens4d` | **{vlm_at('32'):.3f}** | {vlm_at('500'):.3f} |", "",
            "Reading, stated plainly: with enough labels, features built for exactly these artifacts beat the VLM on every lab scale, by 6 to 22 "
            "points. The VLM readout is the more label-efficient one (ahead by about 10 points at 32 labels) and it saturates early: more labels do not help it. "
            "So the lab scales do not show that a VLM is the best tool for blur or JPEG grading; they show how much of a described rubric a frozen "
            "general model can deliver from a few dozen labels, with no feature engineering. On the 25-distortion benchmarks the classical baseline reaches "
            f"{summ['distort25']['accuracy']:.3f} (`distort25`) and {summ['kadid']['accuracy']:.3f} (KADID-10k; Spearman with human scores "
            f"{summ['kadid']['spearman_dmos']:.3f}) with all calibration labels and {summ['distort25']['accuracy_small']:.3f} / "
            f"{summ['kadid']['accuracy_small']:.3f} with about 30; it is near chance where the distortion is structural rather than statistical "
            "(shuffled patches, colour diffusion). The VLM's numbers on those benchmarks are in `RESULTS_GENERALIZATION.md` when that collection finishes.", ""]
else:
    out += ["Pending: `tools/classical_baselines.py` (CPU). Expected to beat the VLM on simple ladders; that bounds the headroom honestly.", ""]

out += ["## 4. Frontier APIs on the same held-out images", ""]
frontier = load("results/lab/frontier_head_to_head.json")
out += (["See `results/lab/frontier_head_to_head.md`.", ""] if frontier and frontier.get("runs")
        else ["Pending: the three eval runs exist; the paid calls need the project owner's API key (`glance baseline --run <id> --env-file <path>`).", ""])

out += ["## 5. A second model family (SmolVLM2-2.2B), same prompts and recipe", "",
        "Pending: registered in `lab/NOTES.md` entries 20, 22 and 22b, queued on the GPU.", ""]

out += ["## 6. Open image-capable typed-decision systems and trained image-quality VLMs, as shipped", "",
        "Pending: license and feasibility survey in `docs/paper/COMPARABLE_SYSTEMS.md`. Only Apache-2.0 or MIT weights may be run here; the rest are "
        "cited with their published numbers and the caveat that blind MOS regression is a different task from level classification with the "
        "distortion named.", ""]

gen = load("lab/GENBENCH.json")
out += ["## 6b. Write the answers or read them? (same frozen VLM, same images, same questions)", ""]
if gen:
    out += ["`glance/lab/gen_bench.py`: cold start per image, end to end, idle GPU, 40 lab test images. Writing = greedy generation of one JSON "
            "object with a token cap (no thinking). Reading = `glance decide`, uncalibrated, which also returns a probability for every answer.", "",
            "| Request | write p50 ms (tokens written) | read `fast2` p50 ms | read `ens4d` p50 ms | reading is faster by | JSON failures | written = read |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, e in gen.items():
        ens = f"{e['read_ens4d_p50_ms']:.0f}" if "read_ens4d_p50_ms" in e else "-"
        speed = f"{e['speedup_vs_write_fast2']:.1f}x" + (f" / {e['speedup_vs_write_ens4d']:.1f}x" if "speedup_vs_write_ens4d" in e else "")
        out.append(f"| {name} | {e['write_p50_ms']:.0f} ({e['write_tokens_mean']:.0f}) | {e['read_fast2_p50_ms']:.0f} | {ens} | {speed} | "
                   f"{e['json_parse_failures']} of {e['images']} | {e['agreement_with_read_on_valid_fields']:.1%} |")
    out += ["", "The single yes/no was written as a small JSON object (7 tokens), not a bare word; a one-token answer would narrow that gap and was not "
            "measured. On 25 ratings the written and the read answers differ on four fields in ten; neither is calibrated and there is no ground truth "
            "for that request, so this is a difference, not a ranking.", ""]
else:
    out += ["Pending.", ""]

cost = load("results/lab/cost_model.json")
out += ["## 7. Cost of 1,000 ratings", ""]
if cost:
    f = lambda lo_hi, d=3: f"${lo_hi[0]:.{d}f} to ${lo_hi[1]:.{d}f}"  # noqa: E731
    out += ["Seconds are measured (`lab/PACKING.json`); dollar figures are arithmetic on stated assumptions (`tools/cost_model.py`), not measurements. "
            "API figures are list-price upper estimates from `glance baseline --estimate-only`; no call was made.", "",
            "| Qwen3-VL-4B + Glance, self-hosted | seconds per 1,000 ratings | electricity only | rented cloud GPU at laptop speed | laptop amortized |",
            "| --- | --- | --- | --- | --- |"]
    for name, r in cost["local"].items():
        out.append(f"| {name} | {r['seconds_per_1000']:.0f} | {f(r['electricity_usd'], 4)} | {f(r['cloud_gpu_usd'])} | {f(r['laptop_amortized_usd'])} |")
    out += ["", "| Hosted frontier API (one image per call, no labels) | list-price upper estimate per 1,000 ratings |", "| --- | --- |"]
    out += [f"| {m} | ${v:.2f} |" for m, v in cost["api_list_price_upper_estimates_usd_per_1000"].items()]
    s, a = cost["summary"], cost["assumptions"]
    out += ["", f"For one rubric per image on a rented GPU the APIs are {s['ratio_api_over_cloud_single_rubric'][0]:.0f} to "
            f"{s['ratio_api_over_cloud_single_rubric'][1]:.0f} times more expensive per rating. Not counted: engineering time and the one-time labeling of "
            f"about {a['labels_per_rubric']} images per rubric ({s['one_time_labeling_minutes_per_rubric'][0]:.0f} to "
            f"{s['one_time_labeling_minutes_per_rubric'][1]:.0f} minutes). The APIs need no labels and no setup. Assumptions: laptop draw "
            f"{a['laptop_power_watts'][0]} to {a['laptop_power_watts'][1]} W (not measured), electricity ${a['electricity_usd_per_kwh'][0]:.2f} to "
            f"${a['electricity_usd_per_kwh'][1]:.2f} per kWh, cloud GPU ${a['cloud_gpu_usd_per_hour'][0]} to ${a['cloud_gpu_usd_per_hour'][1]} per hour "
            "(AWS on-demand g4dn.xlarge and g6.xlarge, us-east-1, checked 2026-09-20) at the laptop's own speed, laptop price "
            f"${a['laptop_price_usd'][0]} to ${a['laptop_price_usd'][1]} amortized over {a['laptop_amortization_hours'][0]} to "
            f"{a['laptop_amortization_hours'][1]} hours. TypeSafe's Jev is not in the table: it does not accept images.", ""]
else:
    out += ["Pending.", ""]
(ROOT / "docs/paper/RESULTS_COMPARISONS.md").write_text("\n".join(out) + "\n")
print("wrote docs/paper/RESULTS_COMPARISONS.md,", len(out), "lines")

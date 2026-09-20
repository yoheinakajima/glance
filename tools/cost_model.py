"""Cost of 1,000 ratings: measured local seconds (lab/PACKING.json) turned into dollar RANGES under stated assumptions, next
to list-price upper estimates for frontier APIs (results/lab/cost_estimates.json). Nothing here is a measurement of
money; every dollar figure is arithmetic on the assumptions printed in the output.

uv run python tools/cost_model.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
pack = json.loads((ROOT / "lab/PACKING.json").read_text())
api = json.loads((ROOT / "results/lab/cost_estimates.json").read_text())["est_usd_per_1000_ratings"]

ASSUMPTIONS = {
    "laptop_power_watts": [15, 60],          # NOT measured (powermetrics needs root). Whole-machine draw of an Apple M5 laptop under sustained GPU load.
    "electricity_usd_per_kwh": [0.10, 0.40],
    "cloud_gpu_usd_per_hour": [0.526, 0.8048],  # AWS on-demand us-east-1, checked 2026-09-20: g4dn.xlarge (T4 16 GB) and g6.xlarge (L4 24 GB)
    "cloud_throughput_vs_laptop": [1.0, 1.0],   # conservative: same seconds per rating as the laptop. A CUDA GPU with batching should be faster; not measured.
    "laptop_price_usd": [2000, 4000],           # assumption, not a quote
    "laptop_amortization_hours": [8760, 26280],  # 3 years at 8 h/day ... 3 years around the clock
    "labels_per_rubric": 32, "seconds_per_label": [3, 10],
}
CONFIGS = {
    "ens4d, one rubric per image": pack["ens4d packed, 1 question"]["p50_ms_per_question"],
    "ens4d, 5 rubrics per image": pack["ens4d packed, 5 questions"]["p50_ms_per_question"],
    "ens4d, 25 rubrics per image": pack["ens4d packed, 25 questions"]["p50_ms_per_question"],
    "fast2 (no crop), 5 rubrics per image": pack["digits+digitsrev packed (no zoom), 5 questions"]["p50_ms_per_question"],
    "fast2 (no crop), 25 rubrics per image": pack["digits+digitsrev packed (no zoom), 25 questions"]["p50_ms_per_question"],
}
a = ASSUMPTIONS
rows = {}
for name, ms in CONFIGS.items():
    hours = ms * 1000 / 1000 / 3600  # 1,000 ratings x ms per rating -> seconds -> hours
    kwh = [hours * w / 1000 for w in a["laptop_power_watts"]]
    rows[name] = {
        "seconds_per_1000": ms, "hours_per_1000": hours,
        "electricity_usd": [kwh[0] * a["electricity_usd_per_kwh"][0], kwh[1] * a["electricity_usd_per_kwh"][1]],
        "cloud_gpu_usd": [hours * a["cloud_gpu_usd_per_hour"][0], hours * a["cloud_gpu_usd_per_hour"][1]],
        "laptop_amortized_usd": [hours * a["laptop_price_usd"][0] / a["laptop_amortization_hours"][1],
                                 hours * a["laptop_price_usd"][1] / a["laptop_amortization_hours"][0]],
    }
api_lo, api_hi = min(api.values()), max(api.values())
base = rows["ens4d, one rubric per image"]
summary = {
    "self_hosted_usd_per_1000_range": [min(r["electricity_usd"][0] for r in rows.values()), max(r["cloud_gpu_usd"][1] for r in rows.values())],
    "api_usd_per_1000_range": [api_lo, api_hi],
    "ratio_api_over_cloud_single_rubric": [api_lo / base["cloud_gpu_usd"][1], api_hi / base["cloud_gpu_usd"][0]],
    "one_time_labeling_minutes_per_rubric": [a["labels_per_rubric"] * s / 60 for s in a["seconds_per_label"]],
}
out = {"assumptions": a, "local": rows, "api_list_price_upper_estimates_usd_per_1000": api, "summary": summary}
(ROOT / "results/lab/cost_model.json").write_text(json.dumps(out, indent=2) + "\n")

f = lambda lo_hi, d=3: f"${lo_hi[0]:.{d}f} to ${lo_hi[1]:.{d}f}"  # noqa: E731
lines = ["# What 1,000 ratings cost: measured seconds, assumed prices", "",
         "Seconds are measured (`lab/PACKING.json`, cold start, Apple M5 laptop, Qwen3-VL-4B). Every dollar figure is arithmetic on the "
         "assumptions listed at the end; none is a measurement. API figures are list-price UPPER estimates from "
         "`glance baseline --estimate-only` (no call was made).", "",
         "| Configuration (Qwen3-VL-4B + Glance, self-hosted) | seconds per 1,000 ratings | electricity only | rented cloud GPU at laptop speed | laptop amortized |",
         "| --- | --- | --- | --- | --- |"]
for name, r in rows.items():
    lines.append(f"| {name} | {r['seconds_per_1000']:.0f} | {f(r['electricity_usd'], 4)} | {f(r['cloud_gpu_usd'])} | {f(r['laptop_amortized_usd'])} |")
lines += ["", "| Hosted frontier API, one image per call, zero labels | list-price upper estimate per 1,000 ratings |", "| --- | --- |"]
lines += [f"| {m} | ${v:.2f} |" for m, v in api.items()]
s = summary
lines += ["", f"Reading: self-hosted costs between {f(s['self_hosted_usd_per_1000_range'], 4)} per 1,000 ratings depending on how it is counted (electricity only ... "
          f"an on-demand cloud GPU that is no faster than the laptop), against {f(s['api_usd_per_1000_range'], 2)} for the APIs. For a single rubric per image on a "
          f"rented GPU the APIs are {s['ratio_api_over_cloud_single_rubric'][0]:.0f} to {s['ratio_api_over_cloud_single_rubric'][1]:.0f} times more expensive per rating. "
          f"Not included: engineering time, and the one-time labeling of about {a['labels_per_rubric']} images per rubric "
          f"({s['one_time_labeling_minutes_per_rubric'][0]:.0f} to {s['one_time_labeling_minutes_per_rubric'][1]:.0f} minutes of a person's time). "
          "The APIs need no labels and no setup; whether they are more or less ACCURATE on these rubrics is the head-to-head that is still open.", "",
          "Assumptions: " + "; ".join(f"{k} = {v}" for k, v in a.items()) + ". Cloud prices: AWS on-demand, us-east-1, g4dn.xlarge (T4) and g6.xlarge (L4), "
          "checked 2026-09-20 (instances.vantage.sh / cloudprice.net listings). Laptop power draw was not measured. TypeSafe's Jev is not listed: it does not accept images."]
(ROOT / "results/lab/cost_model.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

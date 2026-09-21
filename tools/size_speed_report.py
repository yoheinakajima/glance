"""Seconds and dollars per answer for Qwen3-VL at 2B, 4B and 8B, writing JSON against reading with Glance, all timed the same way
(one laptop, GPU otherwise idle): yes/no and pick-one on the full-size Commons photographs (`glance.lab.photo_timing`), ratings on
the 448 px lab images (write: `lab/GENBENCH_SINGLE*.json`; read: the one-pass read, `lab/runs/jsondigits_timing*.jsonl`). Cost is
those seconds at the rented-GPU price range of `results/lab/cost_model.json`, the same for every size. Answers the owner's question
of whether reading saves more as the model grows. Writes results/lab/size_speed.{json,md}.

uv run python tools/size_speed_report.py
"""
import json
import pathlib
import statistics

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIZES = {"2B": ("_2B", "_2b"), "4B": ("", ""), "8B": ("_8B", "_8b")}
GPU = json.loads((ROOT / "results/lab/cost_model.json").read_text())["assumptions"]["cloud_gpu_usd_per_hour"]
usd = lambda sec: [sec / 3600 * 1000 * GPU[0], sec / 3600 * 1000 * GPU[1]]  # noqa: E731  per 1,000 answers

out = {"gpu_usd_per_hour": GPU, "sizes": {}}
for size, (upper, lower) in SIZES.items():
    photo, single, timing = ROOT / f"lab/PHOTO_TIMING{upper}.json", ROOT / f"lab/GENBENCH_SINGLE{upper}.json", ROOT / f"lab/runs/jsondigits_timing{lower}.jsonl"
    if not (photo.exists() and single.exists() and timing.exists()):
        continue
    p, g = json.loads(photo.read_text()), json.loads(single.read_text())
    read_rating = statistics.median(r["latency_ms"] for r in read_jsonl(timing) if r["method_key"] == "jsondigits")
    rows = {"yes/no, full-size photograph": (p["yesno"]["write_p50_ms"], p["yesno"]["read_p50_ms"]),
            "pick-one, full-size photograph": (p["choice"]["write_p50_ms"], p["choice"]["read_p50_ms"]),
            "rating, 448 px image": (g["1 rating"]["write_p50_ms"], read_rating)}
    out["sizes"][size] = {k: {"write_s": w / 1000, "read_s": r / 1000, "write_over_read": w / r, "write_usd_per_1000": usd(w / 1000), "read_usd_per_1000": usd(r / 1000)}
                          for k, (w, r) in rows.items()}
(ROOT / "results/lab/size_speed.json").write_text(json.dumps(out, indent=1) + "\n")

md = ["# Writing JSON against reading with Glance, by model size (seconds per answer, idle GPU, one laptop)", "",
      f"Cost is the measured seconds at a rented-GPU price of ${GPU[0]:.2f} to ${GPU[1]:.2f} per hour, the same for every size (a larger model may need a dearer GPU; not modelled).", "",
      "| Question | size | written, s | read, s | written / read | read, US dollars per 1,000 |", "| --- | --- | --- | --- | --- | --- |"]
for task in next(iter(out["sizes"].values())):
    for size, e in out["sizes"].items():
        c = e[task]
        md.append(f"| {task} | {size} | {c['write_s']:.2f} | {c['read_s']:.2f} | {c['write_over_read']:.1f}x | {c['read_usd_per_1000'][0]:.2f} to {c['read_usd_per_1000'][1]:.2f} |")
(ROOT / "results/lab/size_speed.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

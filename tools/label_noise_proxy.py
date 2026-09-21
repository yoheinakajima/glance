"""A PROXY for label quality on the fresh photo sets (no human audit was done): items that EVERY system got wrong, the open
model and all hosted models alike, are the candidates for a wrong or ambiguous label. Their share bounds how much of the
"error" of any one system can be label noise. Hosted models answered the test half, so only those items are counted.
Writes results/lab/label_noise_proxy.{json,md}.

uv run python tools/label_noise_proxy.py
"""
import collections
import json
import pathlib

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETS = {"Commons": ("20260920T205633Z-99f822", ["", "-gpt", "-gemini", "-haiku", "-gptsmall", "-flashlite"], ("fresh_yesno", "fresh_choice")),
        "iNaturalist": ("20260920T232332Z-80efa7", ["", "-gpt", "-gemini", "-haiku", "-gptsmall", "-flashlite"], ("inat_yesno", "inat_choice"))}
out = {}
for name, (base, suffixes, suites) in SETS.items():
    hits = collections.defaultdict(dict)  # (suite, item) -> system -> correct
    for r in read_jsonl(ROOT / "runs" / base / "predictions.jsonl"):
        if r["backend"] == "vlm" and r["method"] in ("statement", "independent") and r["suite"] in suites:
            ok = ((r["raw"] >= 0.5) == bool(r["label_index"])) if r["type"] == "noul" else (int(np.argmax(r["raw"])) == int(r["label_index"]))
            hits[(r["suite"], r["item_id"])]["open model"] = bool(ok)
    for suffix in suffixes:
        f = ROOT / "runs" / (base + suffix) / "predictions.jsonl"
        for r in (read_jsonl(f) if f.exists() else []):
            if r["backend"] == "frontier" and r["suite"] in suites:
                hits[(r["suite"], r["item_id"])][str(r["model"]).removeprefix("frontier:")] = bool(r["correct"])
    for suite in suites:
        full = {k: v for k, v in hits.items() if k[0] == suite and len(v) >= 7}
        all_wrong = [k[1] for k, v in full.items() if not any(v.values())]
        all_right = sum(all(v.values()) for v in full.values())
        out[suite] = {"photo_set": name, "items_answered_by_all_7_systems": len(full), "every_system_wrong": len(all_wrong), "every_system_right": all_right,
                      "share_every_system_wrong": len(all_wrong) / max(1, len(full)), "items": all_wrong[:20]}
(ROOT / "results/lab/label_noise_proxy.json").write_text(json.dumps(out, indent=1))
md = ["# Label-quality proxy on the fresh photo sets (not a human audit)", "",
      "| Suite | photo set | items answered by all 7 systems | every system right | every system WRONG | share |", "| --- | --- | --- | --- | --- | --- |"]
md += [f"| {s} | {e['photo_set']} | {e['items_answered_by_all_7_systems']} | {e['every_system_right']} | {e['every_system_wrong']} | {e['share_every_system_wrong']:.1%} |" for s, e in out.items()]
md += ["", "An item that seven independent systems all answer 'wrongly' is more likely a wrong or ambiguous label than seven identical mistakes; this share is a rough floor on "
       "label noise and an explanation for part of the gap to 1.0. It is NOT a substitute for an audit by two human readers, which has not been done."]
(ROOT / "results/lab/label_noise_proxy.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

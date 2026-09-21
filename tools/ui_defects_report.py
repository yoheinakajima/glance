"""Two defects of the synthetic interface-screen test (E21), found when the hosted models' results arrived (`lab/NOTES.md` entry 49d),
and one diagnostic. Both defects are picked out by a rule of the GENERATOR, not by any model's answers:

1. Page type: on screens where a dialog is open, its backdrop hides the page, so the page type cannot be seen. Every system is
   perfect on the other screens.
2. "Is this goal already done": for the goal "Go to the next page of results" the after-screen (page 2 of 3) does not show that
   anything was done. Every system calls those screens not done.
Diagnostic (H58): how often each system reports the disabled main button on the screens that have it.

The registered all-item numbers stay in results/lab/ui_screens.json; this file adds the post-hoc numbers without the defective items,
labelled as such. Writes results/lab/ui_defects.{json,md}.

uv run python tools/ui_defects_report.py
"""
import collections
import json
import pathlib

import numpy as np

from glance.logging_utils import read_jsonl

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = "20260921T094322Z-62937c"
HOSTED = {"opus": "Claude Opus 5", "gpt": "GPT-5.6", "gemini": "Gemini 3.1 Pro", "haiku": "Claude Haiku 4.5", "gptsmall": "GPT-5.6 Luna", "flashlite": "Gemini 3.1 Flash-Lite"}
OPEN = "Qwen3-VL-4B + Glance"
NEXT_PAGE = "Go to the next page of results"
src = {r["screen_id"]: r for r in read_jsonl(ROOT / "glance/evals/manifests/ui_screens_source.jsonl")}

hits = collections.defaultdict(dict)  # (suite, item) -> {system: bool}
labels = {}
for r in read_jsonl(ROOT / "runs" / BASE / "predictions.jsonl"):
    if r["backend"] == "vlm" and r["method"] in ("statement", "independent") and r["suite"].startswith("ui_"):
        ok = ((r["raw"] >= 0.5) == bool(r["label_index"])) if r["type"] == "noul" else (int(np.argmax(r["raw"])) == int(r["label_index"]))
        hits[(r["suite"], r["item_id"])][OPEN] = bool(ok)
        labels[(r["suite"], r["item_id"])] = str(r["label"])
for suffix, name in HOSTED.items():
    path = ROOT / "runs" / f"{BASE}-{suffix}" / "predictions.jsonl"
    for r in (read_jsonl(path) if path.exists() else []):
        if r["backend"] == "frontier" and r["suite"].startswith("ui_"):
            hits[(r["suite"], r["item_id"])][name] = bool(r["correct"])
systems = [OPEN] + list(HOSTED.values())


def table(keys):
    shared = [k for k in keys if all(s in hits[k] for s in systems)]
    return {"n_shared": len(shared), "accuracy_same_items": {s: float(np.mean([hits[k][s] for k in shared])) for s in systems} if shared else {},
            "open_all_items": {"n": len(keys), "accuracy": float(np.mean([hits[k][OPEN] for k in keys]))}}


page = [k for k in hits if k[0] == "ui_page"]
covered = lambda k: bool(src[k[1]]["states"].get("dialog_open"))  # noqa: E731
done = [k for k in hits if k[0] == "ui_done"]
next_page = lambda k: src[k[1].split("__")[0]]["done"]["goal_text"] == NEXT_PAGE  # noqa: E731
disabled = [k for k in hits if k[0] == "ui_state" and k[1].endswith("__primary_disabled") and labels[k] == "True" and all(s in hits[k] for s in systems)]
shared_page = [k for k in page if all(s in hits[k] for s in systems)]
out = {"page_type": {"screens_with_a_dialog_over_the_page": int(sum(covered(k) for k in page)), "of_screens": len(page),
                     "wrong_for_every_system": int(sum(not any(hits[k].values()) for k in shared_page)),
                     "of_those_with_a_dialog": int(sum(covered(k) for k in shared_page if not any(hits[k].values()))),
                     "all_items": table(page), "without_covered_screens": table([k for k in page if not covered(k)])},
       "already_done": {"items_with_the_next_page_goal": int(sum(next_page(k) for k in done)), "of_items": len(done),
                        "all_items": table(done), "without_that_goal": table([k for k in done if not next_page(k)])},
       "disabled_button_detected": {"n_screens": len(disabled), "rate": {s: float(np.mean([hits[k][s] for k in disabled])) for s in systems}}}
(ROOT / "results/lab/ui_defects.json").write_text(json.dumps(out, indent=1) + "\n")
md = ["# Interface screens: two defects of the test, and the disabled-button diagnostic (post hoc; `lab/NOTES.md` entry 49d)", ""]
for key, title, a, b in (("page_type", "Page type", "all_items", "without_covered_screens"), ("already_done", "Is the goal already done", "all_items", "without_that_goal")):
    md += [f"## {title}", "", "| System | all items (registered) | without the defective items (post hoc) |", "| --- | --- | --- |"]
    md += [f"| {s} | {out[key][a]['accuracy_same_items'][s]:.3f} | {out[key][b]['accuracy_same_items'][s]:.3f} |" for s in systems]
    md += ["", f"Same items for every system: {out[key][a]['n_shared']} and {out[key][b]['n_shared']}. Open model on all its items: {out[key][a]['open_all_items']['accuracy']:.3f} (n={out[key][a]['open_all_items']['n']}) "
               f"and {out[key][b]['open_all_items']['accuracy']:.3f} (n={out[key][b]['open_all_items']['n']}).", ""]
md += ["## Disabled main button, reported on the screens that have it", "", "| System | detected |", "| --- | --- |"] + [f"| {s} | {v:.2f} |" for s, v in out["disabled_button_detected"]["rate"].items()]
(ROOT / "results/lab/ui_defects.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

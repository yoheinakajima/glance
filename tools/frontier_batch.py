"""One paste, several frontier baselines, no coming back: asks for each provider's API key ONCE (hidden input), checks
it with one tiny test call, shows every cost estimate, asks ONE yes/no, then runs everything one after another and
rebuilds the reports, the comparison matrix and the project page. Keys live only in this process's memory: never
printed, never written, never put in the environment. A provider whose key is left empty is skipped. The Mac is kept awake.

uv run python tools/frontier_batch.py --set cheap     # the cheapest current vision model of each provider, all three tests
uv run python tools/frontier_batch.py --set inat      # the three flagship models on the iNaturalist photos (done 2026-09-20)
"""
import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys

from glance.config import load_config
from glance.evals.baseline import add_baseline, check_model_key

COMMONS, INAT, LAB = "20260920T205633Z-99f822", "20260920T232332Z-80efa7", "20260920T165748Z-8ff72a"
REPORTS = [["uv", "run", "python", "tools/fresh_report.py", "--set", "commons", "--run", COMMONS, "--run", COMMONS + "-gpt", "--run", COMMONS + "-gemini"],
           ["uv", "run", "python", "tools/fresh_report.py", "--set", "inat", "--run", INAT, "--run", INAT + "-gpt", "--run", INAT + "-gemini"],
           ["uv", "run", "python", "tools/compare_frontier_lab.py", "--run", LAB, "--run", "20260920T165949Z-8ff72a", "--run", "20260920T170146Z-8ff72a"]]
REBUILD = [["uv", "run", "python", "tools/frontier_cost.py"], ["uv", "run", "python", "tools/make_matrix.py"], ["uv", "run", "python", "tools/make_results_zeroshot.py"],
           ["uv", "run", "python", "tools/make_site.py"]]
FLAGSHIP_AND_CHEAP = [  # provider, key hint, [(candidate model ids: the first whose test call works is used, run-folder suffix), ...]
    ("Anthropic", "console.anthropic.com -> API keys", [(["anthropic/claude-opus-5"], "opus"), (["anthropic/claude-haiku-4-5"], "haiku")]),
    ("OpenAI", "platform.openai.com -> API keys", [(["openai/gpt-5.6"], "gpt"), (["openai/gpt-5.6-luna", "openai/gpt-5-nano"], "gptsmall")]),
    ("Google, through OpenRouter", "openrouter.ai -> Keys", [(["openrouter/google/gemini-3.1-pro-preview"], "gemini"),
                                                             (["openrouter/google/gemini-3.1-flash-lite", "openrouter/google/gemini-3.1-flash-lite-preview"], "flashlite")]),
]


def newest_run_with(cfg, suite: str):
    """The newest finished eval run that holds local rows for `suite` (the harder insect-order test is run by the GPU queue)."""
    for run_dir in sorted(cfg.path("runs").iterdir(), reverse=True):
        f = run_dir / "predictions.jsonl"
        if f.exists() and "-" not in run_dir.name.split("Z-", 1)[-1] and f'"suite": "{suite}"' in f.read_text():
            return run_dir.name
    return None


# provider -> candidate model ids (the first one whose test call works is used), key hint, run-folder suffix
JOBS = {
    "orders": {"what": "The HARDER fresh test (insect orders on iNaturalist, lab/NOTES.md entry 47): three flagships and three cheapest models, about 315 calls each.",
               "providers": FLAGSHIP_AND_CHEAP, "bases": [("@inat_orders_choice", None)]},
    "probes": {"what": "Procedural probes (counting, spatial, size, stripes, text; lab/NOTES.md entry 50): three flagships and three cheapest models, about 450 calls each, small images.",
               "providers": FLAGSHIP_AND_CHEAP, "bases": [("@probe_count", None)]},
    "ui": {"what": "Synthetic UI screens (state, page type, which element to click, goal done, one-step reasoning; lab/NOTES.md entry 49): three flagships and three cheapest models.",
           "providers": FLAGSHIP_AND_CHEAP, "bases": [("@ui_click", None)]},
    "cheap": {"what": "The cheapest current vision model of each provider on all three tests: Commons photos (196 calls), iNaturalist photos (300), lab ratings (1,000).",
              "providers": [("Anthropic", "console.anthropic.com -> API keys", [(["anthropic/claude-haiku-4-5"], "haiku")]),
                            ("OpenAI", "platform.openai.com -> API keys", [(["openai/gpt-5.6-luna", "openai/gpt-5-nano"], "gptsmall")]),
                            ("Google, through OpenRouter", "openrouter.ai -> Keys", [(["openrouter/google/gemini-3.1-flash-lite", "openrouter/google/gemini-3.1-flash-lite-preview"], "flashlite")])],
              "bases": [(COMMONS, None), (INAT, None), (LAB, 200)]},  # (run to copy, items per suite; 200 = the lab images every other model saw)
    "inat": {"what": "The three flagship models on the 200 iNaturalist photos: 300 calls each.",
             "providers": [("Anthropic", "console.anthropic.com -> API keys", [(["anthropic/claude-opus-5"], "")]), ("OpenAI", "platform.openai.com -> API keys", [(["openai/gpt-5.6"], "gpt")]),
                           ("Google, through OpenRouter", "openrouter.ai -> Keys", [(["openrouter/google/gemini-3.1-pro-preview"], "gemini")])],
             "bases": [(INAT, None)]},
}


def stripped_copy(cfg, base: str, suffix: str):
    """A run folder holds ONE frontier model: copy the base run and drop its frontier rows, so the same items get a new model."""
    src, dst = cfg.path("runs") / base, cfg.path("runs") / (f"{base}-{suffix}" if suffix else base)
    if not dst.exists():
        shutil.copytree(src, dst)
        for name in ("predictions.jsonl", "errors.jsonl"):
            f = dst / name
            if f.exists():
                kept = [line for line in f.read_text().splitlines() if line.strip() and json.loads(line).get("backend") != "frontier"]
                f.write_text("".join(line + "\n" for line in kept))
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=sorted(JOBS), default="cheap")
    parser.add_argument("--yes", action="store_true", help="do not ask the single confirmation")
    args = parser.parse_args()
    job, cfg = JOBS[args.set], load_config()
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    print(f"\n{job['what']}\nOne key per provider. Input is hidden and kept in memory only. Press Enter with nothing to skip a provider.\n")
    bases = []
    for base, n in job["bases"]:
        if base.startswith("@"):  # resolved at run time: the newest local run of that suite
            found = newest_run_with(cfg, base[1:])
            if not found:
                print(f"No finished local run of `{base[1:]}` yet: the GPU queue has not produced it. Try again later.")
                return 1
            base = found
        bases.append((base, n))
    ready = []
    for provider, hint, models in job["providers"]:
        key = getpass.getpass(f"{provider} API key ({hint}): ").strip()
        if not key:
            print("  skipped")
            continue
        for candidates, suffix in models:
            model = next((m for m in candidates if check_model_key(cfg, m, key, out=sys.stdout)), None)
            if model is None:
                print(f"  {provider}: no candidate model answered the test call ({', '.join(candidates)}); skipped")
                continue
            ready += [(stripped_copy(cfg, base, suffix), model, key, n) for base, n in bases]
    if not ready:
        print("\nNothing to run.")
        return 1
    print("\n" + "=" * 78)
    for run_dir, model, _, n in ready:
        add_baseline(cfg, run_dir, model, None, baseline_n=n, estimate_only=True, out=sys.stdout)
    print("=" * 78)
    if not args.yes and input(f"\nRun all {len(ready)} now, one after another? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Stopped. Nothing was spent beyond the test calls.")
        return 1
    done = []
    for run_dir, model, key, n in ready:
        print(f"\n##### {model} -> run {run_dir.name}")
        try:
            result = add_baseline(cfg, run_dir, model, key, baseline_n=n, confirmed=True, out=sys.stdout)
            done.append((run_dir.name, model, "finished" if result is not None else "stopped early"))
        except Exception as exc:  # keep going: one provider's outage must not cost the other runs
            done.append((run_dir.name, model, f"failed: {type(exc).__name__}"))
    del ready, key
    print("\n" + "=" * 78)
    for run_id, model, state in done:
        print(f"  {model} on {run_id}: {state}")
    finished = [run_id for run_id, _, state in done if state == "finished"]
    if args.set in ("probes", "ui") and finished:
        subprocess.run(["uv", "run", "python", "tools/suite_report.py", "--name", {"probes": "probes", "ui": "ui_screens"}[args.set], "--prefix", {"probes": "probe_", "ui": "ui_"}[args.set],
                        "--run", bases[0][0]] + [a for r in finished for a in ("--run", r)], check=False, stdout=subprocess.DEVNULL)
    if args.set == "orders" and finished:
        subprocess.run(["uv", "run", "python", "tools/fresh_report.py", "--set", "orders", "--run", bases[0][0]] + [a for r in finished for a in ("--run", r)], check=False, stdout=subprocess.DEVNULL)
    for cmd in REPORTS:
        base = cmd[cmd.index("--run") + 1]
        extra = [a for run_id in finished if run_id.startswith(base[:16]) and run_id not in cmd for a in ("--run", run_id)]
        subprocess.run(cmd + extra, check=False, stdout=subprocess.DEVNULL)
    for cmd in REBUILD:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
    print("\nDone. Reports, the comparison matrix and the project page are rebuilt. You can close this window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

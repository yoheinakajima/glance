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
# provider -> candidate model ids (the first one whose test call works is used), key hint, run-folder suffix
JOBS = {
    "cheap": {"what": "The cheapest current vision model of each provider on all three tests: Commons photos (196 calls), iNaturalist photos (300), lab ratings (1,000).",
              "providers": [("Anthropic", ["anthropic/claude-haiku-4-5"], "console.anthropic.com -> API keys", "haiku"),
                            ("OpenAI", ["openai/gpt-5.6-luna", "openai/gpt-5-nano"], "platform.openai.com -> API keys", "gptsmall"),
                            ("Google, through OpenRouter", ["openrouter/google/gemini-3.1-flash-lite", "openrouter/google/gemini-3.1-flash-lite-preview"], "openrouter.ai -> Keys", "flashlite")],
              "bases": [(COMMONS, None), (INAT, None), (LAB, 200)]},  # (run to copy, items per suite; 200 = the lab images every other model saw)
    "inat": {"what": "The three flagship models on the 200 iNaturalist photos: 300 calls each.",
             "providers": [("Anthropic", ["anthropic/claude-opus-5"], "console.anthropic.com -> API keys", ""), ("OpenAI", ["openai/gpt-5.6"], "platform.openai.com -> API keys", "gpt"),
                           ("Google, through OpenRouter", ["openrouter/google/gemini-3.1-pro-preview"], "openrouter.ai -> Keys", "gemini")],
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
    ready = []
    for provider, candidates, hint, suffix in job["providers"]:
        key = getpass.getpass(f"{provider} API key ({hint}): ").strip()
        if not key:
            print("  skipped")
            continue
        model = next((m for m in candidates if check_model_key(cfg, m, key, out=sys.stdout)), None)
        if model is None:
            print(f"  {provider}: no candidate model answered the test call ({', '.join(candidates)}); skipped")
            continue
        ready += [(stripped_copy(cfg, base, suffix), model, key, n) for base, n in job["bases"]]
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

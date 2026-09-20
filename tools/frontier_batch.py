"""One paste, several frontier baselines, no coming back: asks for each provider's API key up front (hidden input),
checks every key with one tiny test call, shows every cost estimate, asks ONE yes/no, then runs the baselines one after
another and merges the results. Keys live only in this process's memory: never printed, never written, never put in the
environment. A model whose key is left empty is skipped. The Mac is kept awake while it runs.

uv run python tools/frontier_batch.py                # the current job set (see JOBS)
uv run python tools/frontier_batch.py --yes          # skip the single confirmation (estimates are still printed)
"""
import argparse
import getpass
import os
import subprocess
import sys

from glance.config import load_config
from glance.evals.baseline import add_baseline, check_model_key

# job set -> [(run id, LiteLLM model id, where the key comes from)], then the commands that merge the results
JOBS = {
    "inat": {
        "what": "200 fresh iNaturalist photos (lab/NOTES.md entries 35, 35b): 300 calls per model",
        "runs": [("20260920T232332Z-80efa7", "anthropic/claude-opus-5", "console.anthropic.com -> API keys"),
                 ("20260920T232332Z-80efa7-gpt", "openai/gpt-5.6", "platform.openai.com -> API keys"),
                 ("20260920T232332Z-80efa7-gemini", "openrouter/google/gemini-3.1-pro-preview", "openrouter.ai -> Keys")],
        "after": [["uv", "run", "python", "tools/fresh_report.py", "--set", "inat"], ["uv", "run", "python", "tools/make_results_zeroshot.py"]],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=sorted(JOBS), default="inat")
    parser.add_argument("--yes", action="store_true", help="do not ask the single confirmation")
    args = parser.parse_args()
    job, cfg = JOBS[args.set], load_config()
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    print(f"\n{job['what']}\nPaste one key per model. Input is hidden and kept in memory only. Press Enter with nothing to skip a model.\n")
    ready = []
    for run_id, model, hint in job["runs"]:
        if not (cfg.path("runs") / run_id / "predictions.jsonl").exists():
            print(f"  {model}: run {run_id} not found, skipped")
            continue
        key = getpass.getpass(f"API key for {model} ({hint}): ").strip()
        if not key:
            print("  skipped")
            continue
        if check_model_key(cfg, model, key, out=sys.stdout):
            ready.append((run_id, model, key))
        else:
            print(f"  {model}: skipped (the test call failed)")
    if not ready:
        print("\nNothing to run.")
        return 1
    print("\n" + "=" * 78)
    for run_id, model, _ in ready:
        add_baseline(cfg, cfg.path("runs") / run_id, model, None, estimate_only=True, out=sys.stdout)
    print("=" * 78)
    if not args.yes and input(f"\nRun all {len(ready)} now, one after another? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Stopped. Nothing was spent beyond the test calls.")
        return 1
    done = []
    for run_id, model, key in ready:
        print(f"\n##### {model} -> run {run_id}")
        try:
            result = add_baseline(cfg, cfg.path("runs") / run_id, model, key, confirmed=True, out=sys.stdout)
            done.append((run_id, model, "finished" if result is not None else "stopped early"))
        except Exception as exc:  # keep going: one provider's outage must not cost the other runs
            done.append((run_id, model, f"failed: {type(exc).__name__}"))
    del ready, key
    print("\n" + "=" * 78)
    for run_id, model, state in done:
        print(f"  {model}: {state}")
    runs = [a for run_id, _, state in done if state == "finished" for a in ("--run", run_id)]
    first = job["runs"][0][0]
    if "--run" in runs and first not in runs:  # the report takes its local rows from the first run
        runs = ["--run", first] + runs
    for cmd in job["after"]:
        subprocess.run(cmd + (runs if "fresh_report.py" in " ".join(cmd) else []), check=False)
    print("\nDone. Results: results/lab/fresh_inat.md and docs/paper/RESULTS_ZEROSHOT.md. You can close this window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

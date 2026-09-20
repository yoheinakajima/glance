#!/bin/bash
# E15 (lab/NOTES.md entries 38, 38b): one model size of the same open family, zero-shot, identical settings.
# usage: tools/run_scaling.sh 2b|8b      (8b: run only while no other model process is using the GPU)
set -u
SIZE="$1"; CFG="configs/scaling_qwen3vl_${SIZE}.yaml"; OUT="lab/runs/scaling_${SIZE}.jsonl"; M="digits,zoom_digits,digitsrev,zoom_digitsrev"
cd "$(dirname "$0")/.."
uv run python -m glance.lab.collect --config "$CFG" --bench ladders --split test --limit 200 --out "$OUT" --methods "$M" --prefix-cache || exit 1
uv run python -m glance.lab.collect --config "$CFG" --bench ladders --split calibration --limit 100 --out "$OUT" --methods "$M" --prefix-cache || exit 1
uv run python -m glance.lab.collect --config "$CFG" --bench semantic --split test --limit 100 --out "lab/runs/scaling_${SIZE}_semantic.jsonl" --methods "$M" --prefix-cache || exit 1
uv run glance --config "$CFG" eval --suite fresh_choice --suite fresh_yesno --suite inat_choice --suite inat_yesno --model vlm --prefix-cache > "lab/runs/scaling_${SIZE}_eval.out" 2>&1 || exit 1

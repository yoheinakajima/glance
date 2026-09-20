# State of the project and what comes next (hand-over, 2026-09-20 14:30 local)

Read this first if you are picking the work up (a person, or an assistant whose context was reset). It says what is
done, what is RUNNING RIGHT NOW on this machine, what each finished job needs next (exact commands), and what is
planned. Detail lives in `lab/NOTES.md` (entries 1 to 32, every registration and result), claims and caveats in
`docs/CLAIMS.md`, the reviewer packet in `docs/BRIEFING.md`. Branch `score-lab`; `main` still holds v0 only.

## 1. What this project is now

Glance is how you ask a frozen vision-language model (Qwen3-VL-4B, local) for typed decisions about images: yes/no
(`noul`), pick-one (`choice`), and ratings on a described rubric (`score`), read from logits, no generated text. It is a
readout-and-calibration recipe with a runtime (`glance decide | score | fit | eval | baseline | serve`, Python class
`glance.Glance`), not a model. Owner's direction: see how far AI-only work goes; no hand labeling; document everything
for a paper; publish misses; never call it a model or "Jev for vision"; `glance fit` is the product verb.

## 2. Results that are final (all committed; numbers in `docs/CLAIMS.md`)

- Out of the box, zero setup: yes/no POPE 0.880, GQA 0.732; pick-one 0.892 / 0.919; Opus 5 ahead by 3.4 points
  [1.0, 5.8] on v0's five suites. On 131 photos taken AFTER the models' release (labels = Wikimedia Commons "depicts"
  tags, none made by us): yes/no 0.931, pick-one 0.885 (entry 30b).
- Ratings, five synthetic lab scales: 0.558 with nothing, 0.697 with unlabeled images (`glance fit --unlabeled`), 0.856
  with 32 labels, 0.867 with 500 (`ens4d`); `fast2` 0.833; adaptive compute 0.863 at 2.57 passes (entry 27b, not yet in
  the harness). Opus 5 zero-shot on the same 1,000 images: 0.550 (entry 31). Classical features: 0.979 with 500 labels,
  0.752 with 32 (entry 25). SigLIP2: 0.330 -> 0.588 with the same map.
- KADID-10k, fixed method, scored once: 0.527 exact, 0.880 within one, Spearman with human DMOS 0.763 (ceiling 0.840);
  registered targets missed (entry 28). Universal map across rubrics fails (0.350 vs raw 0.328; with bias removal 0.433;
  per-rubric 0.527).
- Speed and cost: reading vs the same model writing JSON is 2.4x / 3.5x / 6.1x faster for 1 / 5 / 25 questions
  (entry 27c); one rating 1,089 ms, 584 ms at 5 rubrics, 341 ms at 25; cost model `results/lab/cost_model.md`
  ($0.02 to $0.24 per 1,000 ratings self-hosted vs $3.74 to $18.70 list price for frontier APIs).
- Errata on record: latency (16), small-fit overconfidence (18), flawed ECE criterion (15b), notebook timestamps,
  several narrowly missed registered targets (H19, H21), H20 failed, KADID targets failed.

## 3. RUNNING RIGHT NOW (do not start a second copy)

GPU queue runner: `$TMPDIR/glance/gpuq.sh` reads `$TMPDIR/glance/gpuq.txt` one line at a time ("label ::: command"), log
in `$TMPDIR/glance/queue_big.log`. Edit the txt file to add or reorder jobs (write a temp file, then `mv`); create
`$TMPDIR/glance/gpuq.stop` to stop after the current job. If the machine rebooted, the queue is gone: restart the runner
and re-add the unfinished lines below (every collector job is resumable: finished rows are skipped).

| Order | Job | Output | Then do |
| --- | --- | --- | --- |
| running | `lab-hidden`: four `ens4d` readouts + final hidden states on all 5,000 lab items | `lab/runs/lab_hidden.jsonl`, `lab/hidden/ladders/*.npz` | E1: `uv run python tools/readout_ladder.py --bench ladders --logits lab/runs/lab_hidden.jsonl --out lab/READOUT_LADDER --dev` first (calibration split only), then without `--dev` ONCE; write entry with verdicts on H12 to H14 (entry 20). Ships into the harness only if it clearly wins (owner said yes to shipping). |
| next | `kadid-ref`: reference-anchored readouts, first 200 items of each KADID distortion | `lab/runs/kadid_ref.jsonl` | E9 (entry 29): compare `fast2_ref = ref_digits + ref_digitsrev` with no-reference `fast2` and `ens4d` on the SAME items (write a small script using `glance.lab.analyze.combine_rows` and `bench_report.evaluate`); verdicts on H23, H24. NOTE: the `ref_` collector path was never smoke-tested on the GPU; check the first rows (two images, `ref` first; finite logits) before trusting it. |
| then | `letters-lab`: letter, letter4, poles readouts on the first 600 items per lab scale | `lab/runs/lab_letters.jsonl` | E5 (entry 24): `uv run python tools/readout_baselines.py`, then `uv run python tools/make_results_comparisons.py`; verdicts on H15 to H18. |
| then | `smol-lab`: SmolVLM2-2.2B, first 400 items per lab scale, five readouts + hidden states | `lab/runs/smolvlm2_lab.jsonl` | E3 (entries 20, 22, 22b): `uv run python -m glance.lab.bench_report --bench ladders --in lab/runs/smolvlm2_lab.jsonl --out lab/SMOLVLM2_REPORT`; check the registered order (shipped v0 < calibrated v0 <= digits < ens4d). |
| then | `distort25-ens4d`, `kadid-independent`, `distort25-independent`, `kadid-hidden-backfill` | `lab/runs/distort25.jsonl`, `lab/runs/kadid.jsonl` | `bench_report` for distort25 and again for kadid (adds the v0 baseline rows: H7), `tools/loro.py --bench distort25 --in lab/runs/distort25.jsonl`, `tools/make_results_generalization.py`; freeze gz snapshots into `lab/data/`. |

Owner's terminals: frontier baselines. Opus 5 finished (run `20260920T165748Z-8ff72a`, 1,000 picks). GPT-5.6
(`20260920T165949Z-8ff72a`) and Gemini (`20260920T170146Z-8ff72a`) had no rows at 14:25. When they finish:
`uv run python tools/compare_frontier_lab.py --run 20260920T165748Z-8ff72a --run 20260920T165949Z-8ff72a --run 20260920T170146Z-8ff72a`.
Optional, clean photos: `uv run glance baseline --run 20260920T205633Z-99f822 --model anthropic/claude-opus-5`, then
`uv run python tools/fresh_report.py --run 20260920T205633Z-99f822`. New frontier rows now carry `output_tokens` and
`cost_usd` per call (the finished Opus run predates that and has input-token counts only, in `logs/calls/`). The
assistant cannot handle API keys; `glance baseline --env-file <path>` reads one variable from a file the owner names.

Cheaper-model (Sonnet) agents still working when this was written: (1) `glance/lab/semantic.py` creative-QA benchmark:
five scales built (cut-off, occlusion, tilt, caption legibility, watermark); fixing two defects I found (black corner
wedges give tilt away; captions run off the frame). Its files are UNCOMMITTED (`lab/manifests_semantic/`,
`lab/sheets_semantic/`, `glance/lab/semantic.py`, `tests/test_lab_semantic.py`): review the two new sheets, run its tests,
commit, then queue `--bench semantic` with the four `ens4d` readouts + `independent` (E4, entry 20). (2) adapters for two
runnable external systems, `AlexWortega/openjev` v2 and `zhangzicheng/q-sit-mini` (`glance/lab/external_systems.py`,
`tools/external_report.py`): smoke tests only; the lead queues the full runs on `--bench ladders --limit 600`.

## 4. Registered but not started

- **E11 (entry 32, owner's request): the same Qwen3-VL-4B WRITING structured JSON as the baseline for accuracy, speed
  and cost**, on the lab scales (the 200 test images per scale of the frontier table), `fresh_yesno` / `fresh_choice`,
  and the KADID subset. Build on `glance/lab/gen_bench.py` (`Bench.write`, `json_prompt`, `parse_json`, `valid`): write
  `glance/lab/gen_accuracy.py` that stores the parsed answer per item next to the ground truth, then a report with
  paired bootstrap against the read at 0 labels / unlabeled / 32 labels. Needs GPU time: add to the queue.
- E8 (entry 27): `tournament` pick-one (chunks of 26 option letters + final round) to replace 37 to 101 forward passes.
- E7 into the harness: adaptive `fast2` -> `ens4d` (threshold 0.55 on the lab scales); needs the scorer to apply the
  first-stage map before deciding on the second stage.
- E9 secondary: `ens4d_ref` with magnified crops of both images (collector raises NotImplementedError for now).
- Stage C (extra magnified crops, only the exposure scale left) and a second selection with crops: low priority.
- VisualQuality-R1-7B (Apache-2.0, 16.6 GB, trained on KADID itself) as an external system: only when the GPU is free.
- Brier score in `glance fit` and the benchmark reports (promised in entry 27, not done).

## 5. Decisions waiting for the owner

Merge `score-lab` into `main` once the running results are in; create a GitHub remote (none exists; publishing needs
their explicit go-ahead; PyPI name `glance` is taken by OpenStack, `glance-vlm` is free); whether to spend on the GPT
and Gemini baselines and on frontier models for the fresh photos.

## 6. Rules this project runs by (so they survive a hand-over)

Register an experiment in its own commit BEFORE any code or data (use the real clock: `date +%H:%M`); choose on the
calibration split, score the test split once; report misses as misses; Apache-2.0 / MIT weights only, dataset licenses
verified and recorded in `DATASETS.md`; KADID-10k is evaluation only, nothing from it committed except ids, levels and
logits; frontier picks are never stored (only correct / not, token counts and cost); no hand labeling by the owner;
cheaper models (Sonnet agents) for easier parallel work, reviewed before commit; do not run GPU jobs outside the queue
while a timing benchmark is running; commits end with the Co-Authored-By line; pronouns they/them for the owner.

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
  the harness). Frontier models zero-shot on the same 1,000 images: Opus 5 0.550, GPT-5.6 0.597, Gemini 3.1 Pro 0.650 (entries 31, 33, 33c): Gemini beats the zero-label readout (0.570) by 8 points; the unlabeled-images fit (0.702) is ahead of all three. Classical features: 0.979 with 500 labels,
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

**Progress 00:58 on 2026-09-21 (overnight; newest first).** DONE tonight: insect-order test with all six hosted models (entry
47c, on the page), E19 SmolVLM2 on the photographs (entry 46b: level on everyday photos, trails on nature photos; on the page and
in the draft), any-model option checked on the GPU (entry 46c; "experimental" replaced by what was checked), E21 UI screens built,
reviewed, registered and queued (entry 49b; `ui-eval` on lane A), E22 probes built and queued (entry 50b), a batch-tool defect
fixed before it reached the page (entry 47d), `tools/suite_report.py --by SUITE=KEY` for the per-count / per-ratio verdicts,
`beyond()` in `tools/make_site.py` (a data-driven Table 3c for probes and UI screens; NOT yet placed in the page body: place it
under section 2 once `results/lab/probes.json` or `ui_screens.json` exists, with a sentence per registered prediction).
STILL TO DO as jobs finish: `smol-json` -> `tools/scaling_report.py`; `written-inat` -> written rows for the iNaturalist
table; `jsondigits-semantic` + `jsondigits-kadid` -> `tools/jsondigits_hard_report.py` (E17 DECISION); `probes-eval` ->
`uv run python tools/suite_report.py --name probes --prefix probe_ --run <run> --by probe_count=count --by probe_count_color=label --by probe_largest=ratio`
(H53 to H55); `ui-eval` -> `uv run python tools/suite_report.py --name ui_screens --prefix ui_ --run <run>` (H50; H51 / H52 need
the owner's hosted paste `uv run python tools/frontier_batch.py --set ui`); `qsit-lab` + `openjev-lab` ->
`tools/external_report.py` (H31 / H32).

**OVERNIGHT PLAN, written 23:52 on 2026-09-20. The owner is asleep and may PUBLISH in the morning. READ THIS BLOCK FIRST.**
Goal by morning: every blank on the page filled from runs that have finished, the paper draft consistent with the page, `main`
fast-forwarded and pushed to the PRIVATE repository `github.com/yoheinakajima/glance` (created at the owner's request; going
public and turning on Pages are the OWNER's steps, `docs/PUBLISH.md`). Never make the repository public, never enable Pages.
After EVERY result: notebook entry with verdicts (real clock) -> regenerate (`tools/make_matrix.py`, `tools/scaling_report.py`,
`tools/other_models_report.py`, `tools/make_results_zeroshot.py`, `tools/make_site.py`) -> `tools/verify_docs_numbers.py` ->
commit -> `git fetch . score-lab:main && git push origin main score-lab` -> republish the preview artifact (same file path
`site/page.html`).
Wake-ups armed (background tasks): (1) `smol-fresh` + `generic-smoke` + `smol-json` done -> `tools/other_models_report.py`,
`tools/scaling_report.py`, verdict H47 (entry 46), read `$TMPDIR/glance/generic_smoke.out|.err` and only then drop
"experimental" wording; (2) `written-inat` + `jsondigits-semantic` + `jsondigits-kadid` done -> add the written rows to the
page's photo table (from `lab/runs/gen_accuracy_inat.jsonl`), `tools/jsondigits_hard_report.py` (prints H42 / H43 and the
registered DECISION about making the one-pass read the harness default; if YES, wire `jsondigits` into `glance/rating.py` /
`scorer.py` as `score_method` for zero-shot and keep `ens4d` for fits, with tests); (3) the owner's six hosted runs on the
insect-order test complete (run `20260921T061001Z-da85d6` + suffixes opus, haiku, gpt, gptsmall, gemini, flashlite) ->
`tools/fresh_report.py --set orders --run <base> --run <each copy>`, verdict H49 (entry 47), add a short subsection to the page;
(4) `qsit-lab` + `openjev-lab` done -> `tools/external_report.py` (H31 / H32, entry 37). Two assistant-model agents are building
E21 (synthetic UI screens, entry 49: `glance/lab/ui_screens.py`, `glance/evals/suites/ui_screens.py`) and E22 (procedural probes,
entry 50: `glance/lab/probes.py`, `glance/evals/suites/probes.py`); they must NOT edit `suites/__init__.py`: when each reports,
review its contact sheets, register its `MODULES` in `glance/evals/suites/__init__.py` (and `tests/test_m3_evals.py` expects the
suite-name sets to match), commit, queue `uv run glance eval --suite <its suites> --model vlm --prefix-cache` on lane A, add a
job set to `tools/frontier_batch.py` for the owner's morning paste. These two do NOT block publication.
Morning hand-over for the owner: what finished, what changed on the page, the exact publish steps, the optional pastes.

**Update 22:18 (READ THIS BLOCK FIRST; older blocks below are history).**
Results since 17:00, all in `lab/NOTES.md`: 32b (the same model WRITING beats the four-pass read zero-shot on ratings, 0.672 vs
0.570; identical on yes/no and pick-one), 42b (one-pass read at the JSON answer position `jsondigits`: 0.669, 0.758 with 16
unlabeled images), 44 (SmolVLM2: fitted recipe replicates, zero-shot weaker), 35c (frontier on iNaturalist: level), 45b (the
providers' CHEAPEST models, owner's idea: level on yes/no and pick-one except Haiku on nature photos; every cheap model beats
its flagship on ratings; Gemini 3.1 Flash-Lite 0.763 leads zero-shot; hosted cost $0.12 to $1.89 per 1,000, so the open
model is about 5x cheaper on yes/no and NOT cheaper on ratings), 41/41b (positioning: same inference object as Simple Jev;
Glance = calibration and measurement harness). E15 interim: Qwen3-VL-8B four-pass zero-shot 0.537 (NOT above 4B 0.570); its
one-pass read, photo evals and the verdict entry are PENDING: when `DONE timing-single-idle` appears run
`uv run python tools/scaling_report.py`, `tools/other_models_report.py`, `tools/make_matrix.py` (fills the pending timing
cells from `lab/GENBENCH_SINGLE.json` and `lab/runs/jsondigits_timing.jsonl`), `tools/make_results_zeroshot.py`,
`tools/make_site.py`, write the E15 entry with H33 to H36, republish the page.
Lane A after the 8B window: `smol-fresh` (E19, entry 46: SmolVLM2 yes/no and pick-one; then `tools/other_models_report.py`),
`generic-smoke` (three `glance --model-id ... ask` calls on SmolVLM2; read `$TMPDIR/glance/generic_smoke.out|.err`; if it works
drop "experimental" caveats only after a real look), `written-inat` (then move the matrix's yes/no and pick-one columns to the
iNaturalist set), `jsondigits-semantic` + `jsondigits-kadid` (E17, entry 43: decision rule for making `jsondigits` the harness
zero-shot default), then kadid-ref, letters-lab, distort25, baselines. Lane B resumes after the window: qsit-lab, openjev-lab
(E13), semantic-ens4d.
New since 18:00: comparison matrix (`tools/make_matrix.py` -> `results/lab/matrix.*`), project page (`tools/make_site.py`,
`tools/site_charts.py`, `docs/SITE_PLAN.md`; private preview https://claude.ai/artifact/HjAYTcWLY348UVL4rA447s; owner picked
bars-by-question-type as the headline figure after the matrix, then cost-vs-speed and accuracy-vs-cost), README rewritten
around use + `AGENTS.md` + `glance ask` command, GitHub Pages workflow + `site/CNAME` (glance.yohei.me; nothing is pushed,
there is no remote; publishing is the owner's call), any-model backend (`glance/backends/generic_hf.py`, `--model-id`,
`Glance(model_id=...)`; plumbing tested, GPU smoke test queued), one-paste frontier batches (`tools/frontier_batch.py --set
cheap|inat`). Next on the owner-approved list: PyPI package `glance-vlm`, then a Hugging Face Space demo (needs the owner's
HF login for the push).

**Update 16:54.** `inat-eval` done (entry 35b: pick-one 0.940, yes/no 0.945; run `20260920T232332Z-80efa7`, copies `-gpt` and
`-gemini` made for the owner's frontier calls; when they finish:
`uv run python tools/fresh_report.py --set inat --run 20260920T232332Z-80efa7 --run 20260920T232332Z-80efa7-gpt --run 20260920T232332Z-80efa7-gemini`,
then `tools/make_results_zeroshot.py`, verdict on the last part of H30). E14 done and FAILED (entry 39b). E15: weights downloaded;
`scaling-2b` and `scaling-8b-alone` are in lane A after `semantic-zeroshot-test`; lane B has a `hold-for-8b` step after `smol-lab`
(flag files `$TMPDIR/glance/laneB.holding` and `scaling8b.done`; if the 8B job dies, `touch $TMPDIR/glance/scaling8b.done` to release
lane B); afterwards `uv run python tools/scaling_report.py` (verdicts H33 to H35; H36 from the four `lab/runs/scaling_*_eval.out` run
ids with `tools/fresh_report.py`).

**Update 16:32. READ FIRST: the owner re-framed the project (STATUS D44, D45; `lab/NOTES.md` entry 38).** Headline =
what the frozen OPEN model does ZERO-SHOT as a general image decision engine: `docs/paper/RESULTS_ZEROSHOT.md`
(`uv run python tools/make_results_zeroshot.py` after every job below). Labels are a caveat. Lane A order now:
`inat-eval` (running; then `uv run python tools/fresh_report.py --set inat --run <run id printed in $TMPDIR/glance/inat_eval.out>`
and give the owner the one-paste `glance baseline --run <id> --model ...` commands for the three frontier models, as for the
Commons set) -> `null-prior` (E14, entry 39; then `uv run python tools/null_prior.py`) -> `gen-accuracy` (E11) ->
`semantic-zeroshot-test` (E4 test split, four `ens4d` members; zero-shot numbers need no fit) -> `kadid-ref` -> `letters-lab`
-> distort25 / baselines. Lane B: `smol-lab` (running) -> `qsit-lab` -> `openjev-lab` (E13, adapters committed, never run on
the GPU: check the first rows and memory) -> `semantic-ens4d` (full). E15 scaling (entry 38): weights for Qwen3-VL 2B and 8B
are downloading into `.cache/hf` (cap raised to 80 GB by the owner); run with
`uv run python -m glance.lab.collect --config configs/scaling_qwen3vl_2b.yaml --bench ladders --split test --out lab/runs/scaling_2b.jsonl --methods "digits,zoom_digits,digitsrev,zoom_digitsrev" --prefix-cache`
(2B may share the GPU; 8B ONLY when both lanes are idle), plus `uv run glance --config configs/scaling_qwen3vl_8b.yaml eval --suite fresh_choice --suite fresh_yesno --suite inat_choice --suite inat_yesno --model vlm --prefix-cache`;
the analysis tool (`results/lab/scaling.{json,md}`, H33 to H36) is NOT written yet. E1 is done (entry 40: hidden-state
readout 0.965 from one pass; label-hungry; "if you have examples" section; harness integration not started).

**Update 15:56 (newest first; the table below is still the order of lane A).** Lane A was stopped from the app and
resumed at the owner's word (15:27); `lab-hidden` continues (17.5k of 20k rows at 15:55). Lane B
(`$TMPDIR/glance/gpuq_b.sh` + `gpuq_b.txt`, log lines tagged `[gpuqB]`): `smol-lab` running, then `semantic-ens4d`
(E4, five rubrics; entries 36 and 36b: caption levels re-tuned by eye before any model output, two rubrics infeasible
on portrait photos). Two cheaper-model agents are working, files UNCOMMITTED until reviewed: (1) E12 (entry 35), a
second uncontaminated photo set from iNaturalist with community-verified labels (`tools/fetch_fresh_inat.py`,
`glance/evals/suites/fresh_inat.py`, `tests/test_fresh_inat.py`; then run
`uv run glance eval --suite inat_choice --suite inat_yesno --model vlm --model siglip --prefix-cache` on lane B and give
the owner the one-paste frontier commands); (2) E13 (entry 37), adapters for openjev v2 and q-sit-mini
(`glance/lab/external_systems.py`, `tools/external_collect.py`, `tools/external_report.py`); the 4B openjev model must
not be loaded while two lanes are busy. Gemini on the lab scales finished and is merged (entry 33c); it was at 984 of 1,000 answers at 15:56.

GPU queue runner: `$TMPDIR/glance/gpuq.sh` reads `$TMPDIR/glance/gpuq.txt` one line at a time ("label ::: command"), log
in `$TMPDIR/glance/queue_big.log`. Edit the txt file to add or reorder jobs (write a temp file, then `mv`); create
`$TMPDIR/glance/gpuq.stop` to stop after the current job. If the machine rebooted, the queue is gone: restart the runner
and re-add the unfinished lines below (every collector job is resumable: finished rows are skipped).

| Order | Job | Output | Then do |
| --- | --- | --- | --- |
| running | `lab-hidden`: four `ens4d` readouts + final hidden states on all 5,000 lab items | `lab/runs/lab_hidden.jsonl`, `lab/hidden/ladders/*.npz` | E1: `uv run python tools/readout_ladder.py --bench ladders --logits lab/runs/lab_hidden.jsonl --out lab/READOUT_LADDER --dev` first (calibration split only), then without `--dev` ONCE; write entry with verdicts on H12 to H14 (entry 20). Ships into the harness only if it clearly wins (owner said yes to shipping). |
| next | `kadid-ref`: reference-anchored readouts, first 200 items of each KADID distortion | `lab/runs/kadid_ref.jsonl` | E9 (entry 29): compare `fast2_ref = ref_digits + ref_digitsrev` with no-reference `fast2` and `ens4d` on the SAME items (write a small script using `glance.lab.analyze.combine_rows` and `bench_report.evaluate`); verdicts on H23, H24. NOTE: the `ref_` collector path was never smoke-tested on the GPU; check the first rows (two images, `ref` first; finite logits) before trusting it. |
| then | `gen-accuracy` (E11): the same model WRITING one JSON answer per item, lab test items of the frontier table + the fresh suites | `lab/runs/gen_accuracy.jsonl` | Write the report: accuracy of written answers (invalid = wrong) with bootstrap intervals per suite, paired against the read at 0 labels / unlabeled / 32 labels (local rows of `results/lab/frontier_head_to_head.json`; `results/lab/fresh_commons.json`); `write_ms` is only a timing if the GPU was idle; verdicts on H28, H29 (entry 32). KADID subset of E11 not built yet. |
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

### Analysis tools already written for the queued jobs (run them when the job's DONE line appears in the queue log)

- `lab-hidden` -> `uv run python tools/readout_ladder.py --bench ladders --logits lab/runs/lab_hidden.jsonl --out lab/READOUT_LADDER --dev`, then once without `--dev`.
- `kadid-ref` -> `uv run python tools/kadid_ref_report.py` (prints the H23 / H24 verdicts; check the first rows of `lab/runs/kadid_ref.jsonl` first).
- `gen-accuracy` -> `uv run python tools/gen_accuracy_report.py` (H28 / H29; timings there are contended unless the GPU was idle).
- `letters-lab` -> `uv run python tools/readout_baselines.py` then `uv run python tools/make_results_comparisons.py`.
- Gemini baselines (owner started them through OpenRouter at 14:36): lab scales `uv run python tools/compare_frontier_lab.py --run 20260920T165748Z-8ff72a --run 20260920T165949Z-8ff72a --run 20260920T170146Z-8ff72a`; fresh photos `uv run python tools/fresh_report.py --run 20260920T205633Z-99f822 --run 20260920T205633Z-99f822-gpt --run 20260920T205633Z-99f822-gemini`.
After each: notebook entry with verdicts (real clock), regenerate `tools/make_results_comparisons.py` / `make_results_generalization.py`, update `docs/CLAIMS.md`, run `tools/verify_docs_numbers.py`, commit. The owner is away for a few hours from 14:40 and expects unattended progress.

Second lane since 14:50: `$TMPDIR/glance/gpuq_b.sh` + `gpuq_b.txt` runs `smol-lab` in parallel with lane A (log lines tagged `[gpuqB]` in the same log). Add further small-model jobs (external systems) to lane B, not lane A.

Lane A was stopped from the app's task list at about 15:00 and resumed at the owner's word ("resume") at 15:27; `lab-hidden` was put back at the front and continues where it stopped (a few dozen items between the last hidden-state flush and the stop have logits but no hidden vectors; the analysis tool skips them). The task named "Run GPU queue lane A" in the app IS the experiment queue: stopping it stops the GPU work.

### Machine resources (32 GB Apple M5, one GPU)

One model process at a time is comfortable (56% memory free with the queue job running). The heavy swap use seen on
2026-09-20 (8.4 GB) came from running two or three model processes at once (queue + smoke tests + an eval): that also
contaminates any timing. Rules: only the queue runs GPU jobs; at most ONE extra model process beside it, never during a
timing benchmark; agents get smoke tests only. `$TMPDIR/glance/resmon.sh` logs memory, swap, disk, thermal limit and the
number of model processes to `$TMPDIR/glance/resources.log` every 5 minutes and exits with an alert if memory free < 8%,
swap > 24 GB, disk free < 100 GB or the CPU speed limit drops below 60. Jobs run under `caffeinate -i` (no idle sleep),
but closing the lid without an external display still sleeps the machine; the queue then simply pauses. Disk: 1.2 TB free.

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

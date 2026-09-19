# STATUS

Build log for glance v0. `HANDOFF.md` is the spec. Newest milestone at the bottom.

## Decisions

Where the spec was ambiguous, the simpler option was taken and recorded here.

- D1. Modules not named in the section 4 layout: `glance/config.py` (config loader, M0 asks for one),
  `glance/doctor.py` (device detection behind `glance doctor`), `glance/pipeline.py` (the one `decide()` function
  that both the server and the CLI call). `config` and `logging_utils` import nothing from glance; `pipeline`
  sits at the right end of the dependency order with `server`, `cli`, `evals`.
- D2. Tier table gaps resolve downward, with a doctor warning: Apple Silicon with 17-31 GB uses the 8-16 GB tier;
  CUDA under 12 GB and Apple Silicon under 8 GB get the dual encoder only.
- D3. `vram_gb` on Apple Silicon is `torch.mps.recommended_max_memory()` (Metal's working-set ceiling), since
  memory is unified.

## Dependencies beyond the HANDOFF list

- `pyyaml`: reads `configs/default.yaml` (already a transitive dependency of transformers).
- `scipy`: `scipy.optimize` for calibration fits, named in HANDOFF section 7 (already a dependency of scikit-learn).
- `torchvision`: required by the Qwen3-VL and SigLIP2 image/video processors in transformers.
- `hatchling` (build-time only): builds the editable install that provides the `glance` console script.

## M0: scaffold, uv env, config loader, logging utils, `glance doctor` (2026-09-19)

Built: repo layout, `pyproject.toml` + committed `uv.lock` (Python 3.11, torch 2.14.0, transformers 5.17.0),
`configs/default.yaml` with all models pinned by SHA, `glance/config.py`, `glance/logging_utils.py`
(JSONL writer, ULID-style request ids, timer, error records, once-per-op MPS fallback logging),
`glance/doctor.py`, `glance doctor`. `uv` itself was installed with Homebrew (it was not on the machine).

Check: `glance doctor --json` exits 0, writes `logs/doctor.json`, names the tier.

```text
chip Apple M5 | ram_gb 32.0 | vram_gb 25.0 | device mps | dtype float16 | torch 2.14.0 | free_disk_gb 1279
selected_tier apple_32gb
selected_models siglip google/siglip2-base-patch16-256@3f9f96cb  vlm Qwen/Qwen3-VL-4B-Instruct@ebb281ec
image_token_budget 768 | warnings []
pytest: 20 passed
```

Deviations: none. Open questions: none.

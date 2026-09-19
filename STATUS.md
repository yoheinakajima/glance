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
- D4. Responses carry raw probabilities for every type. Section 7 requires "both raw and calibrated probabilities"
  in every response, while the section 5 example shows `raw` only on `noul`. `choice` and `score` answers
  therefore also carry a `raw` map, mirroring `noul.raw`. No existing field changed shape.
- D5. `options.calibrated` defaults to `false`, so a request without options never hits `calibration_mismatch`.
- D6. Dual encoder: one image per question. The scorer picks the image the instructions name by backticked id
  (or the only image); anything else is `unsupported_question_for_backend`. SigLIP2 runs in float32 on every
  device (375M params), text is lowercased and padded to 64 tokens as in its training setup, and `context` is
  not used because a dual encoder embeds image and candidate text separately.
- D7. `usage.forward_passes` counts scored statements (the section 5 example: 1 + 3 + 3 = 7), not prefix passes.
- D8. Image `url` sources are fetched with urllib at request time. That is the only network call the server can
  make at inference, and only when the caller asks for it.
- D9. Frontier baseline contract (used in M5): the scorer renders one prompt plus an enumerated answer list per
  question and the backend returns one pick per question, so the frontier backend stays type-agnostic too.

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

## M1: schema, images, scorer, SigLIP backend, `glance decide` (2026-09-19)

Built: `schema.py` (pydantic v2 request/answer/error models, limits, 422 detail with field paths), `images.py`
(path | url | base64, sha256 of original bytes, EXIF transpose, alpha flatten, 2048 px max side), `prompts.py`
(`PROMPT_VERSION = "p1"`), `backends/base.py` + `backends/siglip.py`, `scorer.py`, `calibration.py`
(fit/save/load/apply, exercised in M4), `pipeline.py` (`Engine.decide`, call log), `glance decide`,
three CC0 samples with request files (`samples/README.md` has provenance).

Check: the three bundled samples return valid typed JSON on `--model siglip`; invalid bodies return 422 with the
field path; scorer unit tests cover sigmoid, softmax, score mean, confidence, margin on fixed logits.

```text
receipt: is_receipt 0.918 | doc_type receipt (0.998) conf 0.986 | legibility 1.36 | image_tokens 256, forward_passes 8
invoice: is_receipt 0.792 | doc_type receipt (0.738) conf 0.476 | legibility 1.76   <- SigLIP calls the invoice a receipt
dog:     is_receipt 0.032 | doc_type other   (0.997) conf 0.977 | legibility 1.03
bad body -> {"code": "validation_error", "detail": [{"path": "questions.q.criteria", "message": "Field required"}]}
pytest: 65 passed (GLANCE_TEST_MODELS=1 includes the three samples on the real SigLIP2 weights)
```

Deviations: none. Open questions: none.

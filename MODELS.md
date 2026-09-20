# Models

Licenses were read from each model card's metadata on the Hugging Face Hub (`/api/models/<id>`, `license` tag and
`cardData.license`) before download. Only Apache-2.0 or MIT weights are allowed (HANDOFF section 11).
Revisions are pinned in `configs/default.yaml`.

| Role | Model id | License | Revision SHA | Size | Checked | Downloaded |
| --- | --- | --- | --- | --- | --- | --- |
| Dual encoder (all devices) | `google/siglip2-base-patch16-256` | Apache-2.0 | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` | 1.54 GB | 2026-09-19 | yes |
| VLM, tier `apple_32gb` and `cuda_12gb` (selected on this machine) | `Qwen/Qwen3-VL-4B-Instruct` | Apache-2.0 | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 8.89 GB | 2026-09-19 | yes |
| VLM, tier `apple_8gb` | `Qwen/Qwen3-VL-2B-Instruct` | Apache-2.0 | `89644892e4d85e24eaac8bacfd4f463576704203` | 4.27 GB | 2026-09-19 | no (not this machine's tier) |
| VLM, tier `cuda_24gb` | `Qwen/Qwen3-VL-8B-Instruct` | Apache-2.0 | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 17.55 GB | 2026-09-19 | no (not this machine's tier) |
| Lab only: second model family for the replication (`lab/NOTES.md` entries 20 and 22; owner approved 2026-09-20). Not used by the harness. | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Apache-2.0 | `482adb537c021c86670beed01cd58990d01e72e4` | 8.99 GB (float32 weights) | 2026-09-20 | yes |
| Outside system for comparison, not shipped (`lab/NOTES.md` entry 37; `docs/paper/COMPARABLE_SYSTEMS.md` "Four systems" item 1). Adapter: `glance/lab/external_systems.py:OpenJevV2`. Never loaded on this lab's machine -- only its tokenizer and image *processor* were used (see the module's HARD RULE note); its `.load()` is left implemented for a machine with headroom. | `AlexWortega/openjev` (v2, subfolder `qwen3.5-4b-nli-v2`) | MIT | `4395b29714015162db6112de91c35688e6e42717` | 9.10 GB (only the v2 4B checkpoint + `code/`/top-level configs were downloaded, per the download budget; the repo's 0.8B and 35B-A3B checkpoints were not) | 2026-09-20 | yes (weights on disk; never instantiated into a model) |
| Outside system for comparison, not shipped (`lab/NOTES.md` entry 37; `docs/paper/COMPARABLE_SYSTEMS.md` "Four systems" item 2). Adapter: `glance/lab/external_systems.py:QSitMini`. CPU-only, float32 on this lab's machine; smoke-tested on 4 images (2026-09-20, peak RSS ~5.75 GB). | `zhangzicheng/q-sit-mini` | MIT | `198f645ecbebd113041aa66fb973a1055d6a8c0d` | 1.86 GB | 2026-09-20 | yes |

Sizes above are the resolved commit's true download size (Hugging Face API `usedStorage`/per-file `size`, cross-checked
against `os.path.getsize` on the cached weight files). A plain `du -sh` on either model's folder under `.cache/hf/hub/`
reports far less, because this machine's `huggingface_hub` stores large files through a shared, symlinked "xet" content
store outside the per-repo folder (`<model dir>/blobs/<hash>` -> `../../blobs/<xx>/<hash>`), which `du` does not follow
by default.

The frontier baseline model is not a local weight. Its LiteLLM id comes from `FRONTIER_MODEL` and is logged per call;
its outputs are evaluation-only and are never written anywhere they could serve as training labels.

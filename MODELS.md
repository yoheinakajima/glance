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

The frontier baseline model is not a local weight. Its LiteLLM id comes from `FRONTIER_MODEL` and is logged per call;
its outputs are evaluation-only and are never written anywhere they could serve as training labels.

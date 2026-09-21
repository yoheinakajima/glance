# Does the zero-shot read improve with model size? (E15; Qwen3-VL, identical prompts and settings; the 1,000 lab images the frontier models saw)

| Model | zero-shot exact, four-pass read | zero-shot exact, one-pass read | within one level | rank agreement (Spearman) | + 16 unlabeled images | + 32 labels | gain from 32 labels, points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.388 [0.358, 0.417] | 0.492 (0.623 with 16 unlabeled) | 0.893 | 0.893 | 0.644 [0.618, 0.669] | 0.842 [0.825, 0.858] | +45.4 |
| Qwen3-VL-4B | 0.570 [0.540, 0.601] | 0.669 (0.758 with 16 unlabeled) | 0.988 | 0.934 | 0.694 [0.669, 0.718] | 0.853 [0.835, 0.870] | +28.3 |
| Qwen3-VL-8B | pending | | | | | | |
| SmolVLM2-2.2B (another family) | 0.419 [0.388, 0.450] | not collected | 0.837 | 0.833 | 0.519 [0.492, 0.546] | 0.814 [0.795, 0.833] | +39.5 |
| anthropic/claude-opus-5, zero-shot written pick | 0.550 | | not stored | not stored | - | - | - |
| openai/gpt-5.6, zero-shot written pick | 0.597 | | not stored | not stored | - | - | - |
| openrouter/google/gemini-3.1-pro-preview, zero-shot written pick | 0.650 | | not stored | not stored | - | - | - |

Per scale, zero-shot exact accuracy:

| Model | blur | exposure | jpeg | noise | resolution |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.265 | 0.250 | 0.470 | 0.595 | 0.360 |
| Qwen3-VL-4B | 0.470 | 0.700 | 0.450 | 0.575 | 0.655 |
| SmolVLM2-2.2B (another family) | 0.345 | 0.515 | 0.305 | 0.485 | 0.445 |

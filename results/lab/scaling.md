# Does the zero-shot read improve with model size? (E15; Qwen3-VL, identical prompts and settings; the 1,000 lab images the frontier models saw)

| Model | zero-shot exact, four-pass read | zero-shot exact, one-pass read | within one level | rank agreement (Spearman) | + 16 unlabeled images | + 32 labels | gain from 32 labels, points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.388 [0.358, 0.417] | 0.492 (0.623 with 16 unlabeled) | 0.893 | 0.893 | 0.644 [0.618, 0.669] | 0.842 [0.825, 0.858] | +45.4 |
| Qwen3-VL-4B | 0.570 [0.540, 0.601] | 0.669 (0.758 with 16 unlabeled) | 0.988 | 0.934 | 0.694 [0.669, 0.718] | 0.853 [0.835, 0.870] | +28.3 |
| Qwen3-VL-8B | 0.537 [0.506, 0.568] | 0.643 (0.677 with 16 unlabeled) | 0.979 | 0.915 | 0.714 [0.690, 0.737] | 0.839 [0.820, 0.857] | +30.2 |
| SmolVLM2-2.2B (another family) | 0.419 [0.389, 0.449] | not collected | 0.837 | 0.833 | 0.519 [0.493, 0.547] | 0.814 [0.794, 0.833] | +39.5 |
| anthropic/claude-opus-5, zero-shot written pick | 0.550 | | not stored | not stored | - | - | - |
| openai/gpt-5.6, zero-shot written pick | 0.597 | | not stored | not stored | - | - | - |
| openrouter/google/gemini-3.1-pro-preview, zero-shot written pick | 0.650 | | not stored | not stored | - | - | - |
| anthropic/claude-haiku-4-5, zero-shot written pick | 0.609 | | not stored | not stored | - | - | - |
| openai/gpt-5.6-luna, zero-shot written pick | 0.686 | | not stored | not stored | - | - | - |
| openrouter/google/gemini-3.1-flash-lite, zero-shot written pick | 0.763 | | not stored | not stored | - | - | - |

Per scale, zero-shot exact accuracy:

| Model | blur | exposure | jpeg | noise | resolution |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.265 | 0.250 | 0.470 | 0.595 | 0.360 |
| Qwen3-VL-4B | 0.470 | 0.700 | 0.450 | 0.575 | 0.655 |
| Qwen3-VL-8B | 0.645 | 0.595 | 0.320 | 0.470 | 0.655 |
| SmolVLM2-2.2B (another family) | 0.345 | 0.515 | 0.305 | 0.485 | 0.445 |

Registered predictions (`lab/NOTES.md` entry 38):

- H33a: zero-shot exact accuracy rises with size (2B < 4B < 8B): NOT SUPPORTED
- H33b: 8B stays below 0.70 zero-shot: SUPPORTED
- H34a: within one level >= 0.97 at 4B and 8B: SUPPORTED
- H34b: rank agreement rises with size: NOT SUPPORTED
- H35a: the gain from 32 labels shrinks with size: NOT SUPPORTED
- H35b: the gain from 32 labels is still >= 15 points at 8B: SUPPORTED

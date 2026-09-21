# Interface screens: two defects of the test, and the disabled-button diagnostic (post hoc; `lab/NOTES.md` entry 49d)

## Page type

| System | all items (registered) | without the defective items (post hoc) |
| --- | --- | --- |
| Qwen3-VL-4B + Glance | 0.780 | 1.000 |
| Claude Opus 5 | 0.807 | 1.000 |
| GPT-5.6 | 0.807 | 1.000 |
| Gemini 3.1 Pro | 0.807 | 1.000 |
| Claude Haiku 4.5 | 0.807 | 1.000 |
| GPT-5.6 Luna | 0.807 | 1.000 |
| Gemini 3.1 Flash-Lite | 0.807 | 1.000 |

Same items for every system: 150 and 113. Open model on all its items: 0.823 (n=300) and 1.000 (n=233).

## Is the goal already done

| System | all items (registered) | without the defective items (post hoc) |
| --- | --- | --- |
| Qwen3-VL-4B + Glance | 0.830 | 0.920 |
| Claude Opus 5 | 0.890 | 1.000 |
| GPT-5.6 | 0.890 | 1.000 |
| Gemini 3.1 Pro | 0.890 | 1.000 |
| Claude Haiku 4.5 | 0.850 | 0.947 |
| GPT-5.6 Luna | 0.890 | 1.000 |
| Gemini 3.1 Flash-Lite | 0.860 | 0.960 |

Same items for every system: 100 and 75. Open model on all its items: 0.825 (n=200) and 0.906 (n=160).

## Disabled main button, reported on the screens that have it

| System | detected |
| --- | --- |
| Qwen3-VL-4B + Glance | 0.07 |
| Claude Opus 5 | 1.00 |
| GPT-5.6 | 0.50 |
| Gemini 3.1 Pro | 0.36 |
| Claude Haiku 4.5 | 0.00 |
| GPT-5.6 Luna | 0.29 |
| Gemini 3.1 Flash-Lite | 0.00 |

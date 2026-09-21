# Five systems, three tests: accuracy, speed, cost (same items in every row)

## Accuracy, zero-shot [95% interval]

| System | Yes/no, 541 questions about fresh photographs (three sets pooled) | Pick one, 270 fresh photographs (three sets pooled) | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 0.959 [0.941, 0.976] | 0.937 [0.907, 0.963] | 0.650 [0.621, 0.679] |
| Claude Opus 5 | 0.937 [0.915, 0.957] | 0.937 [0.907, 0.963] | 0.550 [0.519, 0.581] |
| GPT-5.6 | 0.928 [0.906, 0.948] | 0.904 [0.867, 0.937] | 0.597 [0.567, 0.627] |
| Claude Haiku 4.5 | 0.839 [0.808, 0.869] | 0.785 [0.737, 0.833] | 0.609 [0.579, 0.639] |
| GPT-5.6 Luna | 0.906 [0.880, 0.930] | 0.881 [0.841, 0.919] | 0.686 [0.657, 0.715] |
| Gemini 3.1 Flash-Lite | 0.961 [0.945, 0.976] | 0.933 [0.904, 0.959] | 0.763 [0.737, 0.789] |
| Qwen3-VL-4B, written | 0.945 [0.924, 0.963] | 0.930 [0.900, 0.959] | 0.672 [0.643, 0.701] |
| Qwen3-VL-2B, read (Glance) | 0.904 [0.878, 0.928] | 0.907 [0.874, 0.941] | 0.492 [0.461, 0.524] |
| Qwen3-VL-4B, read (Glance) | 0.939 [0.917, 0.957] | 0.933 [0.904, 0.959] | 0.669 [0.640, 0.698] |
| Qwen3-VL-8B, read (Glance) | 0.933 [0.911, 0.954] | 0.926 [0.893, 0.956] | 0.643 [0.614, 0.672] |

## Median seconds per answer

| System | Yes/no, 541 questions about fresh photographs (three sets pooled) | Pick one, 270 fresh photographs (three sets pooled) | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 2.74 s | 3.18 s | 4.02 s |
| Claude Opus 5 | 2.64 s | 2.98 s | 2.42 s |
| GPT-5.6 | 1.18 s | 1.34 s | 1.07 s |
| Claude Haiku 4.5 | 0.87 s | 0.92 s | 0.72 s |
| GPT-5.6 Luna | 1.22 s | 1.30 s | 1.10 s |
| Gemini 3.1 Flash-Lite | 1.63 s | 1.89 s | 1.62 s |
| Qwen3-VL-4B, written | 1.65 s | 1.87 s | 0.91 s |
| Qwen3-VL-2B, read (Glance) | 0.66 s | 0.85 s | 0.22 s |
| Qwen3-VL-4B, read (Glance) | 1.08 s | 1.43 s | 0.45 s |
| Qwen3-VL-8B, read (Glance) | 2.12 s | 2.65 s | 0.64 s |

## US dollars per 1,000 answers

| System | Yes/no, 541 questions about fresh photographs (three sets pooled) | Pick one, 270 fresh photographs (three sets pooled) | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | $2.62 | $3.85 | $6.15 |
| Claude Opus 5 | $3.37 | $4.48 | $18.70 (est.) |
| GPT-5.6 | $7.55 | $7.82 | $14.96 (est.) |
| Claude Haiku 4.5 | $1.75 | $1.89 | $0.55 |
| GPT-5.6 Luna | $0.38 | $0.40 | $0.12 |
| Gemini 3.1 Flash-Lite | $0.31 | $0.34 | $0.33 |
| Qwen3-VL-4B, written | $0.24 to $0.37 | $0.27 to $0.42 | $0.13 to $0.20 |
| Qwen3-VL-2B, read (Glance) | $0.10 to $0.15 | $0.12 to $0.19 | $0.03 to $0.05 |
| Qwen3-VL-4B, read (Glance) | $0.16 to $0.24 | $0.21 to $0.32 | $0.07 to $0.10 |
| Qwen3-VL-8B, read (Glance) | $0.31 to $0.47 | $0.39 to $0.59 | $0.09 to $0.14 |

## What only the read row can add (rating accuracy)

| Give it | exact-level accuracy | note |
| --- | --- | --- |
| 16 UNLABELED images of the rubric (`glance fit --unlabeled`) | 0.758 | zero labels, but not zero-shot; same 1,000 images, one pass; probabilities improve but are not calibrated |
| 32 labeled images of the rubric (`glance fit`) | 0.857 | same 1,000 images, the four-pass readout; calibrated probabilities (ECE about 0.03) |
| 500 labels, readout fitted on the hidden state (research result, not shipped) | 0.965 | full test split, ONE forward pass; needs on the order of a hundred labels to beat the row above |

Caveats:
- The rating read is one forward pass at the JSON answer position (registered and scored once, notebook entry 42b). The four-pass readout shipped earlier scores 0.570 zero-shot on these images and remains the better option once labels exist.
- Frontier models were run once, zero-shot, with a constrained written pick; a failed call counts as wrong. No few-shot prompt was tried for any written row.
- Yes/no and pick-one: photographs taken after every model's release; labels are uploaders' structured 'depicts' statements (pick-one labels are noisy). Ratings: five synthetic 4-level scales.
- Hosted latency includes the network from one laptop; hosted cost is the provider's bill per call where logged, otherwise a list-price upper estimate.
- Open-model cost is arithmetic on measured seconds and an on-demand cloud GPU price; several questions about one image share its cost and get cheaper per answer (see the cost model).

# Five systems, three tests: accuracy, speed, cost (same items in every row)

## Accuracy, zero-shot [95% interval]

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 0.947 [0.908, 0.985] | 0.923 [0.846, 0.985] | 0.650 [0.621, 0.679] |
| Claude Opus 5 | 0.924 [0.878, 0.969] | 0.908 [0.831, 0.969] | 0.550 [0.519, 0.581] |
| GPT-5.6 | 0.893 [0.840, 0.947] | 0.892 [0.815, 0.954] | 0.597 [0.567, 0.627] |
| Claude Haiku 4.5 | 0.939 [0.893, 0.977] | 0.846 [0.754, 0.923] | 0.609 [0.579, 0.639] |
| GPT-5.6 Luna | 0.924 [0.878, 0.969] | 0.908 [0.831, 0.969] | 0.686 [0.657, 0.715] |
| Gemini 3.1 Flash-Lite | 0.954 [0.916, 0.985] | 0.908 [0.831, 0.969] | 0.763 [0.737, 0.789] |
| Qwen3-VL-4B, written | 0.931 [0.885, 0.969] | 0.892 [0.815, 0.954] | 0.672 [0.643, 0.701] |
| Qwen3-VL-4B, read (Glance) | 0.931 [0.885, 0.969] | 0.862 [0.769, 0.938] | 0.669 [0.640, 0.698] |

## Median seconds per answer

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 2.74 s | 3.18 s | 4.02 s |
| Claude Opus 5 | 2.64 s | 2.98 s | 2.42 s |
| GPT-5.6 | 1.18 s | 1.34 s | 1.07 s |
| Claude Haiku 4.5 | 0.87 s | 0.92 s | 0.72 s |
| GPT-5.6 Luna | 1.22 s | 1.30 s | 1.10 s |
| Gemini 3.1 Flash-Lite | 1.63 s | 1.89 s | 1.62 s |
| Qwen3-VL-4B, written | 1.65 s | 1.87 s | 0.91 s |
| Qwen3-VL-4B, read (Glance) | 1.08 s | 1.43 s | 0.45 s |

## US dollars per 1,000 answers

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | $2.62 | $3.85 | $6.15 |
| Claude Opus 5 | $3.37 | $4.48 | $18.70 (est.) |
| GPT-5.6 | $7.55 | $7.82 | $14.96 (est.) |
| Claude Haiku 4.5 | $1.75 | $1.89 | $0.55 |
| GPT-5.6 Luna | $0.38 | $0.40 | $0.12 |
| Gemini 3.1 Flash-Lite | $0.31 | $0.34 | $0.33 |
| Qwen3-VL-4B, written | $0.24 to $0.37 | $0.27 to $0.42 | $0.13 to $0.20 |
| Qwen3-VL-4B, read (Glance) | $0.16 to $0.24 | $0.21 to $0.32 | $0.07 to $0.10 |

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

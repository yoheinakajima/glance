# Five systems, three tests: accuracy, speed, cost (same items in every row)

## Accuracy, zero-shot [95% interval]

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 0.947 [0.908, 0.985] | 0.923 [0.862, 0.985] | 0.650 [0.621, 0.679] |
| Claude Opus 5 | 0.924 [0.878, 0.969] | 0.908 [0.831, 0.969] | 0.550 [0.518, 0.581] |
| GPT-5.6 | 0.893 [0.840, 0.947] | 0.892 [0.815, 0.954] | 0.597 [0.565, 0.628] |
| Qwen3-VL-4B, written | 0.931 [0.885, 0.969] | 0.892 [0.815, 0.954] | 0.672 [0.642, 0.701] |
| Qwen3-VL-4B, read (Glance) | 0.931 [0.885, 0.969] | 0.862 [0.769, 0.938] | 0.570 [0.539, 0.600] |

## Median seconds per answer

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | 2.74 s | 3.18 s | 4.02 s |
| Claude Opus 5 | 2.64 s | 2.98 s | 2.42 s |
| GPT-5.6 | 1.18 s | 1.34 s | 1.07 s |
| Qwen3-VL-4B, written | 0.80 s | pending | pending |
| Qwen3-VL-4B, read (Glance) | 0.34 s | pending | 1.09 s |

## US dollars per 1,000 answers

| System | Yes/no, 131 fresh Commons questions | Pick one of 13, 65 fresh Commons photos | Rating, exact level of 4, 1,000 lab images |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | $2.62 | $3.85 | $6.15 |
| Claude Opus 5 | $18.70 (est.) | $18.70 (est.) | $18.70 (est.) |
| GPT-5.6 | $7.55 | $7.82 | $14.96 (est.) |
| Qwen3-VL-4B, written | $0.12 to $0.18 | pending | pending |
| Qwen3-VL-4B, read (Glance) | $0.05 to $0.08 | pending | $0.16 to $0.24 |

## What only the read row can add (rating accuracy)

| Give it | exact-level accuracy | note |
| --- | --- | --- |
| UNLABELED images of the rubric (`glance fit --unlabeled`) | 0.702 | zero labels, but not zero-shot; 500 unlabeled images here, and 16 already give 0.686 on the full test split; probabilities improve but are not calibrated |
| 32 labeled images of the rubric (`glance fit`) | 0.857 | same 1,000 images; calibrated probabilities (ECE about 0.03) |
| 500 labels, readout fitted on the hidden state (research result, not shipped) | 0.965 | full test split, ONE forward pass; needs on the order of a hundred labels to beat the row above |

Caveats:
- Frontier models were run once, zero-shot, with a constrained written pick; a failed call counts as wrong. No few-shot prompt was tried for any written row.
- Yes/no and pick-one: photographs taken after every model's release; labels are uploaders' structured 'depicts' statements (pick-one labels are noisy). Ratings: five synthetic 4-level scales.
- Hosted latency includes the network from one laptop; hosted cost is the provider's bill per call where logged, otherwise a list-price upper estimate.
- Open-model cost is arithmetic on measured seconds and an on-demand cloud GPU price; several questions about one image share its cost and get cheaper per answer (see the cost model).

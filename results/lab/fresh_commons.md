# Out of the box on fresh real photos (taken after the models were released; labels from Commons 'depicts' statements)

Uncalibrated decisions, all items, bootstrap 95% intervals. Label noise, the same for every system: a 'depicts' tag means the thing APPEARS in the photo, not that it is the main subject, so the pick-one labels are noisier than the yes/no labels; a 'no' question can be wrong when the other object happens to be in frame.

## Yes/no: "Is there a <class> in the photo?"

| System | n | accuracy | on yes questions | on no questions |
| --- | --- | --- | --- | --- |
| vlm:statement | 262 | 0.931 [0.901, 0.962] | 0.908 | 0.954 |
| anthropic/claude-opus-5 | 131 | 0.924 [0.878, 0.969] | 0.894 | 0.954 |
| openai/gpt-5.6 | 131 | 0.893 [0.840, 0.939] | 0.909 | 0.877 |

## Pick one of 13: "What is the main subject?"

| System | n | accuracy |
| --- | --- | --- |
| vlm:independent | 131 | 0.885 [0.824, 0.939] |
| vlm:letter | 131 | 0.885 [0.832, 0.939] |
| siglip:independent | 131 | 0.855 [0.794, 0.908] |
| anthropic/claude-opus-5 | 65 | 0.908 [0.831, 0.969] |
| openai/gpt-5.6 | 65 | 0.892 [0.815, 0.954] |

Per class (accuracy, photos):

| System | beach | bicycle | bird | boat | bridge | car | cat | church | dog | flower | horse | mountain | train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vlm:independent | 0.92 (12) | 0.00 (1) | 0.71 (7) | 1.00 (12) | 0.92 (12) | 0.58 (12) | 1.00 (10) | 0.92 (12) | 0.92 (12) | 1.00 (12) | 0.60 (5) | 0.92 (12) | 1.00 (12) |
| vlm:letter | 0.75 (12) | 0.00 (1) | 0.71 (7) | 1.00 (12) | 0.92 (12) | 0.75 (12) | 1.00 (10) | 0.92 (12) | 0.92 (12) | 1.00 (12) | 0.60 (5) | 0.92 (12) | 1.00 (12) |
| siglip:independent | 0.83 (12) | 0.00 (1) | 0.57 (7) | 1.00 (12) | 1.00 (12) | 0.33 (12) | 1.00 (10) | 1.00 (12) | 0.83 (12) | 1.00 (12) | 0.60 (5) | 0.92 (12) | 1.00 (12) |
| anthropic/claude-opus-5 | 1.00 (5) | - | 0.50 (4) | 1.00 (5) | 1.00 (6) | 0.71 (7) | 1.00 (4) | 0.88 (8) | 1.00 (6) | 1.00 (5) | 1.00 (2) | 0.86 (7) | 1.00 (6) |
| openai/gpt-5.6 | 1.00 (5) | - | 0.25 (4) | 1.00 (5) | 1.00 (6) | 0.71 (7) | 1.00 (4) | 1.00 (8) | 1.00 (6) | 1.00 (5) | 0.50 (2) | 0.86 (7) | 1.00 (6) |

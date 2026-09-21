# Out of the box on fresh real photos (taken after the models were released; labels are the insect ORDER of iNaturalist research-grade community identifications)

Uncalibrated decisions, all items, bootstrap 95% intervals. A deliberately harder test (lab/NOTES.md entry 47): the seven classes are insect orders that look alike, and every 'no' question names another insect order.

## Yes/no: "Is the main subject a <insect order>?"

| System | n | accuracy | on yes questions | on no questions |
| --- | --- | --- | --- | --- |
| vlm:statement | 420 | 0.948 [0.926, 0.967] | 0.919 | 0.976 |
| anthropic/claude-opus-5 | 210 | 0.938 [0.905, 0.967] | 0.885 | 0.982 |
| anthropic/claude-haiku-4-5 | 210 | 0.786 [0.729, 0.838] | 0.573 | 0.965 |
| openai/gpt-5.6 | 210 | 0.943 [0.910, 0.971] | 0.927 | 0.956 |
| openai/gpt-5.6-luna | 210 | 0.867 [0.819, 0.910] | 0.802 | 0.921 |
| openrouter/google/gemini-3.1-pro-preview | 210 | 0.967 [0.943, 0.990] | 0.948 | 0.982 |
| openrouter/google/gemini-3.1-flash-lite | 210 | 0.962 [0.933, 0.986] | 0.948 | 0.974 |

## Pick one of 7 insect orders: "What kind of insect is the main subject?"

| System | n | accuracy |
| --- | --- | --- |
| vlm:independent | 210 | 0.962 [0.933, 0.986] |
| vlm:letter | 210 | 0.948 [0.914, 0.976] |
| siglip:independent | 210 | 0.710 [0.648, 0.771] |
| anthropic/claude-opus-5 | 105 | 0.962 [0.924, 0.990] |
| anthropic/claude-haiku-4-5 | 105 | 0.743 [0.657, 0.829] |
| openai/gpt-5.6 | 105 | 0.905 [0.848, 0.952] |
| openai/gpt-5.6-luna | 105 | 0.886 [0.819, 0.943] |
| openrouter/google/gemini-3.1-pro-preview | 105 | 0.971 [0.933, 1.000] |
| openrouter/google/gemini-3.1-flash-lite | 105 | 0.952 [0.905, 0.990] |

Per class (accuracy, photos):

| System | bee_wasp_or_ant | beetle | butterfly_or_moth | dragonfly_or_damselfly | fly | grasshopper_or_cricket | true_bug |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlm:independent | 0.97 (30) | 1.00 (30) | 1.00 (30) | 1.00 (30) | 0.90 (30) | 1.00 (30) | 0.87 (30) |
| vlm:letter | 0.97 (30) | 0.97 (30) | 1.00 (30) | 1.00 (30) | 0.77 (30) | 1.00 (30) | 0.93 (30) |
| siglip:independent | 0.93 (30) | 0.83 (30) | 0.90 (30) | 0.90 (30) | 0.23 (30) | 0.70 (30) | 0.47 (30) |
| anthropic/claude-opus-5 | 0.93 (15) | 1.00 (13) | 1.00 (17) | 1.00 (14) | 0.89 (19) | 0.94 (17) | 1.00 (10) |
| anthropic/claude-haiku-4-5 | 1.00 (15) | 0.69 (13) | 0.82 (17) | 0.93 (14) | 0.37 (19) | 0.76 (17) | 0.70 (10) |
| openai/gpt-5.6 | 0.93 (15) | 1.00 (13) | 1.00 (17) | 0.93 (14) | 0.79 (19) | 0.82 (17) | 0.90 (10) |
| openai/gpt-5.6-luna | 0.93 (15) | 1.00 (13) | 1.00 (17) | 1.00 (14) | 0.74 (19) | 0.76 (17) | 0.80 (10) |
| openrouter/google/gemini-3.1-pro-preview | 1.00 (15) | 1.00 (13) | 1.00 (17) | 1.00 (14) | 0.89 (19) | 0.94 (17) | 1.00 (10) |
| openrouter/google/gemini-3.1-flash-lite | 1.00 (15) | 1.00 (13) | 1.00 (17) | 1.00 (14) | 0.79 (19) | 0.94 (17) | 1.00 (10) |

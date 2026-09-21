# Out of the box on fresh real photos (taken after the models were released; labels from iNaturalist research-grade community identifications (iconic taxon))

Uncalibrated decisions, all items, bootstrap 95% intervals. Labels are community-verified identifications of the photographed organism. What stays hard, for every system: tiny or camouflaged subjects, and evidence photos (tracks, droppings, shells, burrows) that count as the organism on iNaturalist.

## Yes/no: "Is the main subject a <class>?"

| System | n | accuracy | on yes questions | on no questions |
| --- | --- | --- | --- | --- |
| vlm:statement | 400 | 0.945 [0.922, 0.968] | 0.950 | 0.940 |
| anthropic/claude-opus-5 | 200 | 0.945 [0.910, 0.975] | 0.907 | 0.989 |
| openai/gpt-5.6 | 200 | 0.935 [0.900, 0.965] | 0.925 | 0.946 |
| openrouter/google/gemini-3.1-pro-preview | 200 | 0.960 [0.930, 0.985] | 0.944 | 0.978 |

## Pick one of 10: "What kind of organism is the main subject?"

| System | n | accuracy |
| --- | --- | --- |
| vlm:independent | 200 | 0.940 [0.905, 0.970] |
| vlm:letter | 200 | 0.950 [0.920, 0.980] |
| siglip:independent | 200 | 0.880 [0.835, 0.925] |
| anthropic/claude-opus-5 | 100 | 0.930 [0.880, 0.980] |
| openai/gpt-5.6 | 100 | 0.910 [0.850, 0.960] |
| openrouter/google/gemini-3.1-pro-preview | 100 | 0.910 [0.850, 0.960] |

Per class (accuracy, photos):

| System | amphibian | arachnid | bird | fish | fungus | insect | mammal | mollusc | plant | reptile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vlm:independent | 0.95 (20) | 0.85 (20) | 1.00 (20) | 0.95 (20) | 0.85 (20) | 0.95 (20) | 0.85 (20) | 1.00 (20) | 1.00 (20) | 1.00 (20) |
| vlm:letter | 1.00 (20) | 0.95 (20) | 1.00 (20) | 0.95 (20) | 0.85 (20) | 0.95 (20) | 0.85 (20) | 1.00 (20) | 1.00 (20) | 0.95 (20) |
| siglip:independent | 0.85 (20) | 0.85 (20) | 1.00 (20) | 0.90 (20) | 0.95 (20) | 0.85 (20) | 0.80 (20) | 0.90 (20) | 0.85 (20) | 0.85 (20) |
| anthropic/claude-opus-5 | 1.00 (9) | 0.62 (8) | 1.00 (11) | 1.00 (8) | 0.89 (9) | 1.00 (6) | 0.87 (15) | 1.00 (12) | 1.00 (10) | 0.92 (12) |
| openai/gpt-5.6 | 1.00 (9) | 0.62 (8) | 1.00 (11) | 0.88 (8) | 0.89 (9) | 1.00 (6) | 0.87 (15) | 0.92 (12) | 1.00 (10) | 0.92 (12) |
| openrouter/google/gemini-3.1-pro-preview | 1.00 (9) | 0.62 (8) | 1.00 (11) | 0.88 (8) | 0.89 (9) | 1.00 (6) | 0.80 (15) | 0.92 (12) | 1.00 (10) | 1.00 (12) |

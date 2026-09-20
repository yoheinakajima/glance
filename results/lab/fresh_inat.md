# Out of the box on fresh real photos (taken after the models were released; labels from iNaturalist research-grade community identifications (iconic taxon))

Uncalibrated decisions, all items, bootstrap 95% intervals. Labels are community-verified identifications of the photographed organism. What stays hard, for every system: tiny or camouflaged subjects, and evidence photos (tracks, droppings, shells, burrows) that count as the organism on iNaturalist.

## Yes/no: "Is the main subject a <class>?"

| System | n | accuracy | on yes questions | on no questions |
| --- | --- | --- | --- | --- |
| vlm:statement | 400 | 0.945 [0.922, 0.968] | 0.950 | 0.940 |

## Pick one of 10: "What kind of organism is the main subject?"

| System | n | accuracy |
| --- | --- | --- |
| vlm:independent | 200 | 0.940 [0.905, 0.970] |
| vlm:letter | 200 | 0.950 [0.920, 0.980] |
| siglip:independent | 200 | 0.880 [0.835, 0.925] |

Per class (accuracy, photos):

| System | amphibian | arachnid | bird | fish | fungus | insect | mammal | mollusc | plant | reptile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vlm:independent | 0.95 (20) | 0.85 (20) | 1.00 (20) | 0.95 (20) | 0.85 (20) | 0.95 (20) | 0.85 (20) | 1.00 (20) | 1.00 (20) | 1.00 (20) |
| vlm:letter | 1.00 (20) | 0.95 (20) | 1.00 (20) | 0.95 (20) | 0.85 (20) | 0.95 (20) | 0.85 (20) | 1.00 (20) | 1.00 (20) | 0.95 (20) |
| siglip:independent | 0.85 (20) | 0.85 (20) | 1.00 (20) | 0.90 (20) | 0.95 (20) | 0.85 (20) | 0.80 (20) | 0.90 (20) | 0.85 (20) | 0.85 (20) |

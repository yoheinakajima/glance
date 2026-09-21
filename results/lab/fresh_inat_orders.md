# Out of the box on fresh real photos (taken after the models were released; labels are the insect ORDER of iNaturalist research-grade community identifications)

Uncalibrated decisions, all items, bootstrap 95% intervals. A deliberately harder test (lab/NOTES.md entry 47): the seven classes are insect orders that look alike, and every 'no' question names another insect order.

## Yes/no: "Is the main subject a <insect order>?"

| System | n | accuracy | on yes questions | on no questions |
| --- | --- | --- | --- | --- |
| vlm:statement | 420 | 0.948 [0.926, 0.969] | 0.919 | 0.976 |

## Pick one of 7 insect orders: "What kind of insect is the main subject?"

| System | n | accuracy |
| --- | --- | --- |
| vlm:independent | 210 | 0.962 [0.933, 0.986] |
| vlm:letter | 210 | 0.948 [0.914, 0.976] |
| siglip:independent | 210 | 0.710 [0.648, 0.771] |

Per class (accuracy, photos):

| System | bee_wasp_or_ant | beetle | butterfly_or_moth | dragonfly_or_damselfly | fly | grasshopper_or_cricket | true_bug |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlm:independent | 0.97 (30) | 1.00 (30) | 1.00 (30) | 1.00 (30) | 0.90 (30) | 1.00 (30) | 0.87 (30) |
| vlm:letter | 0.97 (30) | 0.97 (30) | 1.00 (30) | 1.00 (30) | 0.77 (30) | 1.00 (30) | 0.93 (30) |
| siglip:independent | 0.93 (30) | 0.83 (30) | 0.90 (30) | 0.90 (30) | 0.23 (30) | 0.70 (30) | 0.47 (30) |

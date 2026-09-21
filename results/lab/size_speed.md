# Writing JSON against reading with Glance, by model size (seconds per answer, idle GPU, one laptop)

Cost is the measured seconds at a rented-GPU price of $0.53 to $0.80 per hour, the same for every size (a larger model may need a dearer GPU; not modelled).

| Question | size | written, s | read, s | written / read | read, US dollars per 1,000 |
| --- | --- | --- | --- | --- | --- |
| yes/no, full-size photograph | 2B | 1.08 | 0.66 | 1.6x | 0.10 to 0.15 |
| yes/no, full-size photograph | 4B | 1.65 | 1.08 | 1.5x | 0.16 to 0.24 |
| yes/no, full-size photograph | 8B | 3.01 | 2.12 | 1.4x | 0.31 to 0.47 |
| pick-one, full-size photograph | 2B | 0.94 | 0.85 | 1.1x | 0.12 to 0.19 |
| pick-one, full-size photograph | 4B | 1.87 | 1.43 | 1.3x | 0.21 to 0.32 |
| pick-one, full-size photograph | 8B | 3.02 | 2.65 | 1.1x | 0.39 to 0.59 |
| rating, 448 px image | 2B | 0.30 | 0.22 | 1.4x | 0.03 to 0.05 |
| rating, 448 px image | 4B | 0.91 | 0.45 | 2.0x | 0.07 to 0.10 |
| rating, 448 px image | 8B | 2.02 | 0.64 | 3.2x | 0.09 to 0.14 |

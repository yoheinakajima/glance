# Cost of a rating on a fresh image, and what packing saves

Prefix cache cleared before every measurement; timed end to end from the image file (decode, resize, zoom crop, forward passes, readout). Uncontended GPU. `lab/LATENCY.json` reports the MARGINAL cost of one more readout on an image whose prefix is already cached; these are the full costs.

| Configuration | images | questions answered | p50 ms | p90 ms | p50 ms per question |
| --- | --- | --- | --- | --- | --- |
| independent, no cache (v0 as shipped) | 100 | 1 | 918 | 1013 | 918 |
| independent | 100 | 1 | 441 | 463 | 441 |
| digits | 100 | 1 | 359 | 479 | 359 |
| zoom_digits | 100 | 1 | 575 | 618 | 575 |
| ens4d, no cache, one readout at a time | 100 | 1 | 1571 | 1855 | 1571 |
| ens4d, cached, one readout at a time | 100 | 1 | 1248 | 1387 | 1248 |
| ens4d packed, 1 question | 100 | 1 | 1089 | 1148 | 1089 |
| ens4d packed, 5 questions | 100 | 5 | 2918 | 3030 | 584 |
| digits+digitsrev packed (no zoom), 5 questions | 100 | 5 | 1199 | 1246 | 240 |
| ens4d packed, 25 questions | 25 | 25 | 8529 | 8874 | 341 |
| digits+digitsrev packed (no zoom), 25 questions | 25 | 25 | 3333 | 3484 | 133 |

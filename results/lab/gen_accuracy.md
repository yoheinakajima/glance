# The same frozen Qwen3-VL-4B: WRITE a JSON answer, or READ it (Glance)? Accuracy on identical items

Writing = greedy generation of one JSON field, invalid or unparsable counts as wrong. Reading = raw, zero-label Glance readout (`ens4d` mean logits for ratings; yes/no and pick-one as shipped).

| Suite | n | written | read, 0 labels | read minus written, points | invalid written | write p50 ms (GPU was shared unless noted) |
| --- | --- | --- | --- | --- | --- | --- |
| ladder_blur | 200 | 0.755 [0.695, 0.815] | 0.470 [0.400, 0.540] | -28.5 [-35.0, -22.0] | 0 | 2511 |
| ladder_noise | 200 | 0.700 [0.635, 0.760] | 0.575 [0.505, 0.645] | -12.5 [-18.0, -7.0] | 0 | 2450 |
| ladder_jpeg | 200 | 0.345 [0.280, 0.410] | 0.450 [0.380, 0.520] | +10.5 [+6.0, +15.5] | 0 | 2370 |
| ladder_exposure | 200 | 0.810 [0.755, 0.865] | 0.700 [0.635, 0.765] | -11.0 [-19.5, -2.5] | 0 | 2433 |
| ladder_resolution | 200 | 0.750 [0.690, 0.810] | 0.655 [0.590, 0.720] | -9.5 [-17.5, -1.5] | 0 | 2329 |
| fresh_yesno | 262 | 0.931 [0.901, 0.958] | 0.931 [0.897, 0.962] | +0.0 [-1.1, +1.1] | 0 | 3544 |
| fresh_choice | 131 | 0.885 [0.824, 0.939] | 0.885 [0.832, 0.939] | +0.0 [-3.8, +3.8] | 0 | 3753 |

Lab scales, mean over five rubrics: written 0.672; read with 0 labels 0.570; read with unlabeled images 0.702; read with 32 labels 0.857 (the last two from `results/lab/frontier_head_to_head.json`, same items).

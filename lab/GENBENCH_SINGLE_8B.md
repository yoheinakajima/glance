# Write the answers or read them? Same frozen Qwen3-VL-4B, same images, same questions

Cold start per image, end to end from the image file, idle GPU. Writing = greedy generation of one JSON object with a token cap, no thinking. Reading = Glance (`glance decide`, uncalibrated), which also returns a probability for every answer.

| Request | images | write p50 ms (tokens) | read `fast2` p50 ms | read `ens4d` p50 ms | read is faster by | JSON failed to parse | invalid fields | written = read (valid fields) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 yes/no | 40 | 1834 (9) | 581 | - | 3.2x | 0 | 0.0% | 100.0% |
| 1 pick-one | 40 | 1462 (8) | 655 | 648 | 2.2x / 2.3x | 0 | 0.0% | 97.5% |
| 1 rating | 40 | 2019 (9) | 793 | 1890 | 2.5x / 1.1x | 0 | 0.0% | 47.5% |

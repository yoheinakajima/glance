# Write the answers or read them? Same frozen Qwen3-VL-4B, same images, same questions

Cold start per image, end to end from the image file, idle GPU. Writing = greedy generation of one JSON object with a token cap, no thinking. Reading = Glance (`glance decide`, uncalibrated), which also returns a probability for every answer.

| Request | images | write p50 ms (tokens) | read `fast2` p50 ms | read `ens4d` p50 ms | read is faster by | JSON failed to parse | invalid fields | written = read (valid fields) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 yes/no | 40 | 655 (8) | 196 | - | 3.3x | 19 | 47.5% | 100.0% |
| 1 pick-one | 40 | 424 (7) | 219 | 218 | 1.9x / 1.9x | 5 | 12.5% | 100.0% |
| 1 rating | 40 | 302 (6) | 250 | 613 | 1.2x / 0.5x | 26 | 65.0% | 14.3% |

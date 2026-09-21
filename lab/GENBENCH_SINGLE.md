# Write the answers or read them? Same frozen Qwen3-VL-4B, same images, same questions

Cold start per image, end to end from the image file, idle GPU. Writing = greedy generation of one JSON object with a token cap, no thinking. Reading = Glance (`glance decide`, uncalibrated), which also returns a probability for every answer.

| Request | images | write p50 ms (tokens) | read `fast2` p50 ms | read `ens4d` p50 ms | read is faster by | JSON failed to parse | invalid fields | written = read (valid fields) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 yes/no | 40 | 779 (7) | 328 | - | 2.4x | 0 | 0.0% | 100.0% |
| 1 pick-one | 40 | 998 (10) | 376 | 376 | 2.7x / 2.7x | 0 | 0.0% | 100.0% |
| 1 rating | 40 | 907 (8) | 440 | 1083 | 2.1x / 0.8x | 0 | 0.0% | 85.0% |

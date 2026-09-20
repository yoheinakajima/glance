# Write the answers or read them? Same frozen Qwen3-VL-4B, same images, same questions

Cold start per image, end to end from the image file, idle GPU. Writing = greedy generation of one JSON object with a token cap, no thinking. Reading = Glance (`glance decide`, uncalibrated), which also returns a probability for every answer.

| Request | images | write p50 ms (tokens) | read `fast2` p50 ms | read `ens4d` p50 ms | read is faster by | JSON failed to parse | invalid fields | written = read (valid fields) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 yes/no | 40 | 798 (7) | 338 | - | 2.4x | 0 | 0.0% | 100.0% |
| 5 mixed | 40 | 3478 (38) | 1001 | 2230 | 3.5x / 1.6x | 0 | 0.0% | 92.0% |
| 25 ratings | 40 | 20046 (221) | 3261 | 8380 | 6.1x / 2.4x | 0 | 0.0% | 59.0% |

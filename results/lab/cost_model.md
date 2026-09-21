# What 1,000 ratings cost: measured seconds, assumed prices

Seconds are measured (`lab/PACKING.json`, cold start, Apple M5 laptop, Qwen3-VL-4B). Every dollar figure is arithmetic on the assumptions listed at the end; none is a measurement. API figures are list-price UPPER estimates from `glance baseline --estimate-only` (no call was made).

| Configuration (Qwen3-VL-4B + Glance, self-hosted) | seconds per 1,000 ratings | electricity only | rented cloud GPU at laptop speed | laptop amortized |
| --- | --- | --- | --- | --- |
| ens4d, one rubric per image | 1089 | $0.0005 to $0.0073 | $0.159 to $0.243 | $0.023 to $0.138 |
| ens4d, 5 rubrics per image | 584 | $0.0002 to $0.0039 | $0.085 to $0.130 | $0.012 to $0.074 |
| ens4d, 25 rubrics per image | 341 | $0.0001 to $0.0023 | $0.050 to $0.076 | $0.007 to $0.043 |
| fast2 (no crop), 5 rubrics per image | 240 | $0.0001 to $0.0016 | $0.035 to $0.054 | $0.005 to $0.030 |
| fast2 (no crop), 25 rubrics per image | 133 | $0.0001 to $0.0009 | $0.019 to $0.030 | $0.003 to $0.017 |

| Hosted frontier API, one image per call, zero labels | list-price upper estimate per 1,000 ratings |
| --- | --- |
| anthropic/claude-opus-5 | $18.70 |
| openai/gpt-5.6 | $14.96 |
| gemini/gemini-3.1-pro-preview | $8.28 |
| anthropic/claude-haiku-4-5-20251001 | $3.74 |

## The same open model with and without Glance: write one JSON object, or read the answers (per 1,000 images, rented cloud GPU at laptop speed)

Self-hosted cost is GPU time, so the saving IS the time saving; seconds measured with the GPU otherwise idle (`lab/GENBENCH.json`, 40 images per shape). `fast2` is the two-pass rating read, `ens4d` the four-pass one; yes/no and pick-one are one pass either way.

| Request | without Glance: the model writes JSON | with Glance, read (`fast2`) | saving | with Glance, read (`ens4d`) | saving | written and read answers agree |
| --- | --- | --- | --- | --- | --- | --- |
| 1 yes/no | $0.117 to $0.178 (798 s) | $0.049 to $0.075 (338 s) | 58% | same as `fast2` (no rating in the request) | - | 100% |
| 5 mixed | $0.508 to $0.777 (3478 s) | $0.146 to $0.224 (1001 s) | 71% | $0.326 to $0.498 (2230 s) | 36% | 92% |
| 25 ratings | $2.929 to $4.481 (20046 s) | $0.476 to $0.729 (3261 s) | 84% | $1.224 to $1.873 (8380 s) | 58% | 59% |

Accuracy of the two on identical labeled items is in `results/lab/gen_accuracy.md`: identical on yes/no and pick-one; on zero-shot ratings the written answer is currently MORE accurate than the raw read (0.672 against 0.570), so for ratings the saving above buys speed, probabilities and the ability to fit, not zero-shot accuracy.

Reading: self-hosted costs between $0.0001 to $0.2435 per 1,000 ratings depending on how it is counted (electricity only ... an on-demand cloud GPU that is no faster than the laptop), against $3.74 to $18.70 for the APIs. For a single rubric per image on a rented GPU the APIs are 15 to 118 times more expensive per rating. Not included: engineering time, and the one-time labeling of about 32 images per rubric (2 to 5 minutes of a person's time). The APIs need no labels and no setup; whether they are more or less ACCURATE on these rubrics is the head-to-head that is still open.

Assumptions: laptop_power_watts = [15, 60]; electricity_usd_per_kwh = [0.1, 0.4]; cloud_gpu_usd_per_hour = [0.526, 0.8048]; cloud_throughput_vs_laptop = [1.0, 1.0]; laptop_price_usd = [2000, 4000]; laptop_amortization_hours = [8760, 26280]; labels_per_rubric = 32; seconds_per_label = [3, 10]. Cloud prices: AWS on-demand, us-east-1, g4dn.xlarge (T4) and g6.xlarge (L4), checked 2026-09-20 (instances.vantage.sh / cloudprice.net listings). Laptop power draw was not measured. TypeSafe's Jev is not listed: it does not accept images.

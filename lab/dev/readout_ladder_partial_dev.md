# Readout ladder on `ladders` (DEV: calibration split only (first half fit, second half judged))

Registered in `lab/NOTES.md` entry 20 (E1). Same frozen Qwen3-VL-4B, same images, same forward passes; only the readout differs.

| Readout | blur | noise | mean accuracy | within one | MAE | NLL | mean ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ens4d` + matrix scaling on 4 x K member logits (shipped Glance) | 0.868 | 0.872 | **0.870** | 1.000 | 0.187 | 0.324 | 0.057 |
| R4a: fitted readout on the hidden state of ONE `digits` pass | 0.952 | 0.940 | **0.946** | 1.000 | 0.085 | 0.200 | 0.033 |
| R4b: fitted readout on the four passes' hidden states | 0.956 | 0.960 | **0.958** | 1.000 | 0.053 | 0.114 | 0.017 |
| R5: fitted readout on member logits + hidden states | 0.956 | 0.960 | **0.958** | 1.000 | 0.050 | 0.099 | 0.021 |

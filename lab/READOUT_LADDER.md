# Readout ladder on `ladders` (fit on the calibration split, TEST split scored once)

Registered in `lab/NOTES.md` entry 20 (E1). Same frozen Qwen3-VL-4B, same images, same forward passes; only the readout differs.

| Readout | blur | noise | jpeg | exposure | resolution | mean accuracy | within one | MAE | NLL | mean ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ens4d` + matrix scaling on 4 x K member logits (shipped Glance) | 0.888 | 0.864 | 0.772 | 0.923 | 0.890 | **0.867** | 0.998 | 0.182 | 0.325 | 0.034 |
| R4a: fitted readout on the hidden state of ONE `digits` pass | 0.982 | 0.942 | 0.968 | 0.943 | 0.992 | **0.965** | 1.000 | 0.053 | 0.096 | 0.011 |
| R4b: fitted readout on the four passes' hidden states | 0.990 | 0.962 | 0.986 | 0.951 | 0.980 | **0.974** | 1.000 | 0.035 | 0.075 | 0.010 |
| R5: fitted readout on member logits + hidden states | 0.986 | 0.970 | 0.986 | 0.957 | 0.984 | **0.977** | 1.000 | 0.032 | 0.060 | 0.006 |

## Labels needed (mean accuracy over scales; 10 draws per size; full = all calibration labels)

| Readout | n=16 | n=32 | n=64 | n=128 | full |
| --- | --- | --- | --- | --- | --- |
| `ens4d` | 0.832 | 0.855 | 0.861 | 0.866 | 0.867 |
| R4b | 0.605 | 0.799 | 0.904 | 0.935 | 0.974 |
| R5 | 0.826 | 0.884 | 0.914 | 0.945 | 0.977 |

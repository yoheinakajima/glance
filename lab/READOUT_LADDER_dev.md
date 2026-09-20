# Readout ladder on `ladders` (DEV: calibration split only (first half fit, second half judged))

Registered in `lab/NOTES.md` entry 20 (E1). Same frozen Qwen3-VL-4B, same images, same forward passes; only the readout differs.

| Readout | blur | noise | jpeg | exposure | resolution | mean accuracy | within one | MAE | NLL | mean ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ens4d` + matrix scaling on 4 x K member logits (shipped Glance) | 0.868 | 0.872 | 0.784 | 0.919 | 0.904 | **0.869** | 0.998 | 0.176 | 0.323 | 0.050 |
| R4a: fitted readout on the hidden state of ONE `digits` pass | 0.952 | 0.940 | 0.932 | 0.947 | 0.976 | **0.949** | 0.999 | 0.078 | 0.172 | 0.025 |
| R4b: fitted readout on the four passes' hidden states | 0.956 | 0.960 | 0.964 | 0.931 | 0.992 | **0.961** | 1.000 | 0.050 | 0.122 | 0.019 |
| R5: fitted readout on member logits + hidden states | 0.956 | 0.960 | 0.968 | 0.939 | 0.984 | **0.961** | 0.998 | 0.048 | 0.120 | 0.022 |

## Labels needed (mean accuracy over scales; 10 draws per size; full = all calibration labels)

| Readout | n=16 | n=32 | n=64 | n=128 | full |
| --- | --- | --- | --- | --- | --- |
| `ens4d` | 0.831 | 0.850 | 0.861 | 0.868 | 0.869 |
| R4b | 0.621 | 0.738 | 0.894 | 0.937 | 0.961 |
| R5 | 0.819 | 0.866 | 0.913 | 0.945 | 0.961 |

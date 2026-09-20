# Adaptive compute for ratings: `fast2` first, the magnified-crop passes only when unsure

Threshold chosen on the calibration split (out-of-fold): confidence < 0.55 escalates (dev: 26.6% escalated, accuracy 0.870 against 0.875 for always-`ens4d`).

| Test split, scored once | mean accuracy | images escalated | mean forward passes |
| --- | --- | --- | --- |
| always `fast2` | 0.833 | 0% | 2.00 |
| **adaptive** | **0.863** | 28.4% | 2.57 |
| always `ens4d` | 0.867 | 100% | 4.00 |

Share of the `ens4d`-over-`fast2` gain kept: 88%. Per scale (escalated, accuracy): blur 18%, 0.892; exposure 11%, 0.916; jpeg 69%, 0.770; noise 22%, 0.860; resolution 22%, 0.878.

# Geometry probes, Qwen3-VL-4B: token read, written answer, and a linear probe on the hidden state (E26)

Same 150 images per set. The probe is 5-fold cross-validated on those images (PCA 32 + logistic regression), so it sees labels; the other two routes see none.

| Set | chance | token read (Glance) | written, strict | written, lenient | hidden-state probe, CV | probe minus read, points |
| --- | --- | --- | --- | --- | --- | --- |
| stripe direction (1 of 4) | 0.25 | 0.653 [0.573, 0.727] | 0.753 [0.686, 0.820] | 0.753 [0.687, 0.820] | 0.993 [0.980, 1.000] | +34.0 |
| largest of four shapes (1 of 4) | 0.25 | 0.520 [0.440, 0.600] | 0.520 [0.440, 0.600] | 0.520 [0.440, 0.600] | 0.480 [0.400, 0.560] | -4.0 |
| count the balls (1 of 8) | 0.12 | 0.913 [0.867, 0.953] | 0.860 [0.800, 0.913] | 0.860 [0.807, 0.913] | 0.953 [0.913, 0.980] | +4.0 |

Registered predictions (`lab/NOTES.md` entry 61):

- H65: written within 5 points of the read on all three sets: NOT SUPPORTED
- H66a: the hidden-state probe beats the token read by at least 15 points on stripes: SUPPORTED
- H66b: and by less than 10 points on the largest shape: SUPPORTED

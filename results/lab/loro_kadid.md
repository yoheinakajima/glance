# Leave one rubric out on `kadid` (23 distortions, 5 levels, same question template)

| Setting | labels from the target rubric | mean accuracy | within one level |
| --- | --- | --- | --- |
| raw: 0 labels, nothing | 0 | **0.328** | 0.738 |
| BC: unlabeled images of the target only | 0 | **0.392** | 0.803 |
| LORO: map fit on the other distortions, 0 labels and 0 images of the target | 0 | **0.350** | 0.671 |
| LORO + BC: the same on features z-scored per distortion with unlabeled images | 0 | **0.433** | 0.831 |
| per-rubric fit on the target's own labels (reference) | about 205 | **0.527** | 0.880 |

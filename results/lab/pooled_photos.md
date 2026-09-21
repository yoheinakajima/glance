# Three fresh photo sets pooled (E23, `lab/NOTES.md` entry 54; NOT blind: the per-set results were seen before the rule was fixed)

Items: the test halves the hosted models were asked, the open model on the same items; a failed hosted call counts as wrong. 95% intervals from a bootstrap stratified by photo set that resamples photographs (all questions of a photo together). The difference column is paired on the same items.

## Yes/no: 541 items (Commons 131, iNaturalist, ten groups 200, iNaturalist, insect orders 210)

| System | pooled accuracy | equal weight per set | Commons | iNaturalist, ten groups | iNaturalist, insect orders | open model minus this system, points | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B, read | 0.939 [0.917, 0.959] | 0.938 | 0.931 | 0.940 | 0.943 | - | - |
| Qwen3-VL-4B, written | 0.945 [0.925, 0.963] | 0.943 | 0.931 | 0.945 | 0.952 | - | - |
| Qwen3-VL-2B, read | 0.904 [0.877, 0.930] | 0.908 | 0.939 | 0.915 | 0.871 | +3.5 [+1.5, +5.6] | - |
| Qwen3-VL-8B, read | 0.933 [0.910, 0.955] | 0.932 | 0.924 | 0.940 | 0.933 | +0.6 [-1.3, +2.4] | - |
| SmolVLM2-2.2B, read | 0.839 [0.810, 0.867] | 0.852 | 0.931 | 0.895 | 0.729 | +10.0 [+7.1, +12.8] | - |
| Qwen3-VL-2B, written | 0.584 [0.548, 0.621] | 0.593 | 0.664 | 0.495 | 0.619 | +35.5 [+31.8, +39.1] | - |
| Claude Opus 5 | 0.937 [0.914, 0.958] | 0.936 | 0.924 | 0.945 | 0.938 | +0.2 [-1.7, +2.1] | no difference detected, within 3 points |
| GPT-5.6 | 0.928 [0.904, 0.951] | 0.924 | 0.893 | 0.935 | 0.943 | +1.1 [-0.7, +3.0] | no difference detected, within 3 points |
| Gemini 3.1 Pro | 0.959 [0.941, 0.976] | 0.958 | 0.947 | 0.960 | 0.967 | -2.0 [-3.9, -0.2] | open model behind |
| Claude Haiku 4.5 | 0.839 [0.809, 0.869] | 0.852 | 0.939 | 0.830 | 0.786 | +10.0 [+7.3, +12.8] | open model ahead |
| GPT-5.6 Luna | 0.906 [0.881, 0.930] | 0.908 | 0.924 | 0.935 | 0.867 | +3.3 [+1.1, +5.7] | open model ahead |
| Gemini 3.1 Flash-Lite | 0.961 [0.944, 0.977] | 0.960 | 0.954 | 0.965 | 0.962 | -2.2 [-4.0, -0.6] | open model behind |

Hosted rows missing and counted wrong: .

## Pick-one: 270 items (Commons 65, iNaturalist, ten groups 100, iNaturalist, insect orders 105)

| System | pooled accuracy | equal weight per set | Commons | iNaturalist, ten groups | iNaturalist, insect orders | open model minus this system, points | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B, read | 0.933 [0.904, 0.959] | 0.924 | 0.862 | 0.920 | 0.990 | - | - |
| Qwen3-VL-4B, written | 0.930 [0.900, 0.959] | 0.925 | 0.892 | 0.930 | 0.952 | - | - |
| Qwen3-VL-2B, read | 0.907 [0.874, 0.941] | 0.896 | 0.815 | 0.930 | 0.943 | +2.6 [+0.0, +5.2] | - |
| Qwen3-VL-8B, read | 0.926 [0.893, 0.956] | 0.920 | 0.877 | 0.940 | 0.943 | +0.7 [-1.5, +3.0] | - |
| SmolVLM2-2.2B, read | 0.737 [0.685, 0.789] | 0.752 | 0.846 | 0.790 | 0.619 | +19.6 [+14.8, +24.4] | - |
| Qwen3-VL-2B, written | 0.841 [0.796, 0.885] | 0.839 | 0.831 | 0.810 | 0.876 | +9.3 [+5.5, +13.3] | - |
| Claude Opus 5 | 0.937 [0.907, 0.963] | 0.933 | 0.908 | 0.930 | 0.962 | -0.4 [-3.0, +2.2] | no difference detected, within 3 points |
| GPT-5.6 | 0.904 [0.867, 0.937] | 0.902 | 0.892 | 0.910 | 0.905 | +3.0 [+0.0, +6.3] | no difference detected |
| Gemini 3.1 Pro | 0.937 [0.907, 0.963] | 0.935 | 0.923 | 0.910 | 0.971 | -0.4 [-3.0, +2.2] | no difference detected, within 3 points |
| Claude Haiku 4.5 | 0.785 [0.733, 0.833] | 0.793 | 0.846 | 0.790 | 0.743 | +14.8 [+10.4, +19.3] | open model ahead |
| GPT-5.6 Luna | 0.881 [0.841, 0.919] | 0.884 | 0.908 | 0.860 | 0.886 | +5.2 [+1.5, +8.9] | open model ahead |
| Gemini 3.1 Flash-Lite | 0.933 [0.904, 0.963] | 0.930 | 0.908 | 0.930 | 0.952 | +0.0 [-3.0, +3.0] | no difference detected, within 3 points |

Hosted rows missing and counted wrong: .


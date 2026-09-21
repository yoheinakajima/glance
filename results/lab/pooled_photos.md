# Three fresh photo sets pooled (E23, `lab/NOTES.md` entry 54; NOT blind: the per-set results were seen before the rule was fixed)

Items: the test halves the hosted models were asked, the open model on the same items; a failed hosted call counts as wrong. 95% intervals from a bootstrap stratified by photo set. The difference column is paired on the same items.

## Yes/no: 541 items (Commons 131, iNaturalist, ten groups 200, iNaturalist, insect orders 210)

| System | pooled accuracy | equal weight per set | Commons | iNaturalist, ten groups | iNaturalist, insect orders | open model minus this system, points | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B, read | 0.939 [0.917, 0.957] | 0.938 | 0.931 | 0.940 | 0.943 | - | - |
| Qwen3-VL-4B, written | 0.945 [0.924, 0.963] | 0.943 | 0.931 | 0.945 | 0.952 | - | - |
| Qwen3-VL-2B, read | 0.904 [0.878, 0.928] | 0.908 | 0.939 | 0.915 | 0.871 | +3.5 [+1.7, +5.5] | - |
| Qwen3-VL-8B, read | 0.933 [0.911, 0.954] | 0.932 | 0.924 | 0.940 | 0.933 | +0.6 [-1.1, +2.2] | - |
| Claude Opus 5 | 0.937 [0.915, 0.957] | 0.936 | 0.924 | 0.945 | 0.938 | +0.2 [-1.7, +1.8] | indistinguishable, within 3 points |
| GPT-5.6 | 0.928 [0.906, 0.948] | 0.924 | 0.893 | 0.935 | 0.943 | +1.1 [-0.7, +3.1] | indistinguishable |
| Gemini 3.1 Pro | 0.959 [0.941, 0.976] | 0.958 | 0.947 | 0.960 | 0.967 | -2.0 [-3.7, -0.4] | open model behind |
| Claude Haiku 4.5 | 0.839 [0.808, 0.869] | 0.852 | 0.939 | 0.830 | 0.786 | +10.0 [+7.2, +12.8] | open model ahead |
| GPT-5.6 Luna | 0.906 [0.880, 0.930] | 0.908 | 0.924 | 0.935 | 0.867 | +3.3 [+1.1, +5.5] | open model ahead |
| Gemini 3.1 Flash-Lite | 0.961 [0.945, 0.976] | 0.960 | 0.954 | 0.965 | 0.962 | -2.2 [-3.9, -0.6] | open model behind |

Hosted rows missing and counted wrong: .

## Pick-one: 270 items (Commons 65, iNaturalist, ten groups 100, iNaturalist, insect orders 105)

| System | pooled accuracy | equal weight per set | Commons | iNaturalist, ten groups | iNaturalist, insect orders | open model minus this system, points | reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-VL-4B, read | 0.933 [0.904, 0.959] | 0.924 | 0.862 | 0.920 | 0.990 | - | - |
| Qwen3-VL-4B, written | 0.930 [0.900, 0.959] | 0.925 | 0.892 | 0.930 | 0.952 | - | - |
| Qwen3-VL-2B, read | 0.907 [0.874, 0.941] | 0.896 | 0.815 | 0.930 | 0.943 | +2.6 [+0.0, +5.2] | - |
| Qwen3-VL-8B, read | 0.926 [0.893, 0.956] | 0.920 | 0.877 | 0.940 | 0.943 | +0.7 [-1.5, +3.0] | - |
| Claude Opus 5 | 0.937 [0.907, 0.963] | 0.933 | 0.908 | 0.930 | 0.962 | -0.4 [-3.0, +2.2] | indistinguishable, within 3 points |
| GPT-5.6 | 0.904 [0.867, 0.937] | 0.902 | 0.892 | 0.910 | 0.905 | +3.0 [+0.0, +6.3] | indistinguishable |
| Gemini 3.1 Pro | 0.937 [0.907, 0.963] | 0.935 | 0.923 | 0.910 | 0.971 | -0.4 [-3.0, +2.2] | indistinguishable, within 3 points |
| Claude Haiku 4.5 | 0.785 [0.737, 0.833] | 0.793 | 0.846 | 0.790 | 0.743 | +14.8 [+10.4, +19.3] | open model ahead |
| GPT-5.6 Luna | 0.881 [0.841, 0.919] | 0.884 | 0.908 | 0.860 | 0.886 | +5.2 [+1.5, +8.9] | open model ahead |
| Gemini 3.1 Flash-Lite | 0.933 [0.904, 0.959] | 0.930 | 0.908 | 0.930 | 0.952 | +0.0 [-3.0, +3.0] | indistinguishable, within 3 points |

Hosted rows missing and counted wrong: .


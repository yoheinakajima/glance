# Tutorial: fit Glance to your own rubric

Glance is how you ask a frozen VLM for a score. It is not a VLM. You bring a model you already have and a few dozen
labeled images; `glance fit` gives you a small calibration file for one rubric. This walk-through uses the
license-clean `distort25` benchmark so that you can run it exactly as written. The outputs shown are from a real run
on an Apple M5 laptop with Qwen3-VL-4B (2026-09-20, `lab/NOTES.md` entries 15c and 18).

## 1. Write the rubric

A rubric is a question that names the image as `` `img0` `` and an ordered list of level descriptions, lowest first.

```json
{
  "instructions": "How strong is the JPEG compression artifacts (square blocks and ringing) in `img0`?",
  "criteria": ["Barely noticeable", "Slight", "Moderate", "Strong", "Very strong"]
}
```

(That is the generic wording our benchmark registered in advance, clumsy grammar included. Hand-written level texts
work better: the lab's tuned 4-level rubrics reached 0.772 to 0.924; this generic 5-level one is much harder.)

## 2. Label a few images

One folder per level, named by the level index (anything after `_` is ignored), or a JSONL / CSV with `image,level`.

```bash
uv run python -m glance.lab.distort25          # builds the benchmark from the Pets train split (about 700 MB, once)
uv run python tools/make_fit_demo.py --distortion jpeg --per-level 8
# demo/jpeg/labels/0_barely_noticeable/*.png ... demo/jpeg/labels/4_very_strong/*.png, plus demo/jpeg/try/
```

## 3. Fit

```bash
uv run glance fit --data demo/jpeg/labels --rubric demo/jpeg/rubric.json --prefix-cache --name demo-jpeg
```

```
rate_882964: ens4d calibration for a 5-level rubric, fit on 40 labeled images (per level: [8, 8, 8, 8, 8])
  5-fold cross-validation on those images: accuracy 0.550, within one level 0.850, mean abs. error 0.82 levels,
  ECE 0.274 (a perfectly calibrated model would measure about 0.237 at this sample size)
  saved to calibration/ratings/882964879604.json
```

84 seconds for 40 images while another job shared the GPU. The model was read four times per image; nothing in it
changed. Read the report honestly: with 8 labels per level on the hardest distortion we have, exact accuracy is 0.550
and 0.850 of answers are within one level; the ECE cannot be told apart from a calibrated model's at 40 images. More
labels help (the lab's curve flattens at about 8 per level for 4-level hand-written rubrics, and keeps improving
here). `fit` estimates the sharpness of the probabilities on held-out folds, because at this size a train-fit
sharpness is badly overconfident.

## 4. Use it

Send the same instructions and criteria. `calibrated: "auto"` applies your calibration when one matches and says so.

```bash
uv run glance score demo/jpeg/try/level4_train_Ragdoll_101_L4.png --rubric demo/jpeg/rubric.json --prefix-cache
```

```python
from glance import Glance
g = Glance()
a = g.score("demo/jpeg/try/level4_train_Ragdoll_101_L4.png", rubric["instructions"], rubric["criteria"])
# score 3.65 (expected level), probabilities 0.01 0.01 0.09 0.11 0.79, confidence 0.55, calibration 'rate_882964'
```

On the ten held-out images of the demo folder the top level was right on 6 of 10 and within one level on all 10,
with low confidences (0.07 to 0.55) on the middle levels. Use the expected level and the confidence, not only the top
level: route low-confidence items to a person or a bigger model.

## 5. Ask many things about one image

All questions of a request share the image prefill (one per view), so extra rubrics are cheap:

```python
g.ask("photo.png", {
    "jpeg":    {"type": "score", **rubric},
    "usable":  {"type": "noul", "instructions": "Is `img0` good enough to use on a product page?"},
    "subject": {"type": "choice", "instructions": "What animal is in `img0`?", "criteria": {"dog": None, "cat": None, "other": None}},
})
# {'jpeg': {'score': 0.77, ...}, 'usable': {'noul': 0.91}, 'subject': {'choice': 'cat', 'confidence': 1.0}}
# usage: 8 forward passes, 2.2 s on a shared GPU
```

Measured on an idle GPU (`docs/paper/RESULTS_LAB.md`, section 9): one rating of a fresh image about 1.1 s; five
rubrics about 0.6 s each; 25 about 0.34 s each. `score_method: "digits"` (one pass, no magnified crop) costs about
0.36 s for a fresh image and reached 0.814 mean accuracy on the lab scales against 0.867 for the default `ens4d`.

## What a calibration is tied to

The exact rubric text, the model id and revision, the rating prompt version, the method, and the image-token budget.
Change any of them and you need a new `fit`; `calibrated: true` refuses to answer with a calibration that does not
match (HTTP 409), `"auto"` answers uncalibrated and warns. A calibration fit for one rating dimension does not work
for another; we measured that, so there is no pooled fallback.

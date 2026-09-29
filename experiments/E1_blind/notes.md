# E1 — Blind baseline

> Fill sections 1–4 and commit them BEFORE training the blind model (see `experiments/TEMPLATE_notes.md`).

## 1. Question
Does my model actually use the image?

## 2. Hypothesis and prediction (written before running)
I expect a blind model (no image, just zeros) to do much worse than my real model on
color and shape, because without the image it has no way to know what color or shape
is actually in the picture — it can only guess based on which words were common in
training.

My predictions:
- Exact match: around 12% (my real model got 70.6%)
- Color accuracy: around 30% (my real model got 73.8%)
- Shape accuracy: around 30% (my real model got 74.6%)
- n_objects accuracy: around 70% (my real model got 100%)
- Letter accuracy (teacher-forced): around 87% (my real model got 98.8%)

I expect letter accuracy to stay high even for the blind model, because once it has
generated the first letter or two correctly (even by luck), spelling the rest of a
known word like "circle" or "triangle" doesn't need the image — it just needs to know
English/the vocabulary. So letter accuracy measures spelling, not seeing.

I expect n_objects to be higher than chance because two-object words are much longer
than one-object words, so the model might learn "long word" vs "short word" just from
word length patterns, without needing the image.

If I'm wrong and the blind model scores close to my real model on color/shape, that would
mean my real model isn't really using the image for those — it would be guessing from
letter/spelling patterns instead, which would be a serious problem.
## 3. Baseline
`configs/baseline.yaml` (same seeds).

## 4. One variable changed
`data.image_mode: zeros` (`configs/blind.yaml`). Everything else identical.

## 5. Repetitions
_Seeds, hardware, commands (train + evaluate)._

## 6. Result
_Per-attribute accuracy, exact match and (teacher-forced) letter accuracy, blind vs baseline, mean ± std._

## 7. Interpretation
_What is the gap between letter accuracy and attribute accuracy, and what does it teach?_

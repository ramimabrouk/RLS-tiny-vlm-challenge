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
Seed: 0 (will add seeds 1 and 2 for the mean ± std comparison, matching the baseline).
Hardware: Google Colab, Tesla T4 GPU.
Commands:
    python -m src.train --config configs/blind.yaml --seed 0 --device cuda --out-dir runs/blind_seed0 --resume
    python -m src.evaluate --checkpoint runs/blind_seed0/best.pt --split test --device cuda

## 6. Result

Seed 0 only so far (mean ± std pending seeds 1–2).

| Metric | Real model (baseline) | Blind model (zeros) |
|---|---|---|
| Exact match | 70.63% ± 0.28% | 1.65% |
| Color accuracy | 73.84% ± 0.54% | 20.10% |
| Shape accuracy | 74.64% ± 0.61% | 20.63% |
| Relation accuracy | 43.14% ± 0.64% | 0.00% |
| n_objects accuracy | 100% | 49.00% |
| Letter accuracy (teacher-forced) | 98.83% | 87.89% |

## 7. Interpretation

Letter accuracy dropped only modestly (98.83% -> 87.89%), confirming the hypothesis:
spelling a known word does not require seeing the image, so this metric stays high even
for a model that is completely blind. This is the clearest evidence that letter accuracy
measures "can it spell," not "can it see," and should never be used alone to judge whether
a model understands an image.

Every other metric collapsed far more than predicted, and the pattern is not uniform: the
FIRST object's color and shape accuracy (30.4% and 31.2%) landed close to my 30% prediction,
but relation, and every attribute of the SECOND object, dropped to exactly 0%. The blind
model apparently learned to produce a plausible-looking first object from word-frequency
patterns alone (e.g. "red" and "circle" are common early tokens), but never learned to
produce a correct second object or relation without any image signal to work from. This
is why my n_objects prediction (70%) was far too optimistic: the real result (49%) is
indistinguishable from a coin flip, meaning the model cannot reliably tell whether a scene
has one or two objects without the image, even though two-object words are much longer.

Exact match (1.65%) is consistent with this: since roughly half of all scenes have two
objects, and the blind model gets 0% of every two-object attribute right, only single-object
scenes contribute to correct exact matches at all, and even those require the first three
attributes (size, color, shape) all correct simultaneously.

This result directly supports the finding from test_heldout: the real (non-blind) model
clearly relies on the image for shape and color, since removing the image drops those
metrics from ~74% to ~20-31%, and drops relation and second-object attributes to exactly
zero. The earlier test_heldout finding (that shape accuracy specifically collapses on novel
color-shape pairs) is therefore not evidence that the model ignores the image — it is
evidence of a narrower shortcut (using color to guess shape) layered on top of a model that
genuinely does use the image for its core predictions. A model with zero image signal at
all performs dramatically worse across the board, which rules out the more serious
possibility that the real model was ignoring the image entirely.
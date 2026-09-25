# E1 — Blind baseline

> Fill sections 1–4 and commit them BEFORE training the blind model (see `experiments/TEMPLATE_notes.md`).

## 1. Question
Does my model actually use the image?

## 2. Hypothesis and prediction (written before running)
_Your prediction, with numbers: what per-attribute accuracy and what letter accuracy do you expect from a model that
cannot see the image, and why? If you are wrong, what will you see?_

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

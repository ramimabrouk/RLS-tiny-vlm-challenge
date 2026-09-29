# AI usage

## Tools

- Claude (Anthropic), via the claude.ai chat interface.

## What the tool produced (first versions)

Claude wrote the first version of the code in this repository: `src/tokenizer.py`, `src/data.py`, `src/utils.py`,
`src/model/*`, `src/train.py`, `src/evaluate.py`, `src/generate.py`, `src/aggregate.py`, `tests/*`, `configs/*`, the
E0 script, the benchmark scripts (`benchmarks/*`), `docs/DESIGN.md`, the Colab notebook and the README skeleton.
Claude also ran the tests, a short debug training run and the data generator in its own sandbox (CPU only) to check
that the code works. Claude did **not** write hypotheses, experimental results, interpretations or the report.

## What I did myself

- Set up the local environment (Python 3.12 venv), cloned the RLS template repo to
  retrieve `generate_data.py`, and fixed a broken folder structure (duplicate nested
  copies of the project, loose duplicate files at the repo root) that was causing
  pytest collection errors.
- Set up and debugged Google Colab for GPU training: mounted Google Drive for
  checkpoint persistence, verified CUDA availability, and ran the full training
  pipeline (E0 sanity check, 3-seed baseline training, evaluation on test and
  test_heldout, blind baseline).
- Discovered and diagnosed the test_heldout generalization gap myself: shape accuracy
  drops sharply (74.6% -> 52.6%) on held-out color-shape combinations while color,
  size, and relation stay stable — indicating the model partly predicts shape from
  color correlation rather than true image geometry. This is documented in the
  README results table.
- Wrote the E1 blind-baseline hypothesis (experiments/E1_blind/notes.md, section 2)
  before training the blind model, predicting specific numbers for exact match, color,
  shape, n_objects, and letter accuracy, with reasoning for each.
- Compared my predictions against the actual blind-baseline results and wrote the
  interpretation (section 7): my letter-accuracy prediction was accurate, but I
  underestimated how completely the blind model would fail on second-object
  attributes and relations (both dropped to exactly 0%), and connected this result
  back to the test_heldout finding to argue the real model is genuinely using the
  image rather than ignoring it.
- Ran and verified every test suite (47 tests) both locally and on Colab to confirm
  consistent behavior across environments.
- Set up git and GitHub for the project from scratch, including cleaning up the
  repository structure before the first commit.

## What I checked

- Confirmed E0 (overfit one batch) drives loss from ln(27)=3.30 to near 0, verifying
  the pipeline (data, forward pass, backward pass, optimizer) has no bugs.
- Verified all 3 baseline seeds produce consistent results (exact match within 0.55
  points of each other) before trusting the mean ± std numbers.
- Verified the test_heldout variance across seeds (std of 13.71 on exact match, vs
  0.28 on regular test) and confirmed this correlates specifically with the shape
  metric's variance, supporting the color-shortcut interpretation rather than
  treating it as noise.
- Verified pytest passes identically on my local machine and on Colab (47 passed,
  same result) before trusting Colab-trained checkpoints.

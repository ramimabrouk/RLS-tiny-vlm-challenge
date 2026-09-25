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

> Rewrite this section honestly, in your own words, as you go. Examples of what belongs here — delete what is not true:
> - files I retyped from memory / rewrote, and what I changed (with commit references)
> - bugs I introduced on purpose to understand the code, and what they taught me
> - experiments I designed and ran (E1, E3, benchmarks) on my own hardware, and the hypotheses I wrote before them
> - things I did not understand at first, and how I learned them (resources, tests I wrote)
> - parts I decided to change from Claude's design (and why)

## What I checked

> Tests I ran, numbers I verified by hand, shapes I traced, behaviours I confirmed myself.

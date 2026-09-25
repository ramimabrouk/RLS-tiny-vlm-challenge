<p align="center">
  <img src="assets/tiny_vlm_logo_rounded.png" alt="Tiny VLM" width="200">
</p>

# RLS Entrance Challenge — Tiny Vision-Language Model

A ~1.1 M-parameter vision-language model: a CNN encodes a 64×64 image into 64 visual tokens, and a Transformer decoder
(handwritten multi-head attention) spells a one-word description of the scene, letter by letter
(e.g. `smallbluesquareleftoflargeyellowtriangle`).

## Results

> Fill in from **your own** runs (config + command below for each line). Mean ± std over seeds, hardware recorded.

| Experiment | Configuration | Metric | Result (mean ± std, n seeds) | Interpretation (1 sentence) |
|---|---|---|---|---|
| Main model | `configs/baseline.yaml` | Exact match, test | _todo_ | _todo_ |
| E1 blind | `configs/blind.yaml` | Attribute acc., test | _todo_ | _todo_ |
| S1 throughput | batch 64, _GPU model_ | images/s | _todo_ | _todo_ |

## Reproduce

```bash
pip install -r requirements.txt      # on Colab keep its preinstalled torch: pip install pyyaml tqdm matplotlib
python generate_data.py              # fixed seed 42, writes data/{train,val,test,test_heldout}.pt (do not change the splits)
python -m pytest -q                  # tokenizer, attention == F.scaled_dot_product_attention, shapes, parser, ...

# E0 - overfit one batch (sanity check)
python -m experiments.E0_overfit.overfit_one_batch --config configs/baseline.yaml

# Main model: 3 seeds
for s in 0 1 2; do
  python -m src.train    --config configs/baseline.yaml --seed $s
  python -m src.evaluate --checkpoint runs/baseline_seed$s/best.pt --split test
done
python -m src.aggregate "runs/baseline_seed*/results_test.json" exact_match attribute_acc.size attribute_acc.color attribute_acc.shape attribute_acc.relation

# E1 - blind baseline (same code, images replaced by zeros)
for s in 0 1 2; do
  python -m src.train    --config configs/blind.yaml --seed $s
  python -m src.evaluate --checkpoint runs/blind_seed$s/best.pt --split test
done

# S1 - throughput benchmark, then the plot
python -m benchmarks.S1_throughput.benchmark --devices cpu cuda --batch-sizes 1 16 64 256
python benchmarks/S1_throughput/plot.py

# Level 2 (optional)
python -m src.evaluate --checkpoint runs/baseline_seed0/best.pt --split test_heldout   # error analysis in results_test_heldout.json
python -m benchmarks.L2_profile.profile_step --device cuda --batch-size 64
python -m benchmarks.L2_memory.memory --batch-sizes 1 16 64 256
python -m benchmarks.L2_attention.bench_attention --device cuda
```

Every script takes `--device cpu|cuda|auto`. `python -m src.train ... --resume` continues from `runs/<name>_seed<k>/last.pt`
after a Colab disconnect. Quick CPU debugging: `--epochs 1 --subset 2000`.

## Repository layout

```
.
├── generate_data.py         # RLS-provided ShapeScenes generator — unmodified
├── configs/                 # baseline.yaml, blind.yaml
├── src/
│   ├── tokenizer.py  data.py  utils.py
│   ├── model/               # encoder.py, attention.py, decoder.py, vlm.py
│   ├── train.py  evaluate.py  generate.py  aggregate.py
├── tests/                   # attention equivalence, shapes, model behaviour, parser, tokenizer, data
├── experiments/             # E0_overfit/, E1_blind/ (notes.md), TEMPLATE_notes.md
├── benchmarks/              # S1_throughput/ (required), L2_profile/, L2_memory/, L2_attention/
├── docs/DESIGN.md           # architecture with shapes, design choices, self-test questions
├── notebooks/colab_run.ipynb  # only clones the repo and runs the scripts
└── report/
```

## AI usage

See [AI_USAGE.md](AI_USAGE.md).

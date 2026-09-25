# Design notes and study guide

Everything below is a *starting point to understand and then defend*. The brief leaves these choices to you
("you will justify them"): before the interview, be able to say for each one what you would change and what you expect
to see if you did.

## The pipeline with shapes (default config, batch B, word of L letters)

| Stage | Code | Shape |
|---|---|---|
| input images (uint8 -> float 0..1) | `data.normalize_images` | (B, 3, 64, 64) |
| CNN encoder, 3 blocks (each halves H, W) | `model/encoder.py` | (B, 128, 8, 8) |
| adapter: flatten spatial dims + Linear(128 -> d_model) | `Adapter` in `model/vlm.py` | (B, 64, 128) |
| letter embeddings | `nn.Embedding(27, d_model)` | (B, L, 128) |
| sequence = visual tokens ++ letters | `torch.cat` in `VLM.decode` | (B, 64 + L, 128) |
| + learned positional embeddings, 4 decoder blocks, final LayerNorm | `model/decoder.py` | (B, 64 + L, 128) |
| keep the last visual token and every letter | `h[:, N - 1:]` | (B, L + 1, 128) |
| linear head | `VLM.head` | (B, L + 1, 27) |

Training target for a word `w`: `w + <eos>` (length L + 1). The input letters are the same list without `<eos>`
(shifted right by one). The output at the last visual token predicts the first letter.

## Choices made in this code, the alternatives, and what would tell them apart

| Choice | Here | Alternative | Why it matters / how to test it |
|---|---|---|---|
| Visual tokens | 8x8 grid = 64 tokens | one pooled vector; 4x4 grid | relations ("leftof") need to know WHERE things are: a single pooled vector may lose that (E3 idea in the brief) |
| Mask among visual tokens | bidirectional ("prefix") | causal | with a causal mask, visual token i only sees tokens before it; `visual_attention` in the config switches it |
| Letters | causal, see all visual tokens | -- | mandatory: at generation time the future letters do not exist |
| Positional information | learned absolute embeddings for all 64 + L positions | sinusoidal, 2-D | without them the model cannot tell visual tokens apart by position |
| Norm placement | pre-LayerNorm | post-LN | pre-LN trains more stably without long warm-up |
| Normalisation in the CNN | BatchNorm | GroupNorm | BatchNorm behaves differently in train/eval mode (running statistics): why `greedy_generate` calls `model.eval()` |
| Loss | every letter position + `<eos>`, padding ignored | loss only on the attribute-first letters | see the letter-accuracy discussion in the brief |
| Decoding | greedy, re-runs the decoder on the whole prefix at each step | KV cache | no cache = the cost of step t grows with t; a cache is a natural Level 3 systems experiment |
| Optimiser | AdamW, lr 1e-3, 200 warm-up steps then cosine decay | SGD, constant lr | change one thing, 3 seeds |
| Size | ~1.1 M parameters (d_model 128, 4 heads, 4 layers) | wider/shallower at equal parameters | E3 idea in the brief |

## Where every requirement of the brief lives

| Requirement | File |
|---|---|
| tokenizer, padding | `src/tokenizer.py`, `src/data.py` (`collate`) |
| CNN encoder, adapter | `src/model/encoder.py`, `Adapter` in `src/model/vlm.py` |
| handwritten multi-head attention + equivalence test | `src/model/attention.py`, `tests/test_attention.py` |
| decoder block, positional embeddings | `src/model/decoder.py` |
| training loop, checkpoints, resume | `src/train.py` |
| greedy generation | `src/generate.py` |
| exact match, attribute accuracy, robust parser | `src/evaluate.py` |
| E0 overfit one batch | `experiments/E0_overfit/` |
| E1 blind baseline | `configs/blind.yaml` (+ `data.image_mode`) |
| S1 throughput | `benchmarks/S1_throughput/` |
| Level 2: step breakdown, memory, attention vs SDPA | `benchmarks/L2_profile/`, `L2_memory/`, `L2_attention/` |

## Questions to be able to answer aloud (without looking at the code)

Data and tokens
1. Why 27 tokens? Why no padding token? What does `ignore_index` do, and what would happen without it?
2. Why is `input_ids` the targets shifted by one? Draw it for the word `ab`.
3. Why are images kept as uint8 until they are on the GPU?

CNN
4. What is the output size of a 3x3 conv with padding 1? Of a 2x2 max-pool? How do you get from 64x64 to 8x8?
5. Why `bias=False` before BatchNorm? What differs between `model.train()` and `model.eval()`?

Attention and the Transformer
6. Write the attention formula. Why divide by sqrt(dh)? What is the shape of every tensor from `x` to the output?
7. Where does the mask go, before or after the softmax, and why? Why `-inf` and not 0?
8. Why do the projections split into heads with `view` + `transpose`, and why `contiguous()` when merging?
9. What does the residual connection give you? What does LayerNorm do, and why does a pre-LN block need a final one?
10. Why does memory grow with T^2 in attention? For B=64, H=4, T=100 how many floats is one attention matrix?
11. What do the positional embeddings do for the visual tokens? What would break without them?

Training and evaluation
12. Why should the loss start near ln(27) = 3.30? What if it starts at 30, or at 0.2?
13. What do `zero_grad`, `backward`, `step` each do? What does the scheduler change?
14. Why is letter accuracy misleading? Why does the blind baseline get a high letter accuracy and a low attribute accuracy?
15. Why can the parser not just search for the first exact substring `red`?

GPU
16. Why is the first iteration slower? Why is timing without `synchronize()` wrong?
17. Where does the GPU stop beating the CPU here, and why does a tiny model not use a GPU well at batch size 1?

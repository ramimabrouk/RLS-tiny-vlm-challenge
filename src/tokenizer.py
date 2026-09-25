"""Character-level tokenizer: 26 letters + <eos> (27 tokens).

Implements encode/decode between words and letter-index sequences, and
batches words of different lengths (padding + ignore_index for the loss).

Vocabulary (see Project Brief, section 4):
    ids 0..25 -> 'a'..'z'
    id  26    -> <eos>
There is no <bos> and no <pad> token: the model never *predicts* padding,
so padded positions in the targets use IGNORE_INDEX and are skipped by the loss.
"""

import string

import torch

LETTERS = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
EOS_ID = len(LETTERS)  # 26
VOCAB_SIZE = len(LETTERS) + 1  # 27
IGNORE_INDEX = -100  # default ignore_index of nn.CrossEntropyLoss

_CHAR_TO_ID = {c: i for i, c in enumerate(LETTERS)}


def encode(word: str, add_eos: bool = True) -> list[int]:
    """'abc' -> [0, 1, 2, 26]. Raises KeyError on any character outside a-z."""
    ids = [_CHAR_TO_ID[c] for c in word]
    if add_eos:
        ids.append(EOS_ID)
    return ids


def decode(ids) -> str:
    """[0, 1, 2, 26, 5] -> 'abc'. Stops at the first <eos>; ignores everything after it."""
    chars = []
    for i in ids:
        i = int(i)  # works for python ints and 0-d tensors
        if i == EOS_ID:
            break
        if not 0 <= i < EOS_ID:
            raise ValueError(f"invalid token id {i}")
        chars.append(LETTERS[i])
    return "".join(chars)


def pad_batch(sequences: list[list[int]], pad_value: int = IGNORE_INDEX) -> torch.Tensor:
    """Stack variable-length id lists into one (B, T_max) tensor, filling the gaps with pad_value."""
    t_max = max(len(s) for s in sequences)
    out = torch.full((len(sequences), t_max), pad_value, dtype=torch.long)  # (B, T_max)
    for row, seq in enumerate(sequences):
        out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out

import pytest
import torch

from src.tokenizer import EOS_ID, IGNORE_INDEX, VOCAB_SIZE, decode, encode, pad_batch


def test_vocab_size():
    assert VOCAB_SIZE == 27
    assert EOS_ID == 26


def test_encode_known_word():
    assert encode("abz") == [0, 1, 25, EOS_ID]
    assert encode("abz", add_eos=False) == [0, 1, 25]


def test_roundtrip():
    for word in ["largeredcircle", "smallbluesquareleftoflargeyellowtriangle"]:
        assert decode(encode(word)) == word


def test_decode_stops_at_first_eos():
    assert decode([0, 1, EOS_ID, 2, 3]) == "ab"


def test_decode_without_eos_reads_everything():
    assert decode([0, 1, 2]) == "abc"


def test_encode_rejects_bad_characters():
    with pytest.raises(KeyError):
        encode("Red circle")  # uppercase and a space are not in the vocabulary


def test_decode_rejects_bad_ids():
    with pytest.raises(ValueError):
        decode([0, IGNORE_INDEX])


def test_pad_batch():
    out = pad_batch([[0, 1, EOS_ID], [5, EOS_ID]])
    assert out.shape == (2, 3)
    assert out.tolist() == [[0, 1, EOS_ID], [5, EOS_ID, IGNORE_INDEX]]
    assert out.dtype == torch.long

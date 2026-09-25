import torch

from src.data import collate, normalize_images
from src.tokenizer import EOS_ID, IGNORE_INDEX, encode


def _fake_batch(words):
    return [(torch.zeros(3, 64, 64, dtype=torch.uint8), torch.tensor(encode(w))) for w in words]


def test_collate_shapes():
    images, input_ids, targets = collate(_fake_batch(["abc", "abcde"]))
    assert images.shape == (2, 3, 64, 64)
    assert targets.shape == (2, 6)  # longest word (5) + <eos>
    assert input_ids.shape == (2, 5)  # one shorter than targets


def test_collate_shift_and_padding():
    _, input_ids, targets = collate(_fake_batch(["ab", "abcd"]))
    # short word: a b <eos> then padding, ignored by the loss
    assert targets[0].tolist() == [0, 1, EOS_ID, IGNORE_INDEX, IGNORE_INDEX]
    # inputs are targets shifted right by one, with pads turned into a valid id (0)
    assert input_ids[0].tolist() == [0, 1, EOS_ID, 0]
    assert input_ids[1].tolist() == [0, 1, 2, 3]
    assert targets[1].tolist() == [0, 1, 2, 3, EOS_ID]
    assert (input_ids >= 0).all()


def test_normalize_images():
    x = torch.tensor([[[[0, 255]]]], dtype=torch.uint8)
    y = normalize_images(x)
    assert y.dtype == torch.float32
    assert y.flatten().tolist() == [0.0, 1.0]


def test_corrupt_images_modes():
    from src.data import corrupt_images

    x = torch.randint(1, 255, (6, 3, 4, 4), dtype=torch.uint8)
    assert torch.equal(corrupt_images(x, "none"), x)
    assert corrupt_images(x, "zeros").sum() == 0 and corrupt_images(x, "zeros").shape == x.shape
    shuffled = corrupt_images(x, "shuffle")  # same images, different order relative to the words
    assert torch.equal(shuffled.sum((1, 2, 3)).sort().values, x.sum((1, 2, 3)).sort().values)

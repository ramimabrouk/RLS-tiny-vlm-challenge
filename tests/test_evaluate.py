from pathlib import Path

import pytest
import torch

from src.evaluate import FIELDS, attribute_report, parse_word

DATA = Path("data/val.pt")


def test_parse_clean_words():
    assert parse_word("largeredcircle") == {**dict.fromkeys(FIELDS), "size1": "large", "color1": "red", "shape1": "circle"}
    p = parse_word("smallbluesquareleftoflargeyellowtriangle")
    assert (p["size1"], p["color1"], p["shape1"], p["relation"], p["size2"], p["color2"], p["shape2"]) == (
        "small", "blue", "square", "leftof", "large", "yellow", "triangle")


def test_parse_survives_misspellings():
    assert parse_word("largeredcirle")["shape1"] == "circle"  # missing letter
    assert parse_word("lagrgeredcircle")["size1"] == "large"  # swapped letters
    p = parse_word("smallbluesquarerightoflargegrenetriangle")  # broken color in the 2nd object
    assert p["relation"] == "rightof" and p["color2"] == "green" and p["shape2"] == "triangle"


def test_parse_truncated_and_empty_outputs():
    p = parse_word("smallblu")
    assert (p["size1"], p["color1"], p["shape1"]) == ("small", "blue", None)
    assert all(v is None for v in parse_word("").values())


def test_attribute_report_counts_the_right_things():
    refs = ["largeredcircle", "smallbluesquareleftoflargeyellowtriangle"]
    preds = ["largeredsquare", "smallbluesquareleftoflargeyellowtriangle"]  # wrong shape on the first
    rep = attribute_report(preds, refs)["attribute_acc"]
    assert rep["size"] == 1.0 and rep["color"] == 1.0 and rep["relation"] == 1.0
    assert rep["shape"] == pytest.approx(2 / 3)  # 3 shapes in the references, 1 of them wrong


@pytest.mark.skipif(not DATA.exists(), reason="run generate_data.py first")
def test_parser_reads_every_reference_word_perfectly():
    words = torch.load(DATA)["words"]
    for w in words:
        p = parse_word(w)
        rebuilt = p["size1"] + p["color1"] + p["shape1"]
        if p["relation"] is not None:
            rebuilt += p["relation"] + p["size2"] + p["color2"] + p["shape2"]
        assert rebuilt == w, w

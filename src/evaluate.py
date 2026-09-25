"""Evaluation: exact match, per-attribute accuracy (with a parser that survives typos), teacher-forced letter accuracy.

Usage (from the repo root):
    python -m src.evaluate --checkpoint runs/baseline_seed0/best.pt --split test
Writes results.json (metrics, error confusions, all predictions) next to the checkpoint.
"""

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch.nn as nn

from src.data import corrupt_images, make_loader, normalize_images
from src.generate import greedy_generate, load_checkpoint
from src.tokenizer import IGNORE_INDEX
from src.utils import get_device

SIZES = ["small", "large"]
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle", "cross"]
RELATIONS = ["leftof", "rightof", "above", "below"]
OBJECT_SLOTS = [("size", SIZES), ("color", COLORS), ("shape", SHAPES)]
MIN_RELATION_LEN = 5  # "above" / "below" are the shortest relations: with fewer characters left there is no 2nd object
MIN_SIMILARITY = 0.5  # a chunk that resembles no candidate at all counts as "missing"
FIELDS = ["size1", "color1", "shape1", "relation", "size2", "color2", "shape2"]
GROUPS = {  # attribute type -> the fields pooled into it
    "size": ["size1", "size2"],
    "color": ["color1", "color2"],
    "shape": ["shape1", "shape2"],
    "relation": ["relation"],
}


def _match_slot(word: str, pos: int, vocab: list[str]):
    """Best vocabulary entry for the characters starting at `pos`. Returns (entry or None, next position).

    Every candidate is compared with windows of length len-1, len, len+1 (so a dropped or doubled letter
    still lines up), scored by difflib's similarity ratio (1.0 = identical).
    """
    best_score, best_entry, best_end = -1.0, None, pos
    for entry in vocab:
        for length in range(max(1, len(entry) - 1), len(entry) + 2):
            chunk = word[pos : pos + length]
            if not chunk:
                continue
            score = SequenceMatcher(None, entry, chunk).ratio()
            if score > best_score + 1e-9:
                best_score, best_entry, best_end = score, entry, pos + len(chunk)
    if best_score < MIN_SIMILARITY:
        return None, min(pos + 1, len(word)) if pos < len(word) else pos
    return best_entry, best_end


def parse_word(word: str) -> dict:
    """'largeredcirle' -> {'size1': 'large', 'color1': 'red', 'shape1': 'circle', 'relation': None, ...}.

    The grammar is fixed: size color shape [relation size color shape]. Slots are read left to right and
    each one is matched against ITS OWN vocabulary, so a misspelling inside one part cannot shift the parts
    after it out of place (as it would if we searched for the first exact substring).
    """
    out = dict.fromkeys(FIELDS)
    pos = 0
    for slot, vocab in OBJECT_SLOTS:
        out[f"{slot}1"], pos = _match_slot(word, pos, vocab)
    if len(word) - pos >= MIN_RELATION_LEN:
        out["relation"], pos = _match_slot(word, pos, RELATIONS)
        for slot, vocab in OBJECT_SLOTS:
            out[f"{slot}2"], pos = _match_slot(word, pos, vocab)
    return out


def attribute_report(preds: list[str], refs: list[str]) -> dict:
    """Accuracy per field / per attribute type, plus the most frequent (reference -> prediction) confusions."""
    correct, total = Counter(), Counter()
    confusions = {f: Counter() for f in FIELDS + ["pair1", "pair2"]}
    n_objects_ok = 0
    for pred, ref in zip(preds, refs):
        p, r = parse_word(pred), parse_word(ref)
        n_objects_ok += (p["relation"] is None) == (r["relation"] is None)
        for f in FIELDS:
            if r[f] is None:  # the reference has no such part (single-object scenes)
                continue
            total[f] += 1
            correct[f] += p[f] == r[f]
            if p[f] != r[f]:
                confusions[f][(r[f], p[f])] += 1
        for i in ("1", "2"):  # joint color+shape: the pairs that are held out in test_heldout
            if r["color" + i] is not None:
                if (p["color" + i], p["shape" + i]) != (r["color" + i], r["shape" + i]):
                    confusions["pair" + i][
                        (f"{r['color' + i]} {r['shape' + i]}", f"{p['color' + i]} {p['shape' + i]}")
                    ] += 1
    n = len(refs)
    acc = {f: correct[f] / total[f] for f in FIELDS}
    for group, fields in GROUPS.items():
        acc[group] = sum(correct[f] for f in fields) / sum(total[f] for f in fields)
    acc["n_objects"] = n_objects_ok / n
    top = {f: [[a, b, c] for (a, b), c in cnt.most_common(8)] for f, cnt in confusions.items()}
    return {"attribute_acc": acc, "top_confusions": top}


@torch.no_grad()
def teacher_forced(model, loader, device, image_mode="none"):
    """Validation loss and LETTER accuracy with the TRUE previous letters as input (one forward pass per batch).

    Letter accuracy looks high even for a model that ignores the image: once 'c' is known, 'ircle' can be guessed
    from spelling alone. It measures spelling, not vision -- compare it with the attribute accuracies.
    """
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    loss_sum, n_correct, n_tokens = 0.0, 0, 0
    for images, input_ids, targets in loader:
        images = normalize_images(corrupt_images(images, image_mode).to(device))
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(images, input_ids)  # (B, T, 27)
        loss_sum += criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        valid = targets != IGNORE_INDEX
        n_correct += ((logits.argmax(-1) == targets) & valid).sum().item()
        n_tokens += valid.sum().item()
    model.train(was_training)
    return loss_sum / n_tokens, n_correct / n_tokens


@torch.no_grad()
def evaluate_generation(model, split, device, image_mode="none", data_dir="data", batch_size=250, limit=None):
    """Greedy-decode every image of a split. Returns (predictions, references)."""
    loader = make_loader(split, batch_size, shuffle=False, data_dir=data_dir, limit=limit)
    preds = []
    for images, _, _ in loader:
        images = normalize_images(corrupt_images(images, image_mode).to(device))
        preds += greedy_generate(model, images)
    return preds, loader.dataset.words


def full_evaluation(model, split, device, image_mode="none", data_dir="data", limit=None) -> dict:
    preds, refs = evaluate_generation(model, split, device, image_mode, data_dir, limit=limit)
    val_loss, letter_acc = teacher_forced(model, make_loader(split, 250, False, data_dir, limit=limit), device, image_mode)
    report = attribute_report(preds, refs)
    return {
        "split": split,
        "n_samples": len(refs),
        "exact_match": sum(p == r for p, r in zip(preds, refs)) / len(refs),
        "letter_acc_teacher_forced": letter_acc,
        "loss": val_loss,
        **report,
        "predictions": [{"ref": r, "pred": p} for p, r in zip(preds, refs)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "test_heldout"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples (debugging)")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    result = full_evaluation(model, args.split, device, cfg["data"].get("image_mode", "none"), args.data_dir, args.limit)

    out = Path(args.checkpoint).parent / f"results_{args.split}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.split}: exact match {result['exact_match']:.4f} | letter acc (teacher-forced) "
          f"{result['letter_acc_teacher_forced']:.4f}")
    print("attribute accuracy:", {k: round(v, 4) for k, v in result["attribute_acc"].items()})
    print(f"saved -> {out}")

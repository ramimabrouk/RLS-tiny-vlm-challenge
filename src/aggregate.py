"""Mean and spread of evaluation metrics over several seeds (for the results table).

Usage (quote the pattern so that Python, not the shell, expands it):
    python -m src.aggregate "runs/baseline_seed*/results_test.json" exact_match attribute_acc.color attribute_acc.relation
Nested keys use dots. The std is the sample standard deviation (n - 1); with n = 1 it is reported as n/a.
"""

import glob
import json
import statistics
import sys


def get(d: dict, dotted: str):
    for key in dotted.split("."):
        d = d[key]
    return d


if __name__ == "__main__":
    pattern, metrics = sys.argv[1], sys.argv[2:]
    files = sorted(glob.glob(pattern))
    if not files or not metrics:
        raise SystemExit(__doc__)
    results = [json.load(open(f)) for f in files]
    print(f"{len(files)} runs: {', '.join(files)}")
    for m in metrics:
        vals = [get(r, m) for r in results]
        std = f"{statistics.stdev(vals):.4f}" if len(vals) > 1 else "n/a"
        print(f"{m:28s} {statistics.mean(vals):.4f} +- {std}   (n={len(vals)})   values: {[round(v, 4) for v in vals]}")

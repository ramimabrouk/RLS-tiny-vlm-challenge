"""Plot timings.csv: mean images/s (error bars = std over repeats) against batch size, one line per device."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
groups = defaultdict(list)
with open(folder / "timings.csv") as f:
    for row in csv.DictReader(f):
        groups[(row["device"], int(row["batch_size"]))].append(float(row["images_per_sec"]))

fig, ax = plt.subplots(figsize=(6, 4))
for device in sorted({d for d, _ in groups}):
    sizes = sorted(bs for d, bs in groups if d == device)
    means = [statistics.mean(groups[(device, bs)]) for bs in sizes]
    stds = [statistics.stdev(groups[(device, bs)]) if len(groups[(device, bs)]) > 1 else 0.0 for bs in sizes]
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=device)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("batch size")
ax.set_ylabel("training throughput (images / s)")
ax.set_title("S1: training throughput vs batch size")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(folder / "throughput.png", dpi=120)
print(f"saved {folder / 'throughput.png'}")

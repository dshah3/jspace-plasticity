"""Generate blog-ready fresh-lens figures and their exact count table."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--audit", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
d = json.loads(a.audit.read_text())
a.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
table = []
for ax, cohort, title in zip(
    axes,
    ["pooled", "screen", "transfer"],
    ["Task cohort (129)", "Screen split (65)", "Transfer cohort (49)"],
    strict=True,
):
    x = np.arange(3)
    for offset, lens, color, label in [
        (-0.18, "published", "#8FA6B8", "Published lens"),
        (0.18, "own_fresh", "#225DB0", "Own fresh lens"),
    ]:
        counts = []
        for model in ["base", "primary", "replicate"]:
            key = (
                model
                + "-"
                + (
                    "published"
                    if lens == "published"
                    else ("base_fresh" if model == "base" else model + "_fresh")
                )
            )
            r = d["counts"][key][cohort]
            counts.append(r["jspace"])
            table.append(
                dict(
                    model=model,
                    lens=lens,
                    cohort=cohort,
                    rows=r["rows"],
                    correct=r["jspace"],
                    accuracy=r["jspace"] / r["rows"],
                )
            )
        n = r["rows"]
        bars = ax.bar(
            x + offset, np.array(counts) / n * 100, width=0.34, color=color, label=label
        )
        for b, count in zip(bars, counts, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 2,
                str(count),
                ha="center",
                fontsize=10,
            )
    ax.set(
        xticks=x,
        xticklabels=["Base", "Recovered\nseed 1", "Recovered\nseed 2"],
        ylim=(0, 112),
        yticks=[0, 25, 50, 75, 100],
        title=title,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
axes[0].set_ylabel("Corrected-intervention accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    frameon=False,
)
fig.suptitle("Recovery survives refitting the lens", fontsize=17, y=1.02)
fig.text(
    0.015,
    0.01,
    "Labels are correct counts. Fresh lenses: identical 500-prompt generic-text "
    "fit recipe.\nFiltered, previously explored cohorts; screen is part of the "
    "task cohort. Sequential projection and online output protection remain.",
    fontsize=9,
    color="#444444",
)
fig.tight_layout(rect=(0, 0.11, 1, 0.87))
for ext in ["png", "svg", "pdf"]:
    fig.savefig(a.output / f"fresh_lens_recovery.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
with (a.output / "fresh_lens_counts.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

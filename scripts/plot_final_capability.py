# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
# ]
# ///
"""Plot audited final experiment counts without running any model."""

import argparse
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
names = [
    "j-full-s100-primary",
    "j-full-s100-replicate",
    "random-full-s100",
    "sham-full-s100",
]
labels = [
    "Base",
    "J-trained\nseed 1",
    "J-trained\nseed 2",
    "Random-\ntrained",
    "Sham-\ntrained",
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
for ax, cohort, title in zip(
    axes,
    ["pooled", "screen", "transfer"],
    [
        "Historical task cohort · 129 prompts",
        "Screen split · 65 prompts",
        "Separate transfer · 49 prompts",
    ],
    strict=True,
):
    base = d["arms"][names[0]]["counts"][cohort]
    n = base["rows"]
    counts = [base["initial"]] + [
        d["arms"][name]["counts"][cohort]["terminal"] for name in names
    ]
    x = np.arange(5)
    for offset, key, color, label in [
        (-0.19, "jspace", "#246BCE", "Corrected intervention"),
        (0.19, "clean", "#A6B8A9", "Clean"),
    ]:
        vals = [c[key] for c in counts]
        bars = ax.bar(
            x + offset, np.array(vals) / n * 100, width=0.35, color=color, label=label
        )
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 1.4,
                str(v),
                ha="center",
                fontsize=8,
            )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 113),
        yticks=[0, 25, 50, 75, 100],
        title=title,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
axes[0].set_ylabel("Exact accuracy (%) · labels are correct counts")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.93),
    frameon=False,
)
fig.suptitle(
    "Recovery during continued corrected J-space intervention", fontsize=16, y=1.01
)
fig.text(
    0.02,
    0.015,
    "LR 1e-6 · 100 steps · published base lens · sequential projection · "
    "filtered, previously explored cohorts\n"
    "Validation selected LR/duration; screen is part of the 129-prompt cohort. "
    "No fresh-lens result for these checkpoints.",
    fontsize=9,
    color="#444444",
)
fig.tight_layout(rect=(0, 0.10, 1, 0.88))
a.output.mkdir(parents=True, exist_ok=True)
for ext in ["png", "pdf", "svg"]:
    fig.savefig(a.output / f"final_capability.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)

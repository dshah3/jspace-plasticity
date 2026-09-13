"""Render a direct clean/damaged/recovered activation-CKA comparison.

This is CPU-only. It consumes the hash-verified CKA matrices produced by the
completed recovery route-map evaluation and does not load or mutate a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LESION_LAYERS = (16, 18, 19, 20, 21, 22)
BLOCK_BOUNDARIES = (2.5, 19.5)


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def _mean_absolute_off_diagonal(left: np.ndarray, right: np.ndarray) -> float:
    mask = ~np.eye(left.shape[0], dtype=bool)
    return float(np.abs(left - right)[mask].mean())


def render(source: Path, output: Path, *, dpi: int = 220) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    with np.load(source) as payload:
        clean = payload["task_base_clean_within"]
        damaged = payload["task_base_lesion_within"]
        recovered = payload["task_recovered_lesion_within"]
    shapes_match = damaged.shape == clean.shape and recovered.shape == clean.shape
    if clean.shape != (32, 32) or not shapes_match:
        raise ValueError("route-map CKA matrices must all have shape [32, 32]")

    damage_delta = damaged - clean
    recovery_delta = recovered - damaged
    difference_bound = max(
        float(np.quantile(np.abs(np.stack((damage_delta, recovery_delta))), 0.99)),
        1e-4,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 5, figsize=(19.2, 4.35), constrained_layout=True)
    states = (
        (clean, "Clean base\nlesion off"),
        (damaged, "Damaged base\nlesion on"),
        (recovered, "Recovered (160 steps)\nlesion on"),
    )
    state_image = None
    for axis, (matrix, title) in zip(axes[:3], states, strict=True):
        state_image = axis.imshow(
            matrix, origin="lower", vmin=0.55, vmax=1.0, cmap="viridis"
        )
        axis.set_title(title)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Layer")
        axis.set_xticks((0, 5, 10, 15, 20, 25, 31))
        axis.set_yticks((0, 5, 10, 15, 20, 25, 31))
        for boundary in BLOCK_BOUNDARIES:
            axis.axhline(boundary, color="white", linewidth=0.9, alpha=0.85)
            axis.axvline(boundary, color="white", linewidth=0.9, alpha=0.85)
        axis.text(
            0.02,
            0.02,
            f"mean off-diagonal CKA\n{_off_diagonal_mean(matrix):.3f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
    if state_image is None:
        raise RuntimeError("state heatmaps were not rendered")
    figure.colorbar(
        state_image,
        ax=axes[:3],
        label="Within-state activation CKA",
        shrink=0.76,
        pad=0.012,
    )

    difference_image = None
    differences = (
        (
            damage_delta,
            "Lesion effect\n(damaged − clean)",
            _mean_absolute_off_diagonal(damaged, clean),
        ),
        (
            recovery_delta,
            "Recovery effect\n(recovered − damaged)",
            _mean_absolute_off_diagonal(recovered, damaged),
        ),
    )
    for axis, (matrix, title, mean_absolute) in zip(
        axes[3:], differences, strict=True
    ):
        difference_image = axis.imshow(
            matrix,
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=-difference_bound, vcenter=0, vmax=difference_bound
            ),
        )
        axis.set_title(title)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Layer")
        axis.set_xticks((0, 5, 10, 15, 20, 25, 31))
        axis.set_yticks((0, 5, 10, 15, 20, 25, 31))
        for boundary in BLOCK_BOUNDARIES:
            axis.axhline(boundary, color="black", linewidth=0.7, alpha=0.45)
            axis.axvline(boundary, color="black", linewidth=0.7, alpha=0.45)
        axis.text(
            0.02,
            0.02,
            f"mean |Δ| off diagonal\n{mean_absolute:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
    if difference_image is None:
        raise RuntimeError("difference heatmaps were not rendered")
    figure.colorbar(
        difference_image,
        ax=axes[3:],
        label="Change in activation CKA",
        shrink=0.76,
        pad=0.012,
    )
    figure.suptitle(
        "How the lesion and 160-step recovery change inter-layer organization",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.005,
        "Qwen3.5-4B · 129 held-out multihop prompts · final-position residuals · "
        "white/black boundaries: published sensory–workspace–motor blocks · "
        f"lesioned blocks: {list(LESION_LAYERS)}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in render(args.source, args.output, dpi=args.dpi):
        print(path)


if __name__ == "__main__":
    main()

"""Headless Matplotlib plots over the separate CSV metric streams."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_numeric(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in rows[0]
    }


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, smoothed])


def plot_all(
    output_dir: Path,
    *,
    rolling_window: int = 20,
    dpi: int = 160,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    losses = _read_numeric(metrics_dir / "losses.csv")
    if losses:
        figure, axis = plt.subplots(figsize=(9, 5))
        for field, label in (
            ("total_loss", "total loss"),
            ("policy_loss", "policy loss"),
        ):
            axis.plot(
                losses["step"],
                _rolling(losses[field], rolling_window),
                label=label,
                linewidth=1.8,
            )
        axis.set(title="Training losses", xlabel="optimizer step", ylabel="loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "losses.png", dpi=dpi)
        plt.close(figure)

    rewards = _read_numeric(metrics_dir / "rewards.csv")
    if rewards:
        figure, axis = plt.subplots(figsize=(9, 5))
        reward_mean = _rolling(rewards["reward_mean"], rolling_window)
        reward_std = _rolling(rewards["reward_std"], rolling_window)
        axis.plot(
            rewards["step"],
            reward_mean,
            label="mean reward",
            linewidth=1.8,
        )
        axis.fill_between(
            rewards["step"],
            np.clip(reward_mean - reward_std, 0.0, 1.0),
            np.clip(reward_mean + reward_std, 0.0, 1.0),
            alpha=0.18,
            label="± reward std",
        )
        axis.set(
            title="Rewards",
            xlabel="optimizer step",
            ylabel="fraction",
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "rewards.png", dpi=dpi)
        plt.close(figure)

    evaluations_path = metrics_dir / "evaluations.csv"
    if evaluations_path.exists():
        with evaluations_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            series = sorted({(row["split"], row["condition"]) for row in rows})
            for split, condition in series:
                selected = [
                    row
                    for row in rows
                    if row["split"] == split and row["condition"] == condition
                ]
                axis.plot(
                    [float(row["step"]) for row in selected],
                    [float(row["candidate_accuracy"]) for row in selected],
                    marker="o",
                    label=f"{split}/{condition}",
                )
            chance_values = sorted({float(row["chance_accuracy"]) for row in rows})
            for chance in chance_values:
                axis.axhline(
                    chance,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                    label=f"chance={chance:.3f}",
                )
            axis.set(
                title="Evaluation accuracy",
                xlabel="optimizer step",
                ylabel="candidate accuracy",
                ylim=(-0.02, 1.02),
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, ncol=2)
            figure.tight_layout()
            figure.savefig(plots_dir / "evaluations.png", dpi=dpi)
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    plot_all(
        args.output_dir,
        rolling_window=args.rolling_window,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

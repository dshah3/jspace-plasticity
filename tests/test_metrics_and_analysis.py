from __future__ import annotations

from pathlib import Path

import numpy as np

from jspace_plasticity.analyze_lens import _three_blocks
from jspace_plasticity.metrics import MetricsWriter
from jspace_plasticity.plotting import plot_all


def test_three_block_segmentation_finds_clear_blocks() -> None:
    cka = np.full((9, 9), 0.1, dtype=np.float64)
    for start in (0, 3, 6):
        cka[start : start + 3, start : start + 3] = 0.95
    np.fill_diagonal(cka, 1.0)
    assert _three_blocks(cka) == (3, 6)


def test_metrics_and_plots_are_separate(tmp_path: Path) -> None:
    writer = MetricsWriter(tmp_path)
    for step in range(1, 4):
        writer.write_loss(
            {
                "step": step,
                "total_loss": 1.0 / step,
                "policy_loss": 1.0 / step,
                "entropy": 0.5,
                "grad_norm": 1.0,
                "learning_rate": 1e-5,
            }
        )
        writer.write_reward(
            {
                "step": step,
                "reward_mean": step / 3,
                "reward_std": 0.1,
                "reward_min": 0.0,
                "reward_max": 1.0,
                "success_rate": step / 3,
                "advantage_std": 1.0,
                "zero_variance_fraction": 0.0,
            }
        )
    plot_all(tmp_path, rolling_window=2, dpi=40)
    assert (tmp_path / "metrics" / "losses.csv").is_file()
    assert (tmp_path / "metrics" / "rewards.csv").is_file()
    assert (tmp_path / "plots" / "losses.png").is_file()
    assert (tmp_path / "plots" / "rewards.png").is_file()

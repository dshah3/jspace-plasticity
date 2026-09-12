from __future__ import annotations

from pathlib import Path

import torch

from jspace_plasticity.lens import LensMatrices


def test_loads_saved_lens_mean(tmp_path: Path) -> None:
    path = tmp_path / "lens.pt"
    torch.save(
        {"J": {0: torch.eye(2)}, "n_prompts": 7, "d_model": 2},
        path,
    )
    lens = LensMatrices.load(path)
    assert lens.n_prompts == 7
    assert torch.equal(lens.jacobians[0], torch.eye(2))


def test_loads_neuronpedia_running_sum_as_mean(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "jacobian_sum": {0: 5 * torch.eye(2), 1: 10 * torch.eye(2)},
            "n_done": 5,
            "next_idx": 5,
            "source_layers": [0, 1],
        },
        path,
    )
    lens = LensMatrices.load(path)
    assert lens.n_prompts == 5
    assert torch.equal(lens.jacobians[0], torch.eye(2))
    assert torch.equal(lens.jacobians[1], 2 * torch.eye(2))

"""Memory-bounded geometry diagnostics for lens validation."""

from __future__ import annotations

import torch


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    """Feature-space linear CKA without constructing an ``n_tokens²`` Gram."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must be 2-D with equal row counts")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = left.T @ right
    left_cov = left.T @ left
    right_cov = right.T @ right
    denominator = torch.linalg.matrix_norm(left_cov) * torch.linalg.matrix_norm(
        right_cov
    )
    if denominator <= 0:
        raise ValueError("CKA is undefined for a constant matrix")
    return float(torch.linalg.matrix_norm(cross).square() / denominator)


def excess_kurtosis(values: torch.Tensor) -> float:
    flat = values.float().reshape(-1)
    centered = flat - flat.mean()
    variance = centered.square().mean()
    if variance <= 0:
        raise ValueError("kurtosis is undefined for constant values")
    return float(centered.pow(4).mean() / variance.square() - 3)

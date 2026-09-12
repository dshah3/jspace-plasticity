"""Explicit normalization conventions for readout-aligned directions."""

from __future__ import annotations

import torch
from torch import nn


def effective_norm_gain(norm: nn.Module) -> torch.Tensor | None:
    """Return the multiplicative gain, including Qwen3.5's unit offset.

    Do not infer the convention from parameter values: trained offsets need
    not be near zero. Gated RMSNorm is deliberately not an offset norm.
    """
    weight = getattr(norm, "weight", None)
    if weight is None:
        if isinstance(norm, nn.Identity):
            return None
        raise TypeError(f"unsupported weightless final norm: {type(norm)}")
    name = type(norm).__name__
    if name in {
        "Qwen3_5RMSNorm",
        "Qwen3_5MoeRMSNorm",
        "GemmaRMSNorm",
        "Gemma2RMSNorm",
        "Gemma3RMSNorm",
    }:
        return 1.0 + weight.detach().float()
    if isinstance(norm, (nn.RMSNorm, nn.LayerNorm)) or name in {
        "LlamaRMSNorm",
        "Qwen2RMSNorm",
        "Qwen3RMSNorm",
        "Qwen3MoeRMSNorm",
        "MistralRMSNorm",
        "T5LayerNorm",
    }:
        return weight.detach()
    raise TypeError(f"unverified final norm gain convention: {type(norm)}")


def readout_rows(
    rows: torch.Tensor, norm: nn.Module, *, convention: str = "effective_gain"
) -> torch.Tensor:
    if convention == "legacy_weight":
        gain = getattr(norm, "weight", None)
        return rows if gain is None else rows * gain.detach()
    if convention != "effective_gain":
        raise ValueError(f"unknown direction convention: {convention}")
    gain = effective_norm_gain(norm)
    rows = rows.float() if gain is None else rows.float() * gain.float()
    # LayerNorm centers its input; RMSNorm does not. Bias is independent of h.
    if isinstance(norm, nn.LayerNorm):
        rows = rows - rows.mean(dim=-1, keepdim=True)
    return rows

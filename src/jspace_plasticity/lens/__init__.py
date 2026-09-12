"""Jacobain-lens artifact loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LensMatrices:
    """Training-safe view of Jacobian matrices keyed by decoder layer."""

    jacobians: dict[int, torch.Tensor]
    d_model: int
    n_prompts: int
    source: str

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.jacobians))

    @classmethod
    def load(cls, path: str | Path) -> LensMatrices:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" in checkpoint:
            raw_jacobians = checkpoint["J"]
            n_prompts = int(checkpoint.get("n_prompts", 0))
        elif "jacobian_sum" in checkpoint and "n_done" in checkpoint:
            # Neuronpedia publishes Anthropic's resumable fit checkpoint rather
            # than JacobianLens.save() output.  Its tensors are sums, not means.
            # Divide in place to avoid duplicating a multi-gigabyte fp32 artifact.
            n_prompts = int(checkpoint["n_done"])
            if n_prompts < 1:
                raise ValueError(f"lens fit checkpoint {path} has n_done={n_prompts}")
            raw_jacobians = checkpoint["jacobian_sum"]
        else:
            raise ValueError(
                f"{path} is neither a saved JacobianLens nor a resumable fit "
                "checkpoint; expected 'J' or ('jacobian_sum', 'n_done')"
            )
        jacobians = {
            int(layer): value.float() for layer, value in raw_jacobians.items()
        }
        if "J" not in checkpoint:
            for matrix in jacobians.values():
                matrix.div_(n_prompts)
        if not jacobians:
            raise ValueError(f"lens checkpoint {path} contains no layers")
        d_model = int(
            checkpoint.get("d_model", next(iter(jacobians.values())).shape[0])
        )
        cls._validate_shapes(jacobians, d_model)
        return cls(
            jacobians=jacobians,
            d_model=d_model,
            n_prompts=n_prompts,
            source=str(path),
        )

    @classmethod
    def identity(cls, layers: list[int], d_model: int) -> LensMatrices:
        if not layers:
            raise ValueError("identity lens requires at least one layer")
        identity = torch.eye(d_model, dtype=torch.float32)
        return cls(
            jacobians={layer: identity for layer in layers},
            d_model=d_model,
            n_prompts=0,
            source="identity-smoke-fixture",
        )

    @staticmethod
    def _validate_shapes(jacobians: dict[int, torch.Tensor], d_model: int) -> None:
        expected = (d_model, d_model)
        invalid = {
            layer: tuple(matrix.shape)
            for layer, matrix in jacobians.items()
            if tuple(matrix.shape) != expected
        }
        if invalid:
            raise ValueError(f"lens matrices do not match d_model={d_model}: {invalid}")

    def select(
        self,
        layers: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        missing = sorted(set(layers) - set(self.jacobians))
        if missing:
            raise ValueError(
                f"requested intervention layers {missing} are absent from lens; "
                f"available layers are {list(self.layers)}"
            )
        return {
            layer: self.jacobians[layer].to(device=device, dtype=dtype)
            for layer in layers
        }


from jspace_plasticity.lens.provenance import PublishedLensSpec  # noqa: E402

__all__ = ["LensMatrices", "PublishedLensSpec"]

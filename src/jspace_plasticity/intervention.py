"""Context-adaptive, differentiable J-space activation ablation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from jspace_plasticity.config import InterventionConfig
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.modeling import ResolvedModel
from jspace_plasticity.readout import readout_rows


def _hidden_from_output(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported transformer block output: {type(output).__name__}")


def _replace_hidden(output: object, hidden: torch.Tensor) -> object:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported transformer block output: {type(output).__name__}")


@dataclass(frozen=True)
class AblationPlan:
    """Per-forward state for either clean-trajectory or online selection.

    The clean forward always supplies the output-token exclusion set.  Legacy
    plans also contain preselected directions, while online plans populate
    ``selected_token_ids`` layer by layer inside the intervened forward.
    """

    directions: dict[int, torch.Tensor]  # layer -> [batch, seq, k, d_model]
    selected_token_ids: dict[int, torch.Tensor]  # layer -> [batch, seq, k]
    blocked_token_ids: torch.Tensor  # [batch, seq, exclude_output_top_k]
    attention_mask: torch.Tensor  # [batch, seq]
    prompt_length: int
    clean_next_logits: torch.Tensor  # [batch, vocab]
    clean_top_token_ids: torch.Tensor  # [batch, seq]
    random_directions: dict[int, torch.Tensor] | None = None
    # Optional, audit-only rankings.  These may contain more than ``config.k``
    # token ids, but only ``selected_token_ids`` are projected out.  Keeping the
    # two separate lets diagnostics inspect ranks immediately below the lesion
    # cutoff without changing the intervention itself.
    ranked_token_ids: dict[int, torch.Tensor] | None = None
    audit_top_k: int | None = None


@dataclass(frozen=True)
class CoordinateSwapPlan:
    """Two J-lens coordinates to exchange at every sequence position."""

    directions: dict[int, torch.Tensor]  # layer -> [2, d_model]
    token_ids: tuple[int, int]
    prompt_length: int


class JSpaceAblator:
    """Select and ablate J-space directions during model forwards."""

    def __init__(
        self,
        resolved: ResolvedModel,
        lens: LensMatrices,
        config: InterventionConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not config.enabled:
            raise ValueError("JSpaceAblator requires an enabled intervention")
        self.resolved = resolved
        self.config = config
        self.device = device
        self.dtype = dtype
        if lens.d_model != resolved.lm_head.weight.shape[1]:
            raise ValueError(
                f"lens d_model={lens.d_model} does not match LM head width "
                f"{resolved.lm_head.weight.shape[1]}"
            )
        if max(config.layers) >= len(resolved.layers):
            raise ValueError(
                f"intervention layer {max(config.layers)} is outside the model's "
                f"{len(resolved.layers)} blocks"
            )
        self.jacobians = lens.select(config.layers, device=device, dtype=dtype)

    @property
    def condition_name(self) -> str:
        return "lesioned" if self.config.control == "none" else self.config.control

    def _capture_clean(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        capture_layers: bool,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        captured: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(layer: int) -> Callable[..., None]:
            def hook(module: nn.Module, inputs: object, output: object) -> None:
                del module, inputs
                captured[layer] = _hidden_from_output(output).detach()

            return hook

        try:
            if capture_layers:
                for layer in self.config.layers:
                    handles.append(
                        self.resolved.layers[layer].register_forward_hook(
                            make_hook(layer)
                        )
                    )
            # A no-grad forward must not populate the outer training autocast
            # cache with detached casts of trainable weights.
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=input_ids.device.type,
                    enabled=torch.is_autocast_enabled(input_ids.device.type),
                    cache_enabled=False,
                ),
            ):
                clean_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits.detach()
        finally:
            for handle in handles:
                handle.remove()
        expected_layers = set(self.config.layers) if capture_layers else set()
        missing = sorted(expected_layers - set(captured))
        if missing:
            raise RuntimeError(f"clean forward did not capture layers {missing}")
        return captured, clean_logits

    @staticmethod
    def _filter_blocked(
        candidates: torch.Tensor,
        blocked: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        # candidates: [..., k + n_blocked], blocked: [..., n_blocked]
        if blocked.shape[-1] == 0:
            return candidates[..., :k]
        is_blocked = (candidates.unsqueeze(-1) == blocked.unsqueeze(-2)).any(dim=-1)
        positions = torch.arange(
            candidates.shape[-1], device=candidates.device
        ).expand_as(candidates)
        positions = positions.masked_fill(is_blocked, candidates.shape[-1])
        keep = positions.topk(k, dim=-1, largest=False, sorted=True).indices
        selected = candidates.gather(-1, keep)
        if (selected < 0).any():
            raise RuntimeError("failed to select enough non-output J-space tokens")
        return selected

    @staticmethod
    def _extend_selected_ranking(
        candidates: torch.Tensor,
        blocked: torch.Tensor,
        selected: torch.Tensor,
        total_k: int,
    ) -> torch.Tensor:
        """Append the best remaining candidates after the exact selected prefix.

        Separate ``torch.topk(k)`` and ``torch.topk(audit_k)`` calls can order
        tied BF16 scores differently.  The lesion must retain the exact result
        of its original top-k call, so audit ranks 1..k are anchored to
        ``selected`` and only ranks k+1..audit_k come from the larger candidate
        pool.
        """

        selected_k = selected.shape[-1]
        if total_k < selected_k:
            raise ValueError("audit ranking cannot be shorter than selected prefix")
        if total_k == selected_k:
            return selected
        is_blocked = (candidates.unsqueeze(-1) == blocked.unsqueeze(-2)).any(dim=-1)
        is_selected = (candidates.unsqueeze(-1) == selected.unsqueeze(-2)).any(dim=-1)
        excluded = is_blocked | is_selected
        positions = torch.arange(
            candidates.shape[-1], device=candidates.device
        ).expand_as(candidates)
        positions = positions.masked_fill(excluded, candidates.shape[-1])
        extra_k = total_k - selected_k
        keep = positions.topk(extra_k, dim=-1, largest=False, sorted=True).indices
        if (positions.gather(-1, keep) == candidates.shape[-1]).any():
            raise RuntimeError("failed to find enough audit-only J-space tokens")
        extras = candidates.gather(-1, keep)
        return torch.cat((selected, extras), dim=-1)

    def _directions_for(
        self,
        layer: int,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # For RMSNorm-based decoders the token direction is the corresponding row
        # of W_U diag(g) J_l. The common positive RMS scale does not affect ranking.
        rows = self.resolved.lm_head.weight.detach()[token_ids]
        rows = readout_rows(
            rows,
            self.resolved.final_norm,
            convention=self.config.direction_convention,
        )
        directions = torch.matmul(rows.to(self.dtype), self.jacobians[layer])
        directions = torch.nn.functional.normalize(
            directions.float(), dim=-1, eps=1e-8
        ).to(self.dtype)
        return directions * attention_mask[..., None, None].to(self.dtype)

    def _select_from_hidden(
        self,
        layer: int,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        blocked: torch.Tensor,
        *,
        audit_top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Select active J-directions from one residual-stream activation.

        ``hidden`` may already include interventions at earlier layers. Selection
        is detached because top-k is discrete; the subsequent projection remains
        differentiable with respect to the live residual stream.
        """

        if audit_top_k is not None and audit_top_k < self.config.k:
            raise ValueError("audit_top_k cannot be smaller than intervention k")
        overfetch = self.config.k + blocked.shape[-1]
        with torch.no_grad():
            transported = torch.matmul(
                hidden.detach().to(self.dtype), self.jacobians[layer].T
            )
            readout = self.resolved.lm_head(self.resolved.final_norm(transported))
            candidates = readout.topk(overfetch, dim=-1).indices
            token_ids = self._filter_blocked(candidates, blocked, self.config.k)
            ranked_ids = None
            if audit_top_k is not None:
                # The larger call supplies only the continuation after the exact
                # projected prefix.  It cannot alter the lesion's top-k choice.
                audit_candidates = readout.topk(
                    audit_top_k + blocked.shape[-1] + self.config.k, dim=-1
                ).indices
                ranked_ids = self._extend_selected_ranking(
                    audit_candidates,
                    blocked,
                    token_ids,
                    audit_top_k,
                ).detach()
            directions = self._directions_for(layer, token_ids, attention_mask).detach()
        return token_ids.detach(), directions, ranked_ids

    def _random_seed_for_example(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        layer: int,
    ) -> int:
        seed = self.config.random_seed + layer
        if self.config.control_resample == "fixed":
            return seed
        active_ids = input_ids.detach()[attention_mask.detach().bool()]
        payload = active_ids.to(device="cpu", dtype=torch.int64).numpy().tobytes()
        digest = hashlib.blake2b(
            payload,
            digest_size=8,
            person=b"jspace-random",
        ).digest()
        return (seed + int.from_bytes(digest, "little")) % (2**63 - 1)

    def _random_directions_for(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        layer: int,
    ) -> torch.Tensor:
        """Create deterministic fixed or prompt-resampled random controls."""

        width = self.resolved.lm_head.weight.shape[1]
        rows = []
        for ids, mask in zip(input_ids, attention_mask, strict=True):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._random_seed_for_example(ids, mask, layer=layer))
            if self.config.control == "matched_random":
                row = torch.randn(
                    (attention_mask.shape[1], width),
                    generator=generator,
                    dtype=torch.float32,
                )
                row = torch.nn.functional.normalize(row, dim=-1, eps=1e-8)
                row = row * mask.detach().to(device="cpu", dtype=torch.float32)[:, None]
            elif self.config.control == "matched_random_subspace":
                gaussian = torch.randn(
                    (attention_mask.shape[1], width, self.config.k),
                    generator=generator,
                    dtype=torch.float32,
                )
                basis = torch.linalg.qr(gaussian, mode="reduced").Q
                row = basis.transpose(-2, -1)
                row = (
                    row
                    * mask.detach().to(device="cpu", dtype=torch.float32)[:, None, None]
                )
            else:
                raise ValueError(
                    f"random directions requested for control={self.config.control}"
                )
            rows.append(row)
        return torch.stack(rows).to(device=self.device, dtype=torch.float32)

    def build_plan(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        audit_top_k: int | None = None,
    ) -> AblationPlan:
        """Select directions under `no_grad`; gradients never cross top-k."""

        captured, clean_logits = self._capture_clean(
            model,
            input_ids,
            attention_mask,
            capture_layers=self.config.selection_source == "clean_trajectory",
        )
        blocked_k = self.config.exclude_output_top_k
        blocked = (
            clean_logits.topk(blocked_k, dim=-1).indices
            if blocked_k
            else torch.empty(
                (*clean_logits.shape[:-1], 0),
                dtype=torch.long,
                device=clean_logits.device,
            )
        )
        clean_next_logits = clean_logits[:, -1, :].detach()
        clean_top_token_ids = clean_logits.argmax(dim=-1).detach()
        del clean_logits

        directions: dict[int, torch.Tensor] = {}
        selected_ids: dict[int, torch.Tensor] = {}
        ranked_ids: dict[int, torch.Tensor] | None = (
            {} if audit_top_k is not None else None
        )
        random_directions: dict[int, torch.Tensor] | None = (
            {}
            if self.config.control in ("matched_random", "matched_random_subspace")
            else None
        )
        if audit_top_k is not None and audit_top_k < self.config.k:
            raise ValueError("audit_top_k cannot be smaller than intervention k")
        overfetch = (
            audit_top_k + self.config.k + blocked_k
            if audit_top_k is not None
            else self.config.k + blocked_k
        )
        vocabulary_size = self.resolved.lm_head.weight.shape[0]
        if overfetch > vocabulary_size:
            raise ValueError(
                f"k + exclude_output_top_k={overfetch} exceeds vocabulary size "
                f"{vocabulary_size}"
            )

        with torch.no_grad():
            for layer in self.config.layers:
                if self.config.selection_source == "clean_trajectory":
                    token_ids, layer_directions, layer_ranked_ids = (
                        self._select_from_hidden(
                            layer,
                            captured[layer],
                            attention_mask,
                            blocked,
                            audit_top_k=audit_top_k,
                        )
                    )
                    selected_ids[layer] = token_ids
                    directions[layer] = layer_directions
                    if ranked_ids is not None:
                        if layer_ranked_ids is None:
                            raise RuntimeError("audit ranking was not populated")
                        ranked_ids[layer] = layer_ranked_ids
                if random_directions is not None:
                    random_directions[layer] = self._random_directions_for(
                        input_ids,
                        attention_mask,
                        layer=layer,
                    )

        return AblationPlan(
            directions=directions,
            selected_token_ids=selected_ids,
            blocked_token_ids=blocked.detach(),
            attention_mask=attention_mask.detach(),
            prompt_length=input_ids.shape[1],
            clean_next_logits=clean_next_logits,
            clean_top_token_ids=clean_top_token_ids,
            random_directions=random_directions,
            ranked_token_ids=ranked_ids,
            audit_top_k=audit_top_k,
        )

    def _project(self, hidden: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        if hidden.shape[:2] != directions.shape[:2]:
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} does not match plan shape "
                f"{tuple(directions.shape)}"
            )
        strength = self.config.strength
        if self.config.projection == "sequential":
            projected = hidden.float()
            for index in range(directions.shape[-2]):
                direction = directions[..., index, :].float()
                coefficient = (projected * direction).sum(dim=-1, keepdim=True)
                projected = projected - strength * coefficient * direction
            return projected.to(hidden.dtype)

        hidden_float = hidden.float()
        direction_float = directions.float()
        if self.config.projection == "orthogonal_span":
            # SVD handles dependent/zero rows; unpivoted QR would delete
            # arbitrary extra dimensions for a rank-deficient selected set.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                _, singular, vh = torch.linalg.svd(direction_float, full_matrices=False)
                tolerance = (
                    max(direction_float.shape[-2:])
                    * torch.finfo(torch.float32).eps
                    * singular[..., :1]
                )
                basis = vh * (singular > tolerance).unsqueeze(-1)
                coefficients = torch.einsum("btkd,btd->btk", basis, hidden_float)
                delta = torch.einsum("btk,btkd->btd", coefficients, basis)
                return (hidden_float - strength * delta).to(hidden.dtype)
        if self.config.projection != "joint":
            raise ValueError(f"unknown projection: {self.config.projection}")
        gram = torch.einsum("btkd,btjd->btkj", direction_float, direction_float)
        identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        gram = gram + self.config.ridge * identity
        rhs = torch.einsum("btkd,btd->btk", direction_float, hidden_float)
        coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
        delta = torch.einsum("btk,btkd->btd", coefficients, direction_float)
        return (hidden_float - strength * delta).to(hidden.dtype)

    @staticmethod
    def _shrink_non_j(hidden: torch.Tensor, j_projected: torch.Tensor) -> torch.Tensor:
        """Shrink the projected residual, capped by its norm.

        It is a true orthogonal complement only for an orthogonal projection;
        sequential and ridge projections do not guarantee coordinate retention.
        """

        hidden_float = hidden.float()
        non_j = j_projected.float()
        removal_norm = (hidden_float - non_j).norm(dim=-1, keepdim=True)
        non_j_norm = non_j.norm(dim=-1, keepdim=True)
        shrink = torch.minimum(removal_norm, non_j_norm)
        unit_non_j = non_j / non_j_norm.clamp_min(1e-12)
        return (hidden_float - shrink * unit_non_j).to(hidden.dtype)

    @contextmanager
    def apply(self, plan: AblationPlan) -> Iterator[None]:
        """Keep hooks installed through backward for checkpoint recomputation."""

        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(layer: int) -> Callable[..., object]:
            def hook(module: nn.Module, inputs: object, output: object) -> object:
                del module, inputs
                hidden = _hidden_from_output(output)
                if hidden.shape[1] != plan.prompt_length:
                    raise ValueError(
                        "intervention plan cannot be reused with a different sequence "
                        f"length: {hidden.shape[1]} != {plan.prompt_length}"
                    )
                if self.config.selection_source == "online_current":
                    token_ids, directions, ranked_ids = self._select_from_hidden(
                        layer,
                        hidden,
                        plan.attention_mask,
                        plan.blocked_token_ids,
                        audit_top_k=plan.audit_top_k,
                    )
                    plan.selected_token_ids[layer] = token_ids
                    if plan.ranked_token_ids is not None:
                        if ranked_ids is None:
                            raise RuntimeError("audit ranking was not populated")
                        plan.ranked_token_ids[layer] = ranked_ids
                else:
                    directions = plan.directions[layer]
                projected = self._project(hidden, directions)
                if self.config.control == "shrink_non_j":
                    projected = self._shrink_non_j(hidden, projected)
                elif plan.random_directions is not None:
                    random_direction = plan.random_directions[layer]
                    if self.config.control == "matched_random":
                        perturbation_norm = (
                            (hidden - projected).float().norm(dim=-1, keepdim=True)
                        )
                        projected = (
                            hidden.float()
                            - random_direction.float() * perturbation_norm
                        ).to(hidden.dtype)
                    elif self.config.control == "matched_random_subspace":
                        projected = self._project(hidden, random_direction)
                    else:
                        raise RuntimeError(
                            "ablation plan contains random directions for "
                            f"control={self.config.control}"
                        )
                return _replace_hidden(output, projected)

            return hook

        try:
            for layer in self.config.layers:
                handles.append(
                    self.resolved.layers[layer].register_forward_hook(make_hook(layer))
                )
            yield
            if self.config.selection_source == "online_current":
                missing = sorted(set(self.config.layers) - set(plan.selected_token_ids))
                if missing:
                    raise RuntimeError(
                        "intervened forward did not execute online selection at "
                        f"layers {missing}"
                    )
                if plan.ranked_token_ids is not None:
                    missing_rankings = sorted(
                        set(self.config.layers) - set(plan.ranked_token_ids)
                    )
                    if missing_rankings:
                        raise RuntimeError(
                            "intervened forward did not populate audit rankings at "
                            f"layers {missing_rankings}"
                        )
        finally:
            for handle in handles:
                handle.remove()


class JLensCoordinateSwapper:
    """Apply the paper's two-coordinate pseudoinverse swap intervention.

    For unit-normalized source and target J-lens directions stacked as ``V``, the
    local coordinates are ``c = V^dagger h``. Exchanging the two entries of
    ``c`` and writing the difference back leaves the component orthogonal to the
    two-direction span unchanged.
    """

    def __init__(
        self,
        resolved: ResolvedModel,
        lens: LensMatrices,
        layers: list[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        direction_convention: str = "effective_gain",
    ) -> None:
        if not layers:
            raise ValueError("coordinate swapping requires at least one layer")
        if lens.d_model != resolved.lm_head.weight.shape[1]:
            raise ValueError(
                f"lens d_model={lens.d_model} does not match LM head width "
                f"{resolved.lm_head.weight.shape[1]}"
            )
        if max(layers) >= len(resolved.layers) or min(layers) < 0:
            raise ValueError(
                f"coordinate swap layers {layers} fall outside the model's "
                f"{len(resolved.layers)} blocks"
            )
        self.resolved = resolved
        self.direction_convention = direction_convention
        self.layers = tuple(layers)
        self.device = device
        self.dtype = dtype
        self.jacobians = lens.select(layers, device=device, dtype=dtype)

    def _directions(self, layer: int, token_ids: torch.Tensor) -> torch.Tensor:
        rows = self.resolved.lm_head.weight.detach()[token_ids]
        rows = readout_rows(
            rows, self.resolved.final_norm, convention=self.direction_convention
        )
        directions = torch.matmul(rows.to(self.dtype), self.jacobians[layer])
        return torch.nn.functional.normalize(directions.float(), dim=-1, eps=1e-8).to(
            self.dtype
        )

    @torch.no_grad()
    def build_plan(
        self,
        source_token_id: int,
        target_token_id: int,
        *,
        prompt_length: int,
    ) -> CoordinateSwapPlan:
        if source_token_id == target_token_id:
            raise ValueError("coordinate swap source and target tokens must differ")
        if prompt_length < 1:
            raise ValueError("coordinate swap prompt length must be positive")
        token_ids = torch.tensor(
            [source_token_id, target_token_id],
            dtype=torch.long,
            device=self.device,
        )
        directions = {
            layer: self._directions(layer, token_ids).detach() for layer in self.layers
        }
        return CoordinateSwapPlan(
            directions=directions,
            token_ids=(source_token_id, target_token_id),
            prompt_length=prompt_length,
        )

    @contextmanager
    def apply(self, plan: CoordinateSwapPlan) -> Iterator[None]:
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(layer: int) -> Callable[..., object]:
            directions = plan.directions[layer].float()
            # pinv([2, d]) has shape [d, 2], so h @ pinv are the two local
            # least-squares coordinates used by the paper's intervention.
            pseudoinverse = torch.linalg.pinv(directions)

            def hook(module: nn.Module, inputs: object, output: object) -> object:
                del module, inputs
                hidden = _hidden_from_output(output)
                if hidden.shape[1] != plan.prompt_length:
                    raise ValueError(
                        "coordinate swap plan cannot be reused with a different "
                        f"sequence length: {hidden.shape[1]} != {plan.prompt_length}"
                    )
                hidden_float = hidden.float()
                coefficients = torch.matmul(hidden_float, pseudoinverse)
                swapped = coefficients.flip(dims=(-1,))
                delta = torch.matmul(swapped - coefficients, directions)
                return _replace_hidden(output, (hidden_float + delta).to(hidden.dtype))

            return hook

        try:
            for layer in self.layers:
                handles.append(
                    self.resolved.layers[layer].register_forward_hook(make_hook(layer))
                )
            yield
        finally:
            for handle in handles:
                handle.remove()

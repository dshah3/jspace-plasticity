from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jspace_plasticity.config import InterventionConfig
from jspace_plasticity.intervention import AblationPlan, JSpaceAblator


@pytest.mark.parametrize("projection", ["sequential", "joint"])
def test_projection_removes_selected_orthogonal_components(projection: str) -> None:
    ablator = object.__new__(JSpaceAblator)
    ablator.config = InterventionConfig(  # type: ignore[misc]
        enabled=True,
        lens_path="unused",
        layers=[0],
        projection=projection,  # type: ignore[arg-type]
        strength=1.0,
    )
    hidden = torch.tensor([[[3.0, 4.0, 5.0]]])
    directions = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
    projected = ablator._project(hidden, directions)
    assert torch.allclose(projected, torch.tensor([[[0.0, 0.0, 5.0]]]), atol=1e-3)


def test_output_tokens_are_excluded_from_top_k() -> None:
    candidates = torch.tensor([[[9, 8, 7, 6, 5]]])
    blocked = torch.tensor([[[9, 7]]])
    selected = JSpaceAblator._filter_blocked(candidates, blocked, k=3)
    assert selected.tolist() == [[[8, 6, 5]]]


def test_audit_ranking_keeps_exact_selected_prefix_under_tie_reordering() -> None:
    # Simulate separate top-k calls returning a different order for tied scores.
    selected = torch.tensor([[[2, 1]]])
    larger_call = torch.tensor([[[1, 2, 9, 3, 4, 5]]])
    blocked = torch.tensor([[[9]]])
    ranked = JSpaceAblator._extend_selected_ranking(
        larger_call, blocked, selected, total_k=5
    )
    assert ranked.tolist() == [[[2, 1, 3, 4, 5]]]


def _random_ablator(
    *, control: str = "matched_random", resample: str = "per_example"
) -> JSpaceAblator:
    ablator = object.__new__(JSpaceAblator)
    head = nn.Linear(4, 8, bias=False)
    ablator.resolved = SimpleNamespace(lm_head=head)
    ablator.config = InterventionConfig(
        enabled=True,
        lens_path="unused",
        layers=[0],
        k=2,
        control=control,  # type: ignore[arg-type]
        control_resample=resample,  # type: ignore[arg-type]
        random_seed=17,
    )
    ablator.device = torch.device("cpu")
    ablator.dtype = torch.bfloat16
    return ablator


def test_per_example_random_control_is_reproducible_but_prompt_specific() -> None:
    ablator = _random_ablator()
    mask = torch.ones((1, 3), dtype=torch.long)
    first_ids = torch.tensor([[1, 2, 3]])
    second_ids = torch.tensor([[1, 2, 4]])
    first = ablator._random_directions_for(first_ids, mask, layer=0)
    repeat = ablator._random_directions_for(first_ids, mask, layer=0)
    second = ablator._random_directions_for(second_ids, mask, layer=0)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)


def test_fixed_random_control_reuses_draw_across_prompts() -> None:
    ablator = _random_ablator(resample="fixed")
    mask = torch.ones((1, 3), dtype=torch.long)
    first = ablator._random_directions_for(torch.tensor([[1, 2, 3]]), mask, layer=0)
    second = ablator._random_directions_for(torch.tensor([[4, 5, 6]]), mask, layer=0)
    assert torch.equal(first, second)


def test_random_subspace_control_is_orthonormal() -> None:
    ablator = _random_ablator(control="matched_random_subspace")
    directions = ablator._random_directions_for(
        torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.long), layer=0
    )
    gram = torch.einsum("btkd,btjd->btkj", directions, directions)
    expected = torch.eye(2).expand_as(gram)
    assert torch.allclose(gram, expected, atol=1e-6)


def test_sequential_projection_accumulates_in_fp32_before_cast() -> None:
    ablator = object.__new__(JSpaceAblator)
    ablator.config = InterventionConfig(
        enabled=True,
        lens_path="unused",
        layers=[0],
        projection="sequential",
    )
    hidden = torch.tensor([[[1.234, -0.876, 2.468]]], dtype=torch.bfloat16)
    directions = torch.tensor(
        [[[[0.91, 0.40, 0.10], [0.20, 0.95, 0.24], [0.61, 0.22, 0.76]]]],
        dtype=torch.bfloat16,
    )
    directions = torch.nn.functional.normalize(directions.float(), dim=-1).to(
        torch.bfloat16
    )
    expected = hidden.float()
    for index in range(directions.shape[-2]):
        direction = directions[..., index, :].float()
        expected = expected - (expected * direction).sum(-1, keepdim=True) * direction
    projected = ablator._project(hidden, directions)
    assert torch.equal(projected, expected.to(torch.bfloat16))


def test_non_j_shrink_matches_lesion_norm_and_preserves_j_component() -> None:
    hidden = torch.tensor([[[3.0, 4.0, 0.0]]], dtype=torch.bfloat16)
    # Removing the x-axis J component leaves the non-J y-axis component.
    j_projected = torch.tensor([[[0.0, 4.0, 0.0]]], dtype=torch.bfloat16)
    controlled = JSpaceAblator._shrink_non_j(hidden, j_projected)
    assert torch.allclose(controlled.float(), torch.tensor([[[3.0, 1.0, 0.0]]]))
    assert torch.allclose(
        (hidden.float() - controlled.float()).norm(dim=-1),
        (hidden.float() - j_projected.float()).norm(dim=-1),
    )
    # Unlike the lesion, the selected x-axis J coordinate is untouched.
    assert controlled[0, 0, 0] == hidden[0, 0, 0]


class _TwoIdentityLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_online_selection_uses_residual_after_prior_layer_ablation() -> None:
    model = _TwoIdentityLayers()
    lm_head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(2))
    resolved = SimpleNamespace(
        layers=model.layers,
        lm_head=lm_head,
        final_norm=nn.Identity(),
    )
    ablator = object.__new__(JSpaceAblator)
    ablator.resolved = resolved
    ablator.config = InterventionConfig(
        enabled=True,
        lens_path="unused",
        layers=[0, 1],
        k=1,
        exclude_output_top_k=0,
        selection_source="online_current",
        projection="sequential",
    )
    ablator.device = torch.device("cpu")
    ablator.dtype = torch.float32
    ablator.jacobians = {0: torch.eye(2), 1: torch.eye(2)}
    plan = AblationPlan(
        directions={},
        selected_token_ids={},
        blocked_token_ids=torch.empty((1, 1, 0), dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        prompt_length=1,
        clean_next_logits=torch.zeros((1, 2)),
        clean_top_token_ids=torch.zeros((1, 1), dtype=torch.long),
    )

    with ablator.apply(plan):
        output = model(torch.tensor([[[2.0, 1.0]]]))

    assert plan.selected_token_ids[0].tolist() == [[[0]]]
    assert plan.selected_token_ids[1].tolist() == [[[1]]]
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


def test_audit_ranking_does_not_change_projected_top_k() -> None:
    model = _TwoIdentityLayers()
    lm_head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(4))
    resolved = SimpleNamespace(
        layers=model.layers,
        lm_head=lm_head,
        final_norm=nn.Identity(),
    )
    ablator = object.__new__(JSpaceAblator)
    ablator.resolved = resolved
    ablator.config = InterventionConfig(
        enabled=True,
        lens_path="unused",
        layers=[0],
        k=1,
        exclude_output_top_k=0,
        selection_source="online_current",
        projection="sequential",
    )
    ablator.device = torch.device("cpu")
    ablator.dtype = torch.float32
    ablator.jacobians = {0: torch.eye(4)}

    hidden = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    without_audit = AblationPlan(
        directions={},
        selected_token_ids={},
        blocked_token_ids=torch.empty((1, 1, 0), dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        prompt_length=1,
        clean_next_logits=torch.zeros((1, 4)),
        clean_top_token_ids=torch.zeros((1, 1), dtype=torch.long),
    )
    with ablator.apply(without_audit):
        plain_output = model.layers[0](hidden)

    with_audit = AblationPlan(
        directions={},
        selected_token_ids={},
        blocked_token_ids=torch.empty((1, 1, 0), dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        prompt_length=1,
        clean_next_logits=torch.zeros((1, 4)),
        clean_top_token_ids=torch.zeros((1, 1), dtype=torch.long),
        ranked_token_ids={},
        audit_top_k=3,
    )
    with ablator.apply(with_audit):
        audited_output = model.layers[0](hidden)

    assert torch.equal(plain_output, audited_output)
    assert without_audit.selected_token_ids[0].tolist() == [[[0]]]
    assert with_audit.selected_token_ids[0].tolist() == [[[0]]]
    assert with_audit.ranked_token_ids is not None
    assert with_audit.ranked_token_ids[0].tolist() == [[[0, 1, 2]]]

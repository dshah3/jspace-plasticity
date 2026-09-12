from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from jspace_plasticity.evals.recovery_rank_cutoff_sweep import (
    EXPECTED_K_VALUES,
    _rank_audit,
    classify_rank_11_12,
    condition_for,
    condition_indices_for_worker,
    load_design,
)
from jspace_plasticity.intervention import AblationPlan
from jspace_plasticity.synthetic_recovery_sft import LESION_LAYERS

ROOT = Path(__file__).parents[1]
DESIGN = ROOT / "data/evals/q35-recovery-rank-cutoff-sweep-20260826.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_design_has_complete_model_by_k_matrix() -> None:
    design = load_design(DESIGN, _sha256(DESIGN))
    conditions = design["evaluation_conditions"]
    assert len(conditions) == 18
    assert {row["k"] for row in conditions} == set(EXPECTED_K_VALUES)
    assert {row["model"] for row in conditions} == {
        "base",
        "primary",
        "high_dose",
    }
    assert [condition_for(design, index)["index"] for index in range(18)] == list(
        range(18)
    )
    assert design["evaluation"]["training"] is False


def test_design_hash_and_authorization_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_design(DESIGN, "0" * 64)
    mutated = json.loads(DESIGN.read_text(encoding="utf-8"))
    mutated["decision"] = "training_authorized"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        load_design(path, _sha256(path))


def test_worker_shards_cover_each_condition_once() -> None:
    shards = [
        condition_indices_for_worker(
            worker_index=rank, world_size=8, condition_count=18
        )
        for rank in range(8)
    ]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(18))
    assert len(flattened) == len(set(flattened))
    assert max(len(shard) for shard in shards) == 3


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [11] if text.strip() == "Afghanistan" else [99]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"

    def decode(self, token_ids: list[int]) -> str:
        return f"decoded-{token_ids[0]}"


def test_rank_audit_observes_boundary_without_changing_selected_prefix() -> None:
    ranked = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]])
    plan = AblationPlan(
        directions={},
        selected_token_ids={layer: ranked[..., :10] for layer in LESION_LAYERS},
        blocked_token_ids=torch.tensor([[[42, 90]]]),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        prompt_length=1,
        clean_next_logits=torch.zeros((1, 100)),
        clean_top_token_ids=torch.zeros((1, 1), dtype=torch.long),
        ranked_token_ids={layer: ranked for layer in LESION_LAYERS},
        audit_top_k=12,
    )
    audit = _rank_audit(
        plan,
        _Tokenizer(),
        intermediate="Afghanistan",
        expected_token_id=42,
        intervention_k=10,
    )
    assert audit["answer_output_blocked"] is True
    assert audit["intermediate_output_blocked"] is False
    assert audit["intermediate_best_rank_any_layer"] == 11
    assert audit["intermediate_seen_rank11_12_any_layer"] is True
    assert audit["intermediate_rank11_12_without_top10"] is True
    assert audit["intermediate_selected_any_layer"] is False


def _condition_metrics(jspace: float, random: float = 0.9) -> dict:
    return {
        "synthetic_geo": {
            "rows": 129,
            "clean_accuracy": 0.99,
            "jspace_accuracy": jspace,
            "random_accuracy": random,
            "random_minus_jspace": random - jspace,
        }
    }


def test_rank_11_12_survival_verdict_requires_both_recovered_models() -> None:
    gate = json.loads(DESIGN.read_text(encoding="utf-8"))["rank_11_12_gate"]
    metrics = {
        "primary-k10": _condition_metrics(0.88),
        "primary-k12": _condition_metrics(0.84),
        "high-k10": _condition_metrics(0.92),
        "high-k12": _condition_metrics(0.86),
    }
    verdict, clauses = classify_rank_11_12(metrics, gate)
    assert verdict == "simple_rank_11_12_evasion_not_supported"
    assert all(clauses.values())

    metrics["primary-k12"] = _condition_metrics(0.50)
    verdict, _ = classify_rank_11_12(metrics, gate)
    assert verdict == "simple_rank_11_12_evasion_supported"

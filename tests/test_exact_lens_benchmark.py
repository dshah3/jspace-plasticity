from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from jspace_plasticity.lens import benchmark_exact
from jspace_plasticity.lens.benchmark_exact import (
    compare_sample_artifacts,
    evenly_spaced_source_layers,
    load_prompt,
)
from jspace_plasticity.lens.compare_benchmarks import compare

ROOT = Path(__file__).resolve().parents[1]


def test_evenly_spaced_layers_span_the_entire_source_range() -> None:
    layers = evenly_spaced_source_layers(63, 25)
    assert len(layers) == 25
    assert len(set(layers)) == 25
    assert layers[0] == 0
    assert layers[-1] == 62
    assert all(left < right for left, right in pairwise(layers))


def test_evenly_spaced_layers_reject_impossible_count() -> None:
    with pytest.raises(ValueError, match="source-layer count"):
        evenly_spaced_source_layers(3, 4)


def test_load_prompt_hashes_exact_text(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"text": "first"}) + "\n" + json.dumps({"text": "second"})
    )
    text, digest = load_prompt(path, 1)
    assert text == "second"
    assert len(digest) == 64


def test_tp_model_load_uses_pinned_distributed_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        vision_config = None
        base_model_tp_plan = {"model.layers.*.self_attn.q_proj": "colwise"}
        use_cache = True

        def get_text_config(self) -> FakeConfig:
            return self

    class FakeModel:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config

        def to(self, _device: torch.device) -> None:
            raise AssertionError("a native-TP model must not be moved after loading")

    config = FakeConfig()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        benchmark_exact.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_exact.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )

    def fake_model_load(*_args: object, **kwargs: object) -> FakeModel:
        captured.update(kwargs)
        return FakeModel(config)

    monkeypatch.setattr(
        benchmark_exact.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_load,
    )

    benchmark_exact._load_model(
        "model",
        "revision",
        tensor_parallel=True,
        world_size=4,
        device=torch.device("cpu"),
    )

    assert "tp_plan" not in captured
    distributed_config = captured["distributed_config"]
    assert isinstance(distributed_config, benchmark_exact.DistributedConfig)
    assert distributed_config.tp_size == 4
    assert distributed_config.tp_plan is None


def _samples(rows: torch.Tensor, logits: torch.Tensor) -> dict:
    return {
        "metadata": {
            "model": "model",
            "revision": "revision",
            "prompt_sha256": "digest",
            "source_layers": [0, 2],
            "target_layer": 3,
        },
        "rows": {0: rows.clone(), 2: rows.clone() * 2},
        "final_logits": logits.clone(),
    }


def test_sample_comparison_detects_exact_equivalence() -> None:
    rows = torch.arange(24, dtype=torch.float32).reshape(3, 8) + 1
    logits = torch.tensor([1.0, 2.0, 0.0])
    metrics = compare_sample_artifacts(_samples(rows, logits), _samples(rows, logits))
    assert metrics["common_directions"] == 3
    assert metrics["global_cosine"] == pytest.approx(1.0)
    assert metrics["global_relative_l2"] == pytest.approx(0.0)
    assert metrics["forward"]["argmax_agreement"]
    assert metrics["forward"]["kl_reference_to_candidate"] == pytest.approx(0.0)


def test_sample_comparison_uses_only_common_coordinate_rows() -> None:
    reference_rows = torch.arange(16, dtype=torch.float32).reshape(2, 8) + 1
    candidate_rows = torch.cat(
        [reference_rows, torch.full((2, 8), 999.0)], dim=0
    )
    logits = torch.tensor([1.0, 2.0, 0.0])
    metrics = compare_sample_artifacts(
        _samples(candidate_rows, logits), _samples(reference_rows, logits)
    )
    assert metrics["common_directions"] == 2
    assert metrics["global_relative_l2"] == pytest.approx(0.0)


def _benchmark_result(world_size: int, seconds: float, *, equivalent: bool) -> dict:
    result = {
        "status": "completed",
        "metadata": {"world_size": world_size, "dim_batch": world_size},
        "summary": {
            "projected_exact_prompt_seconds": seconds,
            "minimum_memory_headroom_gib": 10.0,
        },
    }
    if world_size > 1:
        result["reference_equivalence"] = {"passed": equivalent}
    return result


def test_comparison_selects_throughput_not_single_prompt_latency() -> None:
    results = [
        (Path("dp.json"), _benchmark_result(1, 100.0, equivalent=True)),
        (Path("tp2.json"), _benchmark_result(2, 40.0, equivalent=True)),
        (Path("tp4.json"), _benchmark_result(4, 30.0, equivalent=True)),
    ]
    report = compare(results, prompts=1000)
    assert report["recommendations"]["8"]["label"] == "tp2-db2"
    assert report["recommendations"]["16"]["label"] == "tp2-db2"


def test_failed_tp_equivalence_is_ineligible() -> None:
    results = [
        (Path("dp.json"), _benchmark_result(1, 100.0, equivalent=True)),
        (Path("tp2.json"), _benchmark_result(2, 1.0, equivalent=False)),
    ]
    report = compare(results, prompts=1000)
    assert report["recommendations"]["8"]["label"] == "tp1-db1"

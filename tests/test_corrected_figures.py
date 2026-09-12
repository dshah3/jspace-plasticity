import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jspace_plasticity.evals.corrected_figures import capture, full_manifest, parity
from jspace_plasticity.synthetic_recovery_sft import sha256_path

spec = importlib.util.spec_from_file_location(
    "plot_corrected", Path(__file__).parents[1] / "scripts/plot_corrected_figures.py"
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_centered_cka_invariance_and_reference():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(17, 3, 9))
    actual = plot.cka_matrix(x)
    np.testing.assert_allclose(np.diag(actual), 1)
    np.testing.assert_allclose(actual, plot.cka_matrix(x * 3 + 19), atol=1e-12)
    a = x[:, 0] - x[:, 0].mean(0)
    b = x[:, 1] - x[:, 1].mean(0)
    expected = np.linalg.norm(a.T @ b) ** 2 / (
        np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    )
    assert actual[0, 1] == pytest.approx(expected)
    with pytest.raises(ValueError):
        plot.cka_matrix(np.ones((4, 2, 3)))


def test_country_bootstrap_keeps_clusters_together():
    # Duplication within every country leaves the cluster median distribution unchanged.
    values = np.array([0.0, 1.0, 10.0])
    countries = np.array(["a", "b", "c"])
    a = plot.cluster_interval(values, countries, 200)
    b = plot.cluster_interval(np.repeat(values, 5), np.repeat(countries, 5), 200)
    assert a == b


def test_capture_is_post_lesion_final_position_and_removes_hooks():
    from types import SimpleNamespace

    layer = torch.nn.Identity()
    resolved = SimpleNamespace(layers=[layer])
    lesion = layer.register_forward_hook(lambda m, i, o: o * 0)
    with capture(resolved, [0]) as states:
        layer(torch.ones(1, 5, 3))
    assert states[0].shape == (1, 1, 3)
    assert states[0].count_nonzero() == 0
    lesion.remove()
    assert not layer._forward_hooks


def test_hash_gate_rejects_same_size_corruption(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    weight = checkpoint / "weights"
    weight.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"files": [{"path": "weights", "bytes": 8, "sha256": sha256_path(weight)}]}
        )
    )
    full_manifest(manifest, sha256_path(manifest), checkpoint)
    weight.write_bytes(b"corrupt!")
    with pytest.raises(ValueError, match="SHA-256"):
        full_manifest(manifest, sha256_path(manifest), checkpoint)


def test_parity_requires_order_and_predictions():
    rows = [
        {
            "source_id": str(i),
            "expected_token_id": 1,
            "clean_predicted_token_id": 1,
            "jspace_predicted_token_id": 2,
        }
        for i in range(129)
    ]
    assert parity(rows, rows)["passed"]
    with pytest.raises(ValueError):
        parity(rows, rows[::-1])
    bad = [dict(r) for r in rows]
    bad[2]["jspace_predicted_token_id"] = 3
    with pytest.raises(ValueError):
        parity(bad, rows)


def test_measurements_require_global_parity_barrier(tmp_path):
    from jspace_plasticity.evals.corrected_figures import worker

    design = {"output_dir": str(tmp_path)}
    (tmp_path / "design.json").write_text(json.dumps(design))
    (tmp_path / "preflight.json").write_text(
        json.dumps({"status": "passed", "design_sha256": "test"})
    )
    (tmp_path / "parity-base.json").write_text(
        json.dumps({"passed": True, "design_sha256": "test"})
    )
    # The base worker cannot measure while the primary/replicate gates are missing.
    with pytest.raises(FileNotFoundError):
        worker(design, {}, "base", "measure")


def test_partial_repo_cache_is_valid_but_missing_model_shards_fail(tmp_path):
    from jspace_plasticity.evals.corrected_figures import cached_base_snapshot

    revision = "a" * 40
    base = tmp_path / "models--Qwen--Qwen3.5-4B" / "snapshots" / revision
    base.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ):
        (base / name).write_text("{}")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "model-01.safetensors"}})
    )
    shard = base / "model-01.safetensors"
    shard.write_bytes(b"test fixture")
    # Repository documentation is absent, as in the real cache.
    assert cached_base_snapshot(tmp_path, "Qwen/Qwen3.5-4B", revision) == base
    shard.unlink()
    with pytest.raises(FileNotFoundError, match="weight shard"):
        cached_base_snapshot(tmp_path, "Qwen/Qwen3.5-4B", revision)
    shard.symlink_to(base / "missing-blob")
    with pytest.raises(FileNotFoundError, match="weight shard"):
        cached_base_snapshot(tmp_path, "Qwen/Qwen3.5-4B", revision)
    with pytest.raises(ValueError, match="exact base commit"):
        cached_base_snapshot(tmp_path, "Qwen/Qwen3.5-4B", "main")

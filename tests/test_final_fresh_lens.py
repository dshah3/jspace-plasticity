import copy
import hashlib
import json
from pathlib import Path

import pytest

from jspace_plasticity.evals.final_fresh_lens import (
    canonical_sha,
    load_design,
    published_parity,
    verify_fit_config,
)

ROOT = Path(__file__).parents[1]
DESIGN = ROOT / "data/evals/q35-final-fresh-lens-20260905.json"


def test_final_fresh_design_binds_corrected_checkpoints_and_corpus():
    d = load_design(DESIGN, hashlib.sha256(DESIGN.read_bytes()).hexdigest())
    assert d["training"] is False and len(d["conditions"]) == 8
    assert d["fit"]["num_prompts"] == 500
    assert d["base_fresh"]["reused"] is True
    assert d["models"]["primary"]["path"].endswith(
        "arm-00-j-full-s100-primary/checkpoint-terminal"
    )


@pytest.mark.parametrize(
    "key,value", [("direction_convention", "legacy_weight"), ("k", 50)]
)
def test_rejects_lesion_drift_even_with_valid_hash(tmp_path, key, value):
    d = json.loads(DESIGN.read_text())
    d["lesion"][key] = value
    p = tmp_path / "changed.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Corrected sequential"):
        load_design(p, hashlib.sha256(p.read_bytes()).hexdigest())


def test_fit_validation_rejects_wrong_model_corpus_and_layers():
    d = json.loads(DESIGN.read_text())
    config = {
        k: v
        for k, v in d["fit"].items()
        if k not in ("num_prompts", "workers_per_model")
    }
    config.update(
        model=d["models"]["primary"]["path"], revision=d["revision"], world_size=4
    )
    verify_fit_config(config, d, "primary")
    for key, value in [
        ("model", "wrong-model"),
        ("corpus_sha256", "wrong"),
        ("source_layers", [22]),
    ]:
        bad = copy.deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError, match="[Ff]it"):
            verify_fit_config(bad, d, "primary")
    assert canonical_sha(config) == canonical_sha(dict(reversed(list(config.items()))))


def test_published_parity_checks_ids_not_just_correctness():
    r = dict(
        dataset="geo",
        source_id="x",
        expected_token_id=1,
        clean_predicted_token_id=1,
        jspace_predicted_token_id=2,
        random_predicted_token_id=1,
    )
    assert published_parity([r], [r])["passed"]
    bad = dict(r, jspace_predicted_token_id=3)
    with pytest.raises(ValueError, match="parity failed"):
        published_parity([bad], [r])
    with pytest.raises(ValueError, match="Duplicate"):
        published_parity([r, r], [r, r])

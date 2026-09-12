"""Hash-bound fresh-lens check of the final corrected checkpoints; no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from jspace_plasticity.config import ModelConfig
from jspace_plasticity.intervention import JSpaceAblator
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.synthetic_recovery_sft import (
    _normalized_synthetic_rows,
    _transfer_rows,
    evaluate_conditions,
    intervention_config,
    load_triage,
    paired_change,
    read_jsonl,
    sha256_path,
    write_json,
)
from jspace_plasticity.tasks.closedbook_geo import load_split


def require_sha(path, expected):
    actual = sha256_path(Path(path))
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_design(path, expected):
    require_sha(path, expected)
    d = json.loads(Path(path).read_text())
    if (
        d["decision"] != "final_corrected_fresh_lens_evaluation_authorized"
        or d["training"]
    ):
        raise ValueError("Not an authorized evaluation-only design")
    expected_conditions = [
        ("base", "published"),
        ("base", "base_fresh"),
        ("primary", "published"),
        ("primary", "primary_fresh"),
        ("primary", "base_fresh"),
        ("replicate", "published"),
        ("replicate", "replicate_fresh"),
        ("replicate", "base_fresh"),
    ]
    if [(c["model"], c["lens"]) for c in d["conditions"]] != expected_conditions:
        raise ValueError("Condition assignments drifted")
    if [c["index"] for c in d["conditions"]] != list(range(8)):
        raise ValueError("Condition indices drifted")
    expected_lesion = {
        "selection_source": "online_current",
        "projection": "sequential",
        "layers": [16, 18, 19, 20, 21, 22],
        "k": 10,
        "exclude_output_top_k": 10,
        "strength": 1.0,
        "matched_random_resampling": "per_example",
        "direction_convention": "effective_gain",
    }
    if d["lesion"] != expected_lesion:
        raise ValueError("Corrected sequential intervention required")
    if d["fit"]["num_prompts"] != 500:
        raise ValueError("Fixed 500-prompt fit required")
    return d


def verify_fit_config(config, design, model):
    for key, value in design["fit"].items():
        if key in ("num_prompts", "workers_per_model"):
            continue
        if config.get(key) != value:
            raise ValueError(f"Fit setting mismatch: {key}")
    if (
        config["model"] != design["models"][model]["path"]
        or config["revision"] != design["revision"]
    ):
        raise ValueError("Fit model binding mismatch")
    expected_world = 2 if model == "base" else design["fit"]["workers_per_model"]
    if config["world_size"] != expected_world:
        raise ValueError("Fit world size mismatch")


def verify_checkpoint(d, model):
    from jspace_plasticity.lens.fit_exact_dp import validate_model_manifest

    m = d["models"][model]
    result_path = Path(d["source_run"]) / "arms" / m["reference_arm"] / "result.json"
    require_sha(result_path, m["result_sha256"])
    result = json.loads(result_path.read_text())
    if result["saved_checkpoint"]["manifest_sha256"] != m["manifest_sha256"]:
        raise ValueError("Training result/checkpoint manifest mismatch")
    return validate_model_manifest(
        Path(m["manifest_path"]),
        expected_sha256=m["manifest_sha256"],
        model_dir=Path(m["path"]),
    )


def lens_receipt(d, key):
    if key == "published":
        p = d["evidence"]["published_lens"]
        require_sha(p["path"], p["sha256"])
        return dict(p)
    model = key.removesuffix("_fresh")
    if model == "base":
        r = d["base_fresh"]
        config_path = Path(r["config_path"])
        result_path = Path(r["result_path"])
        require_sha(config_path, r["config_sha256"])
        require_sha(result_path, r["result_sha256"])
    else:
        root = Path(d["output_dir"]) / "fits" / model
        config_path = root / "fit_config.json"
        result_path = root / "stages/0500/result.json"
    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())
    verify_fit_config(config, d, model)
    if result["status"] != "completed" or result["num_prompts"] != 500:
        raise ValueError("Fit incomplete or wrong prompt count")
    if canonical_sha(config) != result["fit_config_sha256"]:
        raise ValueError("Fit config canonical hash mismatch")
    if model != "base":
        if result["experiment_image"] != os.environ["EXPERIMENT_IMAGE"]:
            raise ValueError("Fresh fit image differs from evaluator")
        if config["model_manifest"]["sha256"] != d["models"][model]["manifest_sha256"]:
            raise ValueError("Fresh fit checkpoint mismatch")
    artifact = result["artifacts"]["lens"]
    require_sha(artifact["path"], artifact["sha256"])
    if model == "base" and (
        artifact["path"] != d["base_fresh"]["path"]
        or artifact["sha256"] != d["base_fresh"]["sha256"]
    ):
        raise ValueError("Reused base lens binding mismatch")
    return {
        **artifact,
        "n_prompts": 500,
        "fit_result_sha256": sha256_path(result_path),
        "fit_config": config,
        "reused": model == "base",
    }


def preflight(d, design_path):
    root = Path(d["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    require_sha(Path(d["source_run"]) / "summary.json", d["source_summary_sha256"])
    require_sha(d["fit"]["corpus"], d["fit"]["corpus_sha256"])
    base = lens_receipt(d, "base_fresh")
    published = lens_receipt(d, "published")
    checks = {m: verify_checkpoint(d, m) for m in ("primary", "replicate")}
    root.mkdir(parents=True)
    write_json(root / "design.json", d)
    write_json(
        root / "preflight.json",
        {
            "status": "passed",
            "design_sha256": sha256_path(design_path),
            "base_fresh": base,
            "published": published,
            "checkpoints": checks,
            "training": False,
            "experiment_image": os.environ["EXPERIMENT_IMAGE"],
        },
    )


def published_parity(records, reference):
    fields = [
        "expected_token_id",
        "clean_predicted_token_id",
        "jspace_predicted_token_id",
        "random_predicted_token_id",
    ]

    def index(rows):
        return {
            (r["dataset"], r["source_id"]): tuple(r[f] for f in fields) for r in rows
        }

    if len(index(records)) != len(records) or len(index(reference)) != len(reference):
        raise ValueError("Duplicate parity IDs")
    if index(records) != index(reference):
        raise ValueError("Published-lens parity failed")
    return {"passed": True, "rows": len(records), "fields": fields}


def counts(rows):
    out = {}
    for label, splits in [
        ("pooled", {"val", "screen"}),
        ("val", {"val"}),
        ("screen", {"screen"}),
        ("transfer", {"transfer"}),
    ]:
        subset = [r for r in rows if r["split"] in splits]
        out[label] = {
            "rows": len(subset),
            **{
                c: sum(r[c + "_exact"] for r in subset)
                for c in ("clean", "jspace", "random")
            },
        }
    return out


def run_condition(d, index, design_sha):
    condition = d["conditions"][index]
    output = Path(d["output_dir"]) / "conditions" / condition["name"]
    if output.exists():
        raise FileExistsError(output)
    m = d["models"][condition["model"]]
    if condition["model"] != "base":
        verify_checkpoint(d, condition["model"])
    lens_info = lens_receipt(d, condition["lens"])
    e = d["evidence"]
    cohorts, _, _ = load_triage(
        Path(e["triage_summary"]["path"]).parent,
        summary_sha256=e["triage_summary"]["sha256"],
        eligible_sha256=e["eligible_train_rows"]["sha256"],
    )
    data = Path("/opt/experiment") / e["data"]["path"]
    split_rows = {
        s: load_split(data, expected_sha256=e["data"]["sha256"], split=s)[0]
        for s in ("val", "screen")
    }
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    model, tokenizer, resolved = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=m["path"],
            revision=d["revision"],
            dtype="float32",
            attn_implementation="sdpa",
            gradient_checkpointing=False,
            freeze_output_head=True,
        ),
        device,
    )
    model.eval()
    normalized = _normalized_synthetic_rows(tokenizer, split_rows, cohorts)
    transfer = e["anthropic_transfer"]
    rows = (
        normalized["val"]
        + normalized["screen"]
        + _transfer_rows(
            tokenizer,
            Path("/opt/experiment") / transfer["data_path"],
            transfer["data_sha256"],
            Path("/opt/experiment") / transfer["eligibility_path"],
            transfer["eligibility_sha256"],
        )
    )
    assert len(rows) == 178
    lens = LensMatrices.load(lens_info["path"])
    assert lens.n_prompts == lens_info["n_prompts"]

    def ablator(mode):
        config = intervention_config(
            mode, Path(lens_info["path"]), direction_convention="effective_gain"
        )
        return JSpaceAblator(
            resolved, lens, config, device=device, dtype=torch.bfloat16
        )

    print(
        json.dumps({"event": "evaluation_started", "condition": condition}), flush=True
    )
    records = evaluate_conditions(
        model,
        tokenizer,
        ablator("jspace"),
        ablator("matched_random"),
        rows,
        phase="fresh_lens",
    )
    ref_root = Path(d["source_run"]) / "arms" / m["reference_arm"]
    ref_result = json.loads((ref_root / "result.json").read_text())
    require_sha(ref_root / "predictions.jsonl", ref_result["predictions_sha256"])
    reference = [
        r
        for r in read_jsonl(ref_root / "predictions.jsonl")
        if r["phase"] == m["reference_phase"] and r["split"] != "train"
    ]
    parity = (
        published_parity(records, reference)
        if condition["lens"] == "published"
        else None
    )
    # Clean outputs must remain unchanged by lens replacement.
    ref_ids = {(r["dataset"], r["source_id"]): r for r in reference}
    assert all(
        r["clean_predicted_token_id"]
        == ref_ids[(r["dataset"], r["source_id"])]["clean_predicted_token_id"]
        for r in records
    )
    output.mkdir(parents=True)
    p = output / "predictions.jsonl"
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    write_json(
        output / "result.json",
        {
            "status": "completed",
            "training": False,
            "condition": condition,
            "design_sha256": design_sha,
            "experiment_image": os.environ["EXPERIMENT_IMAGE"],
            "model": m,
            "lens": lens_info,
            "direction_convention": "effective_gain",
            "evaluation": counts(records),
            "published_parity": parity,
            "predictions_sha256": sha256_path(p),
        },
    )
    print(
        json.dumps(
            {
                "event": "evaluation_completed",
                "condition": condition,
                "counts": counts(records),
            }
        ),
        flush=True,
    )


def reduce(d, design_sha):
    root = Path(d["output_dir"])
    results = {}
    predictions = {}
    hashes = {}
    for c in d["conditions"]:
        p = root / "conditions" / c["name"]
        r = json.loads((p / "result.json").read_text())
        if (
            r["status"] != "completed"
            or r["condition"] != c
            or r["design_sha256"] != design_sha
            or r["experiment_image"] != os.environ["EXPERIMENT_IMAGE"]
        ):
            raise ValueError("Condition provenance mismatch")
        require_sha(p / "predictions.jsonl", r["predictions_sha256"])
        rows = read_jsonl(p / "predictions.jsonl")
        assert len(rows) == 178
        for row in rows:
            for mode in ("clean", "jspace", "random"):
                assert row[mode + "_exact"] == int(
                    row[mode + "_predicted_token_id"] == row["expected_token_id"]
                )
        assert counts(rows) == r["evaluation"]
        results[c["name"]] = r
        predictions[c["name"]] = rows
        hashes[c["name"]] = sha256_path(p / "result.json")
    comparisons = {}
    for model in ("primary", "replicate"):
        comparisons[model] = {}
        for cohort, splits in [
            ("pooled", {"val", "screen"}),
            ("screen", {"screen"}),
            ("transfer", {"transfer"}),
        ]:

            def select(name, selected_splits=splits):
                return [r for r in predictions[name] if r["split"] in selected_splits]

            comparisons[model][cohort] = {
                "own_fresh_minus_base_fresh": paired_change(
                    select("base-base_fresh"),
                    select(model + "-" + model + "_fresh"),
                    field="jspace_exact",
                ),
                "own_fresh_minus_published": paired_change(
                    select(model + "-published"),
                    select(model + "-" + model + "_fresh"),
                    field="jspace_exact",
                ),
            }
    out = {
        "status": "completed",
        "training": False,
        "design_sha256": design_sha,
        "experiment_image": os.environ["EXPERIMENT_IMAGE"],
        "conditions": {n: r["evaluation"] for n, r in results.items()},
        "result_sha256": hashes,
        "paired_comparisons": comparisons,
        "claim_boundary": d["claim_boundary"],
        "verdict": "Inspect fresh-base damage and recovery; mechanism remains open.",
    }
    write_json(root / "summary.json", out)
    (root / "SUMMARY.sha256").write_text(
        sha256_path(root / "summary.json") + "  summary.json\n"
    )
    (root / "_SUCCESS").touch()
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight", "condition", "reduce"])
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--design-sha256", required=True)
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    d = load_design(a.design, a.design_sha256)
    if a.action == "preflight":
        preflight(d, a.design)
    elif a.action == "condition":
        run_condition(d, a.index, a.design_sha256)
    else:
        reduce(d, a.design_sha256)


if __name__ == "__main__":
    main()

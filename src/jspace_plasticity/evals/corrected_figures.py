"""Corrected 100-step checkpoints only: hash gate, prediction gate, measurements.

No training, fitting, setting selection, or legacy checkpoint fallback.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch

from jspace_plasticity.config import ModelConfig
from jspace_plasticity.evals.final_fresh_lens import (
    load_design,
    require_sha,
    verify_checkpoint,
)
from jspace_plasticity.evals.recovery_figure_suite import _activation_metrics
from jspace_plasticity.intervention import JSpaceAblator, _hidden_from_output
from jspace_plasticity.lens import LensMatrices
from jspace_plasticity.modeling import load_model_and_tokenizer
from jspace_plasticity.synthetic_recovery_sft import (
    _encode,
    _normalized_synthetic_rows,
    evaluate_conditions,
    intervention_config,
    load_triage,
    read_jsonl,
    sha256_path,
    write_json,
)
from jspace_plasticity.tasks.closedbook_geo import load_split

MODELS = ("base", "primary", "replicate")
ROOT = Path(__file__).resolve().parents[3]


def full_manifest(path, expected, model_dir):
    require_sha(path, expected)
    manifest = json.loads(Path(path).read_text())
    for entry in manifest["files"]:
        p = Path(model_dir) / entry["path"]
        if not p.resolve().is_relative_to(Path(model_dir).resolve()):
            raise ValueError("Manifest path escapes checkpoint")
        if p.stat().st_size != entry["bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {p}")
        require_sha(p, entry["sha256"])
    if {
        str(p.relative_to(model_dir)) for p in Path(model_dir).rglob("*") if p.is_file()
    } != {e["path"] for e in manifest["files"]}:
        raise ValueError("Checkpoint contains unmanifested or missing files")
    return manifest


def parity(records, reference):
    fields = (
        "source_id",
        "expected_token_id",
        "clean_predicted_token_id",
        "jspace_predicted_token_id",
    )
    if len(records) != 129 or len({r["source_id"] for r in records}) != 129:
        raise ValueError("Expected 129 unique ordered prompts")
    if [[r[f] for f in fields] for r in records] != [
        [r[f] for f in fields] for r in reference
    ]:
        raise ValueError("Ordered prediction parity failed")
    return {"passed": True, "rows": 129, "fields": fields}


@contextlib.contextmanager
def capture(resolved, layers):
    """Install AFTER intervention hooks; retain only final-position clones."""
    states, handles = {}, []

    def hook(layer):
        def save(module, inputs, output):
            states[layer] = _hidden_from_output(output)[:, -1:, :].detach().clone()

        return save

    try:
        for layer in layers:
            handles.append(resolved.layers[layer].register_forward_hook(hook(layer)))
        yield states
        if set(states) != set(layers):
            raise ValueError("Missing decoder captures")
    finally:
        for handle in handles:
            handle.remove()


def read_config(path):
    spec = json.loads(path.read_text())
    for p, expected in spec["source_designs"].items():
        require_sha(ROOT / p, expected)
    source = "data/evals/q35-final-fresh-lens-20260905.json"
    d = load_design(ROOT / source, spec["source_designs"][source])
    if spec["training"] or spec["measurement"]["layers"] != d["lesion"]["layers"]:
        raise ValueError("Measurement design drift")
    return spec, d


def reference(spec, d, name):
    binding = spec["bindings"][name]
    root = Path(d["output_dir"]) / "conditions" / name
    require_sha(root / "result.json", binding["result_sha256"])
    require_sha(root / "predictions.jsonl", binding["predictions_sha256"])
    require_sha(binding["lens"]["path"], binding["lens"]["sha256"])
    return [
        r
        for r in read_jsonl(root / "predictions.jsonl")
        if r["split"] in ("val", "screen")
    ]


def cached_base_snapshot(cache_dir, model_id, revision):
    """Require pinned model-loading inputs, not a complete repository download.

    Preflight subsequently hashes every cached file, including the weight shards.
    """
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An exact base commit is required")
    if model_id != "Qwen/Qwen3.5-4B":
        raise ValueError("Unexpected base model")
    base = Path(cache_dir) / "models--Qwen--Qwen3.5-4B" / "snapshots" / revision
    required = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    for name in required:
        if not (base / name).is_file():
            raise FileNotFoundError(f"Missing pinned base model input: {base / name}")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("Empty base weight index")
    for name in shards:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Unexpected base weight shard path: {name}")
        if not (base / name).is_file():
            raise FileNotFoundError(f"Missing pinned base weight shard: {base / name}")
    return base


def preflight(spec, d, path):
    output = Path(spec["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    checkpoints = {}
    for model in MODELS[1:]:
        verify_checkpoint(d, model)
        m = d["models"][model]
        checkpoints[model] = full_manifest(
            m["manifest_path"], m["manifest_sha256"], m["path"]
        )
        print(f"Full checkpoint hashes verified: {model}", flush=True)
    # Model caches need not contain repository documentation or auxiliary assets.
    from huggingface_hub.constants import HF_HUB_CACHE

    base = cached_base_snapshot(
        HF_HUB_CACHE, d["models"]["base"]["path"], d["revision"]
    )
    base_files = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            actual = sha256_path(p)
            blob = p.resolve().name
            if len(blob) == 64 and actual != blob:
                raise ValueError(f"Base cached blob hash mismatch: {p}")
            base_files.append(
                {
                    "path": str(p.relative_to(base)),
                    "sha256": actual,
                    "bytes": p.stat().st_size,
                }
            )
    refs = {name: reference(spec, d, name) for name in spec["bindings"]}
    # Independently bind published replay targets to the final capability receipts.
    require_sha(Path(d["source_run"]) / "summary.json", d["source_summary_sha256"])
    for model in MODELS:
        m = d["models"][model]
        arm = Path(d["source_run"]) / "arms" / m["reference_arm"]
        result = json.loads((arm / "result.json").read_text())
        require_sha(arm / "predictions.jsonl", result["predictions_sha256"])
        original = [
            r
            for r in read_jsonl(arm / "predictions.jsonl")
            if r["phase"] == m["reference_phase"] and r["split"] in ("val", "screen")
        ]
        parity(refs[model + "-published"], original)
    ids = [r["source_id"] for r in refs["base-published"]]
    if len(ids) != 129 or any(
        [r["source_id"] for r in rows] != ids for rows in refs.values()
    ):
        raise ValueError("Source cohorts/order differ")
    require_sha(ROOT / d["evidence"]["data"]["path"], d["evidence"]["data"]["sha256"])
    sources = {
        str(p.relative_to(ROOT)): sha256_path(p)
        for directory in ("src", "scripts", "data/evals", "infra")
        for p in sorted((ROOT / directory).rglob("*"))
        if p.is_file() and "__pycache__" not in str(p)
    }
    versions = {
        p: importlib.metadata.version(p)
        for p in ("torch", "transformers", "numpy", "accelerate")
    }
    if versions["torch"] != "2.10.0+cu129" or versions["transformers"] != "5.15.0":
        raise ValueError("Executed experiment runtime required")
    output.mkdir(parents=True)
    write_json(output / "design.json", spec)
    write_json(
        output / "preflight.json",
        {
            "status": "passed",
            "design_sha256": sha256_path(path),
            "checkpoint_files": checkpoints,
            "base_snapshot": str(base),
            "base_files": base_files,
            "prompt_ids": ids,
            "source_hashes": sources,
            "versions": versions,
            "intervention": d["lesion"],
            "models": d["models"],
            "bindings": spec["bindings"],
            "image": os.environ.get("EXPERIMENT_IMAGE"),
            "model_dtype": "float32",
            "autocast_dtype": "bfloat16",
            "eval_mode": True,
            "attention": "sdpa",
        },
    )


@torch.inference_mode()
def worker(spec, d, model_name, stage):
    output = Path(spec["output_dir"])
    pre = json.loads((output / "preflight.json").read_text())
    if (
        pre["status"] != "passed"
        or json.loads((output / "design.json").read_text()) != spec
    ):
        raise ValueError("Missing or mismatched preflight")
    if stage == "measure":
        for m in MODELS:
            gate = json.loads((output / f"parity-{m}.json").read_text())
            if not gate["passed"] or gate["design_sha256"] != pre["design_sha256"]:
                raise ValueError("All models must pass parity before measurements")
    destination = output / (
        f"parity-{model_name}.json"
        if stage == "parity"
        else f"measure-{model_name}.json"
    )
    if destination.exists():
        raise FileExistsError(destination)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    m = d["models"][model_name]
    model, tokenizer, resolved = load_model_and_tokenizer(
        ModelConfig(
            name_or_path=pre["base_snapshot"] if model_name == "base" else m["path"],
            revision=d["revision"],
            dtype="float32",
            attn_implementation="sdpa",
            gradient_checkpointing=False,
            freeze_output_head=True,
        ),
        device,
    )
    model.eval()
    e = d["evidence"]
    cohorts, _, _ = load_triage(
        Path(e["triage_summary"]["path"]).parent,
        summary_sha256=e["triage_summary"]["sha256"],
        eligible_sha256=e["eligible_train_rows"]["sha256"],
    )
    split_rows = {
        s: load_split(
            ROOT / e["data"]["path"], expected_sha256=e["data"]["sha256"], split=s
        )[0]
        for s in ("val", "screen")
    }
    normalized = _normalized_synthetic_rows(tokenizer, split_rows, cohorts)
    rows = normalized["val"] + normalized["screen"]
    if [r["source_id"] for r in rows] != pre["prompt_ids"]:
        raise ValueError("Encoded cohort order differs")
    outcomes = {}
    for lens_key in ("published", model_name + "_fresh"):
        name = model_name + "-" + lens_key
        ref = reference(spec, d, name)
        info = spec["bindings"][name]["lens"]
        lens = LensMatrices.load(info["path"])

        def ablator(mode, lens=lens, info=info):
            return JSpaceAblator(
                resolved,
                lens,
                intervention_config(
                    mode, Path(info["path"]), direction_convention="effective_gain"
                ),
                device=device,
                dtype=torch.bfloat16,
            )

        jspace = ablator("jspace")
        if stage == "parity":
            records = evaluate_conditions(
                model,
                tokenizer,
                jspace,
                ablator("matched_random"),
                rows,
                phase="figure_parity",
            )
            outcomes[name] = parity(records, ref)
            if [r["random_predicted_token_id"] for r in records] != [
                r["random_predicted_token_id"] for r in ref
            ]:
                raise ValueError("Random-control parity failed")
            write_json(output / f"parity-predictions-{name}.json", records)
        else:
            layers = (
                list(range(len(resolved.layers)))
                if lens_key == "published"
                else spec["measurement"]["layers"]
            )
            metrics, predictions, clean_array, lesion_array = [], [], [], []
            for i, row in enumerate(rows):
                encoded = _encode(tokenizer, row["prompt"], device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    plan = jspace.build_plan(
                        model, encoded["input_ids"], encoded["attention_mask"]
                    )
                    with capture(resolved, layers) as clean:
                        clean_logits = (
                            model(**encoded, use_cache=False).logits[0, -1].float()
                        )
                    with jspace.apply(plan):
                        with capture(resolved, layers) as lesion:
                            lesion_logits = (
                                model(**encoded, use_cache=False).logits[0, -1].float()
                            )
                if set(plan.selected_token_ids) != set(spec["measurement"]["layers"]):
                    raise ValueError("Lesion did not fire at every intervention layer")
                predicted = {
                    **row,
                    "clean_predicted_token_id": int(clean_logits.argmax()),
                    "jspace_predicted_token_id": int(lesion_logits.argmax()),
                }
                predictions.append(predicted)
                if (
                    int(plan.clean_next_logits[0].argmax())
                    != predicted["clean_predicted_token_id"]
                ):
                    raise ValueError("Clean capture changed prediction")
                # Fail immediately on prediction drift, and recheck whole ordered cohort below.  # noqa: E501
                for field in (
                    "expected_token_id",
                    "clean_predicted_token_id",
                    "jspace_predicted_token_id",
                ):
                    if predicted[field] != ref[i][field]:
                        raise ValueError(f"Capture parity failed: {name}/{i}/{field}")
                values = _activation_metrics(
                    resolved,
                    jspace.jacobians,
                    clean,
                    lesion,
                    spec["measurement"]["layers"],
                )
                metrics.append(
                    {
                        "source_id": row["source_id"],
                        "cluster_id": row["cluster_id"],
                        "layers": values,
                    }
                )
                clean_array.append(
                    torch.stack([clean[k][0, 0].float().cpu() for k in layers]).numpy()
                )
                lesion_array.append(
                    torch.stack([lesion[k][0, 0].float().cpu() for k in layers]).numpy()
                )
                if i % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "model": model_name,
                                "lens": lens_key,
                                "completed": i + 1,
                                "total": 129,
                            }
                        ),
                        flush=True,
                    )
            outcomes[name] = parity(predictions, ref)
            np.savez_compressed(
                output / f"activations-{name}.npz",
                clean=np.stack(clean_array),
                lesioned=np.stack(lesion_array),
                layers=np.array(layers),
                prompt_ids=np.array(pre["prompt_ids"]),
                country_ids=np.array([str(r["cluster_id"]) for r in rows]),
            )
            write_json(output / f"metrics-{name}.json", metrics)
            write_json(output / f"capture-predictions-{name}.json", predictions)
        del jspace, lens
    write_json(
        destination,
        {
            "passed": True,
            "design_sha256": pre["design_sha256"],
            "conditions": outcomes,
            "final_norm_class": type(resolved.final_norm).__module__
            + "."
            + type(resolved.final_norm).__name__,
            "gpu": torch.cuda.get_device_name(0),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "parity", "measure"])
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    spec, d = read_config(args.design)
    if args.stage == "preflight":
        preflight(spec, d, args.design)
    else:
        if args.model is None:
            parser.error("--model required")
        worker(spec, d, args.model, args.stage)


if __name__ == "__main__":
    main()

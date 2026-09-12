"""Blocking HF/vLLM equivalence check for one fixed lesion artifact.

This is a GPU preflight. It never trains or changes model weights, but its
receipt is bound to the exact model, projection SHA-256, and container image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from jspace_plasticity.lesion.equivalence import argmax_gate
from jspace_plasticity.lesion.hf import install_hf_projection
from jspace_plasticity.lesion.projection import ProjectionArtifact
from jspace_plasticity.lesion.vllm import install_vllm_projection
from jspace_plasticity.tasks.cipher import CipherTask
from jspace_plasticity.trl_config import load_trl_config
from jspace_plasticity.trl_vllm import engine_arguments


def _hf_next(
    model: Any, encoded: dict[str, torch.Tensor]
) -> tuple[list[int], list[float], torch.Tensor]:
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[:, -1].float()
    logprobs = logits.log_softmax(dim=-1)
    token_ids = logits.argmax(dim=-1)
    selected = logprobs.gather(1, token_ids[:, None]).squeeze(1)
    return token_ids.cpu().tolist(), selected.cpu().tolist(), logprobs.cpu()


def _vllm_next(llm: Any, prompt_ids: list[list[int]]) -> tuple[list[int], list[float]]:
    from vllm import SamplingParams

    outputs = llm.generate(
        [{"prompt_token_ids": item} for item in prompt_ids],
        SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=1),
        use_tqdm=False,
    )
    token_ids: list[int] = []
    logprobs: list[float] = []
    for output in outputs:
        completion = output.outputs[0]
        token_id = int(completion.token_ids[0])
        token_ids.append(token_id)
        logprobs.append(float(completion.logprobs[0][token_id].logprob))
    return token_ids, logprobs


def _agreement(left: list[int], right: list[int]) -> float:
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def _mean_abs(left: list[float], right: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def run(config_path: Path, output: Path, *, prompts: int) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM

    config = load_trl_config(config_path)
    if config.lesion.condition == "sham":
        raise ValueError("equivalence preflight requires a lesioned config")
    artifact_path = Path(str(config.lesion.artifact_path))
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    artifact = ProjectionArtifact.load(
        artifact_path, expected_model_id=config.model.name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    task = CipherTask(config.task)
    texts = [task.sample("val_id", index, stage="A").prompt for index in range(prompts)]
    prompt_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in texts
    ]
    encoded = tokenizer(
        texts,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    # Transformers' direct forward otherwise assigns positions across left pad
    # tokens, whereas vLLM starts every unpadded request at position zero.
    position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    encoded["position_ids"] = position_ids
    for row, expected in zip(
        encoded["input_ids"], prompt_ids, strict=True
    ):
        actual = row[row != tokenizer.pad_token_id].tolist()
        if actual != expected:
            raise RuntimeError("HF and vLLM prompt tokenization disagrees")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        dtype=config.model.dtype,
        attn_implementation=config.model.attn_implementation,
        device_map="cuda",
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    hf_clean_ids, hf_clean_logprobs, hf_clean_full = _hf_next(model, encoded)
    install_hf_projection(model, artifact)
    hf_lesion_ids, hf_lesion_logprobs, hf_lesion_full = _hf_next(model, encoded)

    arguments = engine_arguments(config)
    arguments["enforce_eager"] = True
    llm = LLM(**arguments)
    vllm_clean_ids, vllm_clean_logprobs = _vllm_next(llm, prompt_ids)
    install_vllm_projection(
        llm,
        artifact_path=artifact_path,
        model_id=config.model.name_or_path,
    )
    vllm_lesion_ids, vllm_lesion_logprobs = _vllm_next(llm, prompt_ids)

    clean_agreement = _agreement(hf_clean_ids, vllm_clean_ids)
    lesion_agreement = _agreement(hf_lesion_ids, vllm_lesion_ids)
    clean_delta = _mean_abs(hf_clean_logprobs, vllm_clean_logprobs)
    lesion_delta = _mean_abs(hf_lesion_logprobs, vllm_lesion_logprobs)
    clean_mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(hf_clean_ids, vllm_clean_ids, strict=True)
        )
        if left != right
    ]
    lesion_mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(hf_lesion_ids, vllm_lesion_ids, strict=True)
        )
        if left != right
    ]
    changed = sum(
        clean != lesion
        for clean, lesion in zip(hf_clean_ids, hf_lesion_ids, strict=True)
    ) / prompts
    clean_vs_lesion_kl = float(
        (hf_clean_full.exp() * (hf_clean_full - hf_lesion_full))
        .sum(dim=-1)
        .mean()
    )
    argmax_passed, argmax_criterion = argmax_gate(
        clean_agreement=clean_agreement,
        lesion_agreement=lesion_agreement,
        smoke_fixture=bool(artifact.metadata.get("smoke_fixture", False)),
    )
    passed = (
        argmax_passed
        and lesion_delta <= 2 * max(clean_delta, 1e-6)
        and clean_vs_lesion_kl > 1e-5
    )
    result = {
        "schema_version": 2,
        "model_id": config.model.name_or_path,
        "projection_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "experiment_image": os.environ.get("EXPERIMENT_IMAGE"),
        "prompts": prompts,
        "clean_argmax_agreement": clean_agreement,
        "clean_argmax_mismatch_indices": clean_mismatches,
        "lesion_argmax_agreement": lesion_agreement,
        "lesion_argmax_mismatch_indices": lesion_mismatches,
        "argmax_criterion": argmax_criterion,
        "clean_mean_abs_logprob_delta": clean_delta,
        "lesion_mean_abs_logprob_delta": lesion_delta,
        "clean_vs_lesion_argmax_change_frac": changed,
        "clean_vs_lesion_kl": clean_vs_lesion_kl,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"lesion equivalence preflight failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompts", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, prompts=args.prompts), indent=2))


if __name__ == "__main__":
    main()

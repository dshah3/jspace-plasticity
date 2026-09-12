"""Boot the real vLLM engine and generate before any training starts.

Every VERL-era A100 failure happened inside the rollout engine — dynamic
loading, Triton's C compiler, FlashInfer's nvcc, cuRAND headers — and each cost
a whole job because the preflight only checked imports. This preflight
traverses the same corridor the trainer will: engine construction, model load,
kernel selection, CUDA-graph capture, stochastic sampling and greedy sampling.
It runs on one process before the distributed trainer launches, so a failure
costs about two minutes instead of a full job.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from jspace_plasticity.caesar_cipher import CaesarCipherTask, CaesarTaskConfig
from jspace_plasticity.gsm8k import Gsm8kTaskConfig, direct_prompt
from jspace_plasticity.silent_arithmetic import (
    ArithmeticTaskConfig,
    SilentArithmeticTask,
)
from jspace_plasticity.tasks.cipher import CipherTask, CipherTaskConfig
from jspace_plasticity.trl_config import TrlExperimentConfig, load_trl_config
from jspace_plasticity.trl_vllm import engine_arguments

PROBE_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Answer with one short sentence. Which single word names a tree: "
            "cedar or bicycle?"
        ),
    }
]


def _engine(config: TrlExperimentConfig) -> Any:
    from vllm import LLM

    from jspace_plasticity.lesion.vllm import install_vllm_projection

    if config.lesion.condition != "sham":
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(**engine_arguments(config))
    if config.lesion.condition != "sham":
        install_vllm_projection(
            llm,
            artifact_path=str(config.lesion.artifact_path),
            model_id=config.model.name_or_path,
        )
    return llm


def preflight(config: TrlExperimentConfig, *, max_tokens: int = 32) -> dict[str, Any]:
    """Return engine diagnostics, raising if any production path is broken."""

    from transformers import AutoProcessor
    from vllm import SamplingParams

    from jspace_plasticity.trl_data import prompt_token_ids, text_tokenizer

    # Match GRPOTrainer's processing boundary.  Qwen3.5 resolves to a
    # Qwen3VLProcessor here, so this catches tokenizer-only callback code before
    # the distributed trainer is started.
    processor = AutoProcessor.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if isinstance(config.task, ArithmeticTaskConfig):
        tokenizer = text_tokenizer(processor)
        task = SilentArithmeticTask(config.task, tokenizer)
        probe_prompt: Any = task.sample("eval", 0).prompt
        prompt_ids = prompt_token_ids(processor, probe_prompt)
        generation_tokens = config.rollout.max_response_tokens
    elif isinstance(config.task, CaesarTaskConfig):
        task = CaesarCipherTask(config.task)
        probe_prompt = task.sample("eval", 0).prompt
        prompt_ids = prompt_token_ids(processor, probe_prompt)
        generation_tokens = config.rollout.max_response_tokens
    elif isinstance(config.task, Gsm8kTaskConfig):
        probe_prompt = [
            {
                "role": "user",
                "content": direct_prompt(
                    "A shop sold 12 pencils in the morning and 7 later. "
                    "How many pencils did it sell?"
                ),
            }
        ]
        prompt_ids = prompt_token_ids(
            processor,
            probe_prompt,
            enable_thinking=False,
        )
        generation_tokens = config.rollout.max_response_tokens
    elif isinstance(config.task, CipherTaskConfig):
        task = CipherTask(config.task)
        probe_prompt = task.sample("val_id", 0, stage="A").prompt
        prompt_ids = prompt_token_ids(processor, probe_prompt)
        generation_tokens = config.rollout.max_response_tokens
    else:
        probe_prompt = PROBE_MESSAGES
        prompt_ids = prompt_token_ids(
            processor,
            probe_prompt,
            enable_thinking=config.rollout.enable_thinking,
        )
        generation_tokens = max_tokens

    started = time.monotonic()
    llm = _engine(config)
    load_seconds = time.monotonic() - started

    passes = {
        # The production sampler exercises the stochastic kernels, which is a
        # different code path from greedy decoding.
        "stochastic": SamplingParams(
            n=config.rollout.num_generations,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            max_tokens=generation_tokens,
            seed=config.run.seed,
        ),
        # Greedy decoding is what the held-out validation callback uses.
        "greedy": SamplingParams(
            n=1, temperature=0.0, top_p=1.0, max_tokens=generation_tokens
        ),
    }

    result: dict[str, Any] = {
        "model": config.model.name_or_path,
        "engine_load_seconds": round(load_seconds, 3),
        "max_model_length": config.max_model_length,
        "gpu_memory_utilization": config.rollout.gpu_memory_utilization,
        "enforce_eager": config.rollout.enforce_eager,
        "prompt_style": (
            "raw_next_token"
            if isinstance(config.task, ArithmeticTaskConfig)
            else "raw_direct_cipher_rehab"
            if isinstance(config.task, CipherTaskConfig)
            else "raw_direct_caesar"
            if isinstance(config.task, CaesarTaskConfig)
            else "chat_direct_gsm8k"
            if isinstance(config.task, Gsm8kTaskConfig)
            else "conversational"
        ),
        "generation_tokens": generation_tokens,
    }
    for name, sampling in passes.items():
        started = time.monotonic()
        outputs = llm.generate(
            [{"prompt_token_ids": list(prompt_ids)}],
            sampling_params=sampling,
            use_tqdm=False,
        )
        texts = [completion.text for completion in outputs[0].outputs]
        if len(texts) != sampling.n or not any(text.strip() for text in texts):
            raise RuntimeError(
                f"{name} sampling produced no usable completion: {texts!r}"
            )
        result[f"{name}_completions"] = len(texts)
        result[f"{name}_seconds"] = round(time.monotonic() - started, 3)
        result[f"{name}_sample_chars"] = len(texts[0])

    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_trl_config(args.config)
    result = preflight(config, max_tokens=args.max_tokens)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

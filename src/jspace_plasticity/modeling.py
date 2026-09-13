"""Hugging Face loading and decoder-layout discovery without freezing weights."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
)

from jspace_plasticity.config import ModelConfig


@dataclass(frozen=True)
class ModelLayout:
    text_path: str
    layers_name: str = "layers"
    norm_name: str = "norm"
    embed_name: str = "embed_tokens"
    head_name: str = "lm_head"


@dataclass(frozen=True)
class ResolvedModel:
    root: nn.Module
    text: nn.Module
    layers: nn.ModuleList
    final_norm: nn.Module
    embedding: nn.Module
    lm_head: nn.Module
    layout: ModelLayout


_LAYOUTS = (
    ModelLayout("model"),
    ModelLayout("model.language_model"),
    ModelLayout("language_model"),
    ModelLayout("model", norm_name="final_layernorm"),
    ModelLayout("transformer", layers_name="h", norm_name="ln_f", embed_name="wte"),
    ModelLayout(
        "gpt_neox",
        norm_name="final_layer_norm",
        embed_name="embed_in",
        head_name="embed_out",
    ),
)


def auto_model_class_for_config(hf_config: Any) -> Any:
    """Select the text or multimodal causal-LM auto class for a checkpoint."""
    return (
        AutoModelForMultimodalLM
        if getattr(hf_config, "vision_config", None) is not None
        else AutoModelForCausalLM
    )


def _resolve_path(root: Any, path: str) -> Any:
    return functools.reduce(getattr, path.split("."), root)


def resolve_model(model: nn.Module) -> ResolvedModel:
    for layout in _LAYOUTS:
        try:
            text = _resolve_path(model, layout.text_path)
        except AttributeError:
            continue
        required = (layout.layers_name, layout.norm_name, layout.embed_name)
        if not all(hasattr(text, name) for name in required):
            continue
        if not hasattr(model, layout.head_name):
            continue
        layers = getattr(text, layout.layers_name)
        if not isinstance(layers, nn.ModuleList):
            continue
        return ResolvedModel(
            root=model,
            text=text,
            layers=layers,
            final_norm=getattr(text, layout.norm_name),
            embedding=getattr(text, layout.embed_name),
            lm_head=getattr(model, layout.head_name),
            layout=layout,
        )
    raise ValueError(
        f"could not locate decoder blocks, final norm, embedding, and LM head in "
        f"{type(model).__name__}"
    )


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(
    config: ModelConfig, device: torch.device
) -> tuple[nn.Module, Any, ResolvedModel]:
    revision_kwargs = (
        {} if Path(config.name_or_path).is_dir() else {"revision": config.revision}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.name_or_path,
        trust_remote_code=config.trust_remote_code,
        **revision_kwargs,
    )
    hf_config = AutoConfig.from_pretrained(
        config.name_or_path,
        trust_remote_code=config.trust_remote_code,
        **revision_kwargs,
    )
    kwargs: dict[str, Any] = {
        "config": hf_config,
        "dtype": torch_dtype(config.dtype),
        "trust_remote_code": config.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    auto_model = auto_model_class_for_config(hf_config)
    model = auto_model.from_pretrained(config.name_or_path, **revision_kwargs, **kwargs)
    model.to(device)
    model.config.use_cache = False
    text_config = model.config.get_text_config()
    text_config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    resolved = resolve_model(model)
    # Qwen3.5 checkpoints include a vision tower and an auxiliary MTP block.
    # Text-only RL does not execute either. Freeze every parameter outside the
    # decoder stack, embeddings, final norm, and unembedding so DDP sees no
    # unexpected unused trainable parameters.
    trained_modules = (
        resolved.layers,
        resolved.embedding,
        resolved.final_norm,
        resolved.lm_head,
    )
    trained_parameter_ids = {
        id(parameter) for module in trained_modules for parameter in module.parameters()
    }
    for parameter in model.parameters():
        if id(parameter) not in trained_parameter_ids:
            parameter.requires_grad_(False)

    if config.freeze_output_head:
        for parameter in resolved.final_norm.parameters():
            parameter.requires_grad_(False)
        for parameter in resolved.lm_head.parameters():
            parameter.requires_grad_(False)
        # If input and output embeddings are tied, the call above freezes both.
        if resolved.lm_head.weight.data_ptr() == resolved.embedding.weight.data_ptr():
            for parameter in resolved.embedding.parameters():
                parameter.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError("model has no trainable parameters")
    return model, tokenizer, resolved


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

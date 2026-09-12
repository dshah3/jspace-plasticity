"""Strict, typed experiment configuration."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    seed: int = 17
    smoke_test: bool = False
    resume_from: str | None = None
    save_optimizer: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str | None = "sdpa"
    gradient_checkpointing: bool = True
    freeze_output_head: bool = False
    trust_remote_code: bool = False


@dataclass(frozen=True)
class TaskConfig:
    variant: Literal["walk", "latent_branch"] = "walk"
    num_nodes: int = 8
    num_relations: int = 3
    train_hops: int = 6
    eval_hops: int = 6
    ood_hops: int = 8
    # The latent-branch variant first follows the ordinary route, then uses the
    # reached (unreported) node to select one of `program_count` continuation
    # routes. These fields are inert for the historical walk task.
    branch_hops: int = 0
    program_count: int = 0
    train_seed: int = 10_000
    eval_seed: int = 20_000
    heldout_seed: int = 30_000
    max_prompt_tokens: int = 384
    shuffle_tables: bool = True

    def validate(self) -> None:
        if self.num_nodes < 4:
            raise ValueError("task.num_nodes must be at least 4")
        if self.num_relations < 2:
            raise ValueError("task.num_relations must be at least 2")
        if min(self.train_hops, self.eval_hops, self.ood_hops) < 2:
            raise ValueError("all task hop counts must be at least 2")
        if self.max_prompt_tokens < 1:
            raise ValueError("task.max_prompt_tokens must be positive")
        if self.variant == "walk":
            if self.branch_hops != 0 or self.program_count != 0:
                raise ValueError("walk tasks require branch_hops=0 and program_count=0")
            return
        if self.variant == "latent_branch":
            if self.branch_hops < 1:
                raise ValueError("latent_branch tasks require branch_hops >= 1")
            if not 2 <= self.program_count <= self.num_nodes:
                raise ValueError(
                    "latent_branch program_count must be in [2, num_nodes]"
                )
            if self.program_count > self.num_relations**self.branch_hops:
                raise ValueError(
                    "latent_branch program_count exceeds the number of unique "
                    "continuation routes"
                )
            return
        raise ValueError(f"unsupported task.variant: {self.variant!r}")


@dataclass(frozen=True)
class InterventionConfig:
    enabled: bool = False
    lens_path: str | None = None
    layers: list[int] = field(default_factory=list)
    k: int = 10
    exclude_output_top_k: int = 10
    selection_source: Literal["clean_trajectory", "online_current"] = "clean_trajectory"
    projection: Literal["sequential", "joint", "orthogonal_span"] = "sequential"
    direction_convention: Literal["effective_gain", "legacy_weight"] = "effective_gain"
    strength: float = 1.0
    ridge: float = 1e-4
    # "matched_random" perturbs along one random direction whose norm equals the
    # J-space removal at the same (layer, position) -- the paper's "perturbing
    # along a random direction" control.  "matched_random_subspace" instead
    # projects out k random orthonormal directions, matching the rank and the
    # norm-reducing geometry of the J-space projection rather than its norm.
    # "shrink_non_j" shrinks the projected residual by the removal norm, capped
    # by the residual norm. Coordinate preservation requires an orthogonal
    # projection; sequential/joint modes do not guarantee a true complement.
    control: Literal[
        "none", "matched_random", "matched_random_subspace", "shrink_non_j"
    ] = "none"
    # Random controls draw from `random_seed + layer`.  With "fixed" every
    # example in a run reuses the same draw, so the control is a single random
    # sample and its variance is unestimable; "per_example" mixes a stable hash
    # of the input ids into the seed so each prompt gets an independent draw.
    control_resample: Literal["fixed", "per_example"] = "per_example"
    random_seed: int = 91_337


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 1_000
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    group_size: int = 16
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 50
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 1.0
    log_steps: int = 1
    plot_steps: int = 20
    eval_steps: int = 50
    save_steps: int = 50
    eval_examples: int = 256
    initial_clean_accuracy_min: float | None = None
    initial_clean_accuracy_max: float | None = None
    initial_intervention_accuracy_min: float | None = None
    initial_intervention_accuracy_max: float | None = None


@dataclass(frozen=True)
class DistributedConfig:
    backend: Literal["nccl", "gloo"] = "nccl"
    timeout_minutes: int = 30


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 160
    rolling_window: int = 20


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    task: TaskConfig
    intervention: InterventionConfig
    training: TrainingConfig
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)

    def validate(self) -> None:
        if not self.run.name.strip():
            raise ValueError("run.name must not be empty")
        if self.training.per_device_batch_size != 1:
            raise ValueError(
                "per_device_batch_size must be 1; intervention plans are intentionally "
                "built per prompt"
            )
        self.task.validate()
        if self.training.group_size < 2:
            raise ValueError("training.group_size must be at least 2")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.max_steps < 1:
            raise ValueError("training.max_steps must be positive")
        for name in ("log_steps", "plot_steps", "eval_steps", "save_steps"):
            if getattr(self.training, name) < 1:
                raise ValueError(f"training.{name} must be positive")
        for name in (
            "initial_clean_accuracy_min",
            "initial_clean_accuracy_max",
            "initial_intervention_accuracy_min",
            "initial_intervention_accuracy_max",
        ):
            value = getattr(self.training, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"training.{name} must be between zero and one")
        for prefix in ("clean", "intervention"):
            minimum = getattr(self.training, f"initial_{prefix}_accuracy_min")
            maximum = getattr(self.training, f"initial_{prefix}_accuracy_max")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(
                    f"initial {prefix} accuracy minimum exceeds its maximum"
                )
        if (
            self.training.initial_intervention_accuracy_min is not None
            or self.training.initial_intervention_accuracy_max is not None
        ) and not self.intervention.enabled:
            raise ValueError(
                "initial intervention accuracy gates require an enabled intervention"
            )
        if self.intervention.enabled:
            if not self.intervention.lens_path:
                raise ValueError("an enabled intervention requires lens_path")
            if not self.intervention.layers:
                raise ValueError(
                    "an enabled intervention requires explicit, CKA-verified layers"
                )
            if len(set(self.intervention.layers)) != len(self.intervention.layers):
                raise ValueError("intervention.layers contains duplicates")
            if min(self.intervention.layers) < 0:
                raise ValueError("intervention.layers must be non-negative")
            if self.intervention.k < 1:
                raise ValueError("intervention.k must be positive")
            if self.intervention.exclude_output_top_k < 0:
                raise ValueError("exclude_output_top_k must be non-negative")
            if self.intervention.selection_source not in (
                "clean_trajectory",
                "online_current",
            ):
                raise ValueError("unsupported intervention.selection_source")
            if self.intervention.direction_convention not in (
                "effective_gain",
                "legacy_weight",
            ):
                raise ValueError("unsupported intervention.direction_convention")
            if self.intervention.projection not in (
                "sequential",
                "joint",
                "orthogonal_span",
            ):
                raise ValueError("unsupported intervention.projection")
            if self.intervention.control not in (
                "none",
                "matched_random",
                "matched_random_subspace",
                "shrink_non_j",
            ):
                raise ValueError("unsupported intervention.control")
            if self.intervention.control_resample not in (
                "fixed",
                "per_example",
            ):
                raise ValueError("unsupported intervention.control_resample")
            if not 0.0 <= self.intervention.strength <= 1.5:
                raise ValueError("intervention.strength must be between 0 and 1.5")
            if self.intervention.ridge <= 0:
                raise ValueError("intervention.ridge must be positive")
            if self.intervention.lens_path == "identity" and not self.run.smoke_test:
                raise ValueError("the identity lens is restricted to smoke tests")

    @property
    def output_dir(self) -> Path:
        return Path(_expand_path(self.run.output_dir))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _expand_path(value: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(value)))


def _expand_reference(value: str) -> str:
    if value.startswith(("/", ".", "~")) or "$" in value:
        return _expand_path(value)
    return value


def _construct(cls: type[T], raw: Any, section: str) -> T:
    if not isinstance(raw, dict):
        raise TypeError(f"configuration section {section!r} must be a mapping")
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {unknown}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise TypeError(f"invalid {section} configuration: {exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file, reject unknown keys, and validate invariants."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("top-level configuration must be a mapping")

    expected = {
        "run",
        "model",
        "task",
        "intervention",
        "training",
        "distributed",
        "plotting",
    }
    unknown = sorted(set(raw) - expected)
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {unknown}")

    required = {"run", "model", "task", "intervention", "training"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"missing configuration sections: {missing}")

    run = _construct(RunConfig, raw["run"], "run")
    run = dataclasses.replace(
        run,
        output_dir=_expand_path(run.output_dir),
        resume_from=(
            _expand_path(run.resume_from) if run.resume_from is not None else None
        ),
    )
    model = _construct(ModelConfig, raw["model"], "model")
    model = dataclasses.replace(
        model, name_or_path=_expand_reference(model.name_or_path)
    )
    intervention = _construct(InterventionConfig, raw["intervention"], "intervention")
    if intervention.lens_path not in (None, "identity"):
        intervention = dataclasses.replace(
            intervention,
            lens_path=_expand_reference(intervention.lens_path or ""),
        )

    config = ExperimentConfig(
        run=run,
        model=model,
        task=_construct(TaskConfig, raw["task"], "task"),
        intervention=intervention,
        training=_construct(TrainingConfig, raw["training"], "training"),
        distributed=_construct(
            DistributedConfig, raw.get("distributed", {}), "distributed"
        ),
        plotting=_construct(PlotConfig, raw.get("plotting", {}), "plotting"),
    )
    config.validate()
    return config

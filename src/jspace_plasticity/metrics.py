"""Rank-zero CSV metrics with stable, analysis-friendly schemas."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from jspace_plasticity.config import ExperimentConfig

LOSS_FIELDS = (
    "step",
    "total_loss",
    "policy_loss",
    "entropy",
    "grad_norm",
    "learning_rate",
)
REWARD_FIELDS = (
    "step",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "success_rate",
    "advantage_std",
    "zero_variance_fraction",
)
EVALUATION_FIELDS = (
    "step",
    "split",
    "condition",
    "examples",
    "chance_accuracy",
    "candidate_accuracy",
    "full_vocabulary_accuracy",
    "mean_answer_probability",
    "prediction_match_to_clean",
    "kl_from_clean",
)


@dataclass
class CsvSeries:
    path: Path
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def append(self, values: dict[str, Any]) -> None:
        unknown = sorted(set(values) - set(self.fields))
        missing = sorted(set(self.fields) - set(values))
        if unknown or missing:
            raise ValueError(
                f"metric schema mismatch for {self.path.name}: "
                f"unknown={unknown}, missing={missing}"
            )
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writerow(values)
            handle.flush()


class MetricsWriter:
    def __init__(self, output_dir: Path) -> None:
        metrics_dir = output_dir / "metrics"
        self.losses = CsvSeries(metrics_dir / "losses.csv", LOSS_FIELDS)
        self.rewards = CsvSeries(metrics_dir / "rewards.csv", REWARD_FIELDS)
        self.evaluations = CsvSeries(metrics_dir / "evaluations.csv", EVALUATION_FIELDS)

    def write_loss(self, values: dict[str, Any]) -> None:
        self.losses.append(values)

    def write_reward(self, values: dict[str, Any]) -> None:
        self.rewards.append(values)

    def write_evaluation(self, values: dict[str, Any]) -> None:
        self.evaluations.append(values)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_url(distribution: str) -> dict[str, Any] | None:
    try:
        raw = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(raw) if raw else None


def _nvidia_smi() -> list[dict[str, str]] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    devices: list[dict[str, str]] = []
    for row in result.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 3:
            devices.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "memory_mib": fields[2],
                }
            )
    return devices


def write_environment_report(
    output_dir: Path,
    project_dir: Path,
    config: ExperimentConfig | None = None,
) -> None:
    lens_path = (
        Path(config.intervention.lens_path)
        if config is not None
        and config.intervention.enabled
        and config.intervention.lens_path not in (None, "identity")
        else None
    )
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            name: _version(name)
            for name in (
                "jspace-plasticity",
                "jlens",
                "matplotlib",
                "numpy",
                "pyyaml",
                "safetensors",
                "transformers",
                "uv",
            )
        },
        "direct_urls": {"jlens": _direct_url("jlens")},
        "git_revision": _git_revision(project_dir),
        "uv_lock_sha256": _sha256(project_dir / "uv.lock"),
        "experiment": (
            {
                "run_name": config.run.name,
                "model": config.model.name_or_path,
                "lens_path": str(lens_path) if lens_path else None,
                "lens_sha256": _sha256(lens_path) if lens_path else None,
            }
            if config is not None
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "environment.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

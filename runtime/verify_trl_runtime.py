"""Verify the locked TRL/vLLM runtime at build time and on a real GPU.

The VERL era taught that import-shaped checks miss behavior-shaped failures, so
every gate here *does* the thing it claims to verify: it compiles and links a
real CUDA translation unit, resolves libraries in a freshly exec'd process, and
validates the trainer argument mapping against the installed ``GRPOConfig``
dataclass rather than against documentation.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from jspace_plasticity.config import TaskConfig
from jspace_plasticity.cuda_compat import driver_supports_runtime
from jspace_plasticity.trl_config import (
    TrlDataConfig,
    TrlExperimentConfig,
    TrlModelConfig,
    TrlOptimizationConfig,
    TrlRolloutConfig,
    TrlRunConfig,
)
from jspace_plasticity.trl_train import grpo_arguments
from jspace_plasticity.trl_vllm import engine_arguments
from jspace_plasticity.verl_reward import compute_score

EXPECTED_VERSIONS = {
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "flashinfer-python": "0.6.6",
    "jlens": "0.1.0",
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "torch": "2.10.0+cu129",
    "transformers": "5.15.0",
    "trl": "1.9.2",
    "vllm": "0.19.1",
}
SUPPORTED_AMPERE_CAPABILITIES = {(8, 0), (8, 6)}
CUDA_ARCH = os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "8.0").split()[0]


def _require_source_symbols(
    distribution: str, relative_path: str, symbols: tuple[str, ...]
) -> Path:
    """Check package source without importing GPU-only native extensions."""

    metadata = importlib.metadata.distribution(distribution)
    path = Path(metadata.locate_file(relative_path))
    if not path.is_file():
        raise RuntimeError(f"missing {distribution} runtime file: {relative_path}")
    source = path.read_text(encoding="utf-8")
    missing = [symbol for symbol in symbols if symbol not in source]
    if missing:
        raise RuntimeError(
            f"{distribution} file {relative_path} is missing symbols: {missing}"
        )
    return path


def _verify_locked_packages() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS
    }
    for distribution, expected in EXPECTED_VERSIONS.items():
        if versions[distribution] != expected:
            raise RuntimeError(
                f"unexpected {distribution} version: {versions[distribution]} "
                f"(expected {expected})"
            )
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"expected a CUDA 12.9 PyTorch build: {torch.version.cuda}")
    if not (torch.distributed.is_available() and torch.distributed.is_nccl_available()):
        raise RuntimeError("PyTorch distributed/NCCL support is unavailable")

    installed = {
        distribution.metadata["Name"].lower()
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    cuda_13 = sorted(name for name in installed if "cu13" in name)
    if cuda_13:
        raise RuntimeError(f"CUDA 13 distributions leaked into cu129 image: {cuda_13}")
    return versions


def _run_checked(command: list[str], *, what: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command, capture_output=True, check=False, text=True, cwd=cwd
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{what} failed: exit={completed.returncode}: {detail[-2000:]}"
        )
    return completed.stdout


def _verify_runtime_compiler() -> str:
    """Compile and link the kind of native helper Triton builds at runtime."""

    configured = os.environ.get("CC", "cc")
    compiler = shutil.which(configured)
    if compiler is None:
        raise RuntimeError(f"runtime C compiler is unavailable: CC={configured!r}")
    with tempfile.TemporaryDirectory(prefix="trl-compiler-check-") as tmp:
        directory = Path(tmp)
        source = directory / "probe.c"
        source.write_text("int jspace_compiler_probe(void) { return 0; }\n")
        _run_checked(
            [
                compiler,
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(directory / "probe.so"),
            ],
            what="runtime C compiler shared-object link",
        )
    return compiler


def _cuda_home() -> Path:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        raise RuntimeError("CUDA_HOME is unset; source infra/activate_trl_runtime.sh")
    path = Path(home)
    required = [
        path / "bin" / "nvcc",
        path / "bin" / "ptxas",
        path / "include" / "cuda_runtime.h",
        path / "lib64",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(f"incomplete CUDA toolkit at {path}: missing {missing}")
    return path


def _verify_cuda_toolchain() -> dict[str, str]:
    """Compile a real device translation unit with FlashInfer's flag shape."""

    cuda_home = _cuda_home()
    nvcc = str(cuda_home / "bin" / "nvcc")
    version = _run_checked([nvcc, "--version"], what="nvcc --version")
    if "release 12.9" not in version:
        raise RuntimeError(
            f"expected CUDA 12.9 nvcc matching the runtime headers, found: "
            f"{version.strip()[-200:]}"
        )

    # FlashInfer 0.6.6 uses the CCCL shipped by the CUDA toolkit. Debian's
    # cuda-cccl package exposes cub/, thrust/, and cuda/std/ directly below the
    # canonical CUDA include directory; it does not create include/cccl and the
    # FlashInfer wheel does not bundle a separate CCCL tree.
    cuda_include = cuda_home / "include"
    cccl_headers = [
        cuda_include / "cub" / "cub.cuh",
        cuda_include / "thrust" / "version.h",
        cuda_include / "cuda" / "std" / "array",
    ]
    missing_cccl = [str(path) for path in cccl_headers if not path.is_file()]
    if missing_cccl:
        raise RuntimeError(f"FlashInfer CCCL headers are missing: {missing_cccl}")

    arch = CUDA_ARCH.replace(".", "")
    with tempfile.TemporaryDirectory(prefix="trl-cuda-check-") as tmp:
        directory = Path(tmp)
        source = directory / "probe.cu"
        source.write_text(
            "#include <cuda/std/array>\n"
            "#include <cub/cub.cuh>\n"
            "#include <cuda_runtime.h>\n"
            "__global__ void jspace_probe(float* out) { out[threadIdx.x] = 1.0f; }\n"
            "extern \"C\" int jspace_launch(float* out) {\n"
            "  jspace_probe<<<1, 32>>>(out);\n"
            "  return (int)cudaGetLastError();\n"
            "}\n"
        )
        # The driver stub is deliberately absent from the CUDA wheels, so the
        # build-time link omits -lcuda. The GPU preflight links against the real
        # driver through FLASHINFER_EXTRA_LDFLAGS.
        _run_checked(
            [
                nvcc,
                "-shared",
                "-Xcompiler",
                "-fPIC",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                f"-I{cuda_include}",
                str(source),
                "-o",
                str(directory / "probe.so"),
                f"-L{cuda_home / 'lib64'}",
                "-lcudart",
            ],
            what=f"nvcc sm_{arch} shared-object link",
        )
    return {
        "cuda_home": str(cuda_home),
        "cuda_compiler": nvcc,
        "cuda_arch": CUDA_ARCH,
        "cuda_toolchain": "ok",
        "flashinfer_cccl": "ok",
    }


def _verify_trainer_arguments() -> str:
    """Validate every GRPOConfig key this repository sets actually exists."""

    from trl import GRPOConfig, GRPOTrainer

    if not callable(getattr(GRPOTrainer, "train", None)):
        raise RuntimeError("TRL GRPOTrainer is unavailable")

    reference = TrlExperimentConfig(
        run=TrlRunConfig(name="runtime-check", output_dir="/tmp/runtime-check", gpus=2),
        model=TrlModelConfig(),
        task=TaskConfig(),
        data=TrlDataConfig(train_examples=64, eval_examples_per_split=4),
        rollout=TrlRolloutConfig(),
        optimization=TrlOptimizationConfig(prompts_per_step=2, total_training_steps=2),
    )
    reference.validate()

    known = {field.name for field in dataclasses.fields(GRPOConfig)}
    unknown = sorted(set(grpo_arguments(reference)) - known)
    if unknown:
        raise RuntimeError(f"GRPOConfig does not accept: {unknown}")
    return "ok"


def _verify_vllm_engine_arguments() -> str:
    """Reject preflight kwargs outside the pinned vLLM EngineArgs dataclass."""

    from vllm.engine.arg_utils import EngineArgs

    reference = TrlExperimentConfig(
        run=TrlRunConfig(name="runtime-check", output_dir="/tmp/runtime-check"),
        model=TrlModelConfig(),
        task=TaskConfig(),
        data=TrlDataConfig(train_examples=64, eval_examples_per_split=4),
        rollout=TrlRolloutConfig(),
        optimization=TrlOptimizationConfig(prompts_per_step=2, total_training_steps=2),
    )
    reference.validate()
    known = {field.name for field in dataclasses.fields(EngineArgs)}
    unknown = sorted(set(engine_arguments(reference)) - known)
    if unknown:
        raise RuntimeError(f"vLLM EngineArgs does not accept: {unknown}")
    return "ok"


def _verify_static_interfaces() -> None:
    """Verify integration surfaces without importing GPU-only extensions."""

    from transformers import Qwen3_5ForCausalLM
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )

    if MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get("qwen3_5") != "Qwen3_5ForCausalLM":
        raise RuntimeError("Transformers does not map qwen3_5 to a causal LM")
    if not getattr(Qwen3_5ForCausalLM, "_supports_sdpa", False):
        raise RuntimeError("Transformers Qwen3.5 does not support SDPA")

    # The official Qwen3.5 checkpoints declare the multimodal conditional-
    # generation architecture. vLLM 0.19.1 supports that exact registry entry;
    # it does not need to expose Transformers' separate causal-LM training
    # wrapper for the colocated rollout engine.
    _require_source_symbols(
        "vllm",
        "vllm/model_executor/models/registry.py",
        ('"Qwen3_5ForConditionalGeneration"',),
    )

    score = compute_score(
        data_source="graph_walk/runtime_check",
        solution_str="Final node: cedar",
        ground_truth="cedar",
        extra_info={"nodes": ["cedar", "plum"]},
    )
    if score["score"] != 1.0 or score["format_valid"] != 1.0:
        raise RuntimeError(f"custom reward smoke check failed: {score}")


def _verify_gpu_interfaces() -> dict[str, object]:
    """Exercise the driver-linked paths that only exist on a GPU node."""

    driver = ctypes.CDLL("libcuda.so.1")
    driver_version = ctypes.c_int()
    result = driver.cuDriverGetVersion(ctypes.byref(driver_version))
    if result != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with CUDA status {result}")
    # NVIDIA drivers are backward compatible with older CUDA runtimes. Reject
    # only a runtime from a newer major family than the driver supports (the
    # Crusoe failure was driver API 12.7 versus a CUDA 13 runtime). A local
    # CUDA-13-capable driver running the locked CUDA 12.9 runtime is valid.
    if not driver_supports_runtime(driver_version.value, str(torch.version.cuda)):
        raise RuntimeError(
            f"CUDA driver is too old: driver API {driver_version.value}, "
            f"PyTorch runtime {torch.version.cuda}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires a visible CUDA device")
    capability = torch.cuda.get_device_capability(0)
    if capability not in SUPPORTED_AMPERE_CAPABILITIES:
        raise RuntimeError(
            f"unsupported GPU compute capability {capability}; expected A100 sm80 "
            "or A10G sm86"
        )

    # A freshly exec'd process must resolve the wheel-provided CUDA libraries
    # on its own; the parent can mask a broken LD_LIBRARY_PATH once torch has
    # already loaded them.
    child = "\n".join(
        (
            "import ctypes",
            "ctypes.CDLL('libcudart.so.12')",
            "import torch",
            "assert torch.cuda.is_available()",
            "probe = torch.ones(32, device='cuda')",
            "assert float(probe.sum().cpu()) == 32.0",
            "torch.cuda.synchronize()",
            "from vllm import LLM, SamplingParams",
            "assert LLM is not None and SamplingParams is not None",
        )
    )
    _run_checked([sys.executable, "-c", child], what="clean CUDA/vLLM subprocess")

    # Link against the real driver exactly as FlashInfer's JIT would.
    extra = os.environ.get("FLASHINFER_EXTRA_LDFLAGS", "")
    if not extra:
        raise RuntimeError(
            "FLASHINFER_EXTRA_LDFLAGS is unset; the activation script did not "
            "find libcuda.so.1"
        )
    cuda_home = _cuda_home()
    arch = CUDA_ARCH.replace(".", "")
    with tempfile.TemporaryDirectory(prefix="trl-driver-link-") as tmp:
        directory = Path(tmp)
        source = directory / "driver.cu"
        source.write_text(
            "#include <cuda.h>\n"
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n"
            "__global__ void jspace_kernel(int* out) { *out = 17; }\n"
            "int main(void) {\n"
            "  if (cuInit(0) != CUDA_SUCCESS) return 2;\n"
            "  int* ptr = nullptr;\n"
            "  if (cudaMalloc(&ptr, sizeof(int)) != cudaSuccess) return 3;\n"
            "  jspace_kernel<<<1, 1>>>(ptr);\n"
            "  int value = 0;\n"
            "  if (cudaMemcpy(&value, ptr, sizeof(int), cudaMemcpyDeviceToHost) "
            "!= cudaSuccess) return 4;\n"
            "  cudaFree(ptr);\n"
            "  if (value != 17) return 5;\n"
            "  std::printf(\"sm80_probe=%d\\n\", value);\n"
            "  return 0;\n"
            "}\n"
        )
        executable = directory / "driver-probe"
        _run_checked(
            [
                str(cuda_home / "bin" / "nvcc"),
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                f"-I{cuda_home / 'include'}",
                str(source),
                "-o",
                str(executable),
                f"-L{cuda_home / 'lib64'}",
                *extra.split(),
                "-lcudart",
                "-lcuda",
            ],
            what="nvcc driver executable link",
        )
        _run_checked([str(executable)], what="sm80 CUDA compile-and-launch probe")
    return {
        "cuda_driver_api": driver_version.value,
        "cuda_device": torch.cuda.get_device_name(0),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "clean_cuda_subprocess": "ok",
        "driver_link": "ok",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="also exercise the driver-linked paths on a visible GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    versions = _verify_locked_packages()
    _verify_static_interfaces()
    result: dict[str, object] = {
        **versions,
        **_verify_cuda_toolchain(),
        "runtime_compiler": _verify_runtime_compiler(),
        "trainer_arguments": _verify_trainer_arguments(),
        "vllm_engine_arguments": _verify_vllm_engine_arguments(),
        "torch_cuda": torch.version.cuda,
        "static_interfaces": "ok",
    }
    if args.require_gpu:
        result.update(_verify_gpu_interfaces())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Boot the configured SGLang rollout path and complete one real generation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch

from jspace_plasticity.verl_config import VerlExperimentConfig, load_verl_config

CACHE_ENVIRONMENT = (
    "XDG_CACHE_HOME",
    "FLASHINFER_WORKSPACE_BASE",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(config: VerlExperimentConfig, port: int) -> list[str]:
    """Render the standalone server with the production model and kernel path."""

    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model.name_or_path,
        "--tokenizer-path",
        config.model.name_or_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--attention-backend",
        config.rollout.attention_backend,
        "--mem-fraction-static",
        str(config.rollout.gpu_memory_utilization),
        "--max-running-requests",
        "2",
        "--cuda-graph-max-bs",
        "2",
        "--dtype",
        config.model.dtype,
        "--tp-size",
        str(config.rollout.tensor_parallel_size),
    ]
    if config.model.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _verify_runtime_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SGLang rollout preflight requires a visible CUDA device")
    capability_tuple = torch.cuda.get_device_capability(0)
    capability = f"{capability_tuple[0]}.{capability_tuple[1]}"
    for variable in ("FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST"):
        if os.environ.get(variable) != capability:
            raise RuntimeError(
                f"{variable} must match the visible GPU ({capability}), "
                f"found {os.environ.get(variable)!r}"
            )

    cuda_home = Path(os.environ.get("CUDA_HOME", ""))
    if cuda_home.resolve() != Path("/usr/local/cuda-12.8"):
        raise RuntimeError(f"unexpected CUDA_HOME for rollout preflight: {cuda_home}")

    cache_paths: dict[str, str] = {}
    for variable in CACHE_ENVIRONMENT:
        raw_path = os.environ.get(variable)
        if not raw_path:
            raise RuntimeError(
                f"required persistent cache variable is unset: {variable}"
            )
        path = Path(raw_path)
        if not path.is_absolute() or not path.is_dir():
            raise RuntimeError(
                f"cache path is not an absolute directory: {variable}={path}"
            )
        with tempfile.TemporaryFile(dir=path):
            pass
        cache_paths[variable] = str(path)

    return {
        "cache_paths": cache_paths,
        "compute_capability": capability,
        "cuda_device": torch.cuda.get_device_name(0),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()
        if response.status != 200:
            raise RuntimeError(f"SGLang returned HTTP {response.status} for {url}")
    return None if not content else json.loads(content)


def _wait_until_ready(
    process: subprocess.Popen[bytes], base_url: str, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not accept a connection"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"SGLang exited before becoming healthy (exit={returncode})"
            )
        remaining = max(1.0, deadline - time.monotonic())
        try:
            _request_json(
                "GET",
                f"{base_url}/health",
                timeout=min(5.0, remaining),
            )
            return
        except (OSError, TimeoutError, urllib.error.HTTPError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
    raise TimeoutError(
        f"SGLang did not become healthy within {timeout}s: {last_error}"
    )


def _complete_generation(
    base_url: str, config: VerlExperimentConfig
) -> dict[str, Any]:
    result = _request_json(
        "POST",
        f"{base_url}/generate",
        payload={
            "text": "Continue with a short answer: one, two, three,",
            "sampling_params": {
                "max_new_tokens": 8,
                "min_new_tokens": 2,
                "temperature": config.rollout.temperature,
                "top_p": config.rollout.top_p,
            },
        },
        timeout=300,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected SGLang generation payload: {result!r}")
    text = result.get("text")
    completion_tokens = result.get("meta_info", {}).get("completion_tokens", 0)
    if not isinstance(text, str) or not text.strip() or completion_tokens < 1:
        raise RuntimeError(f"SGLang returned an empty generation: {result!r}")
    return result


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--port", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.timeout < 60:
        raise ValueError("--timeout must be at least 60 seconds")
    config = load_verl_config(args.config)
    runtime = _verify_runtime_environment()
    port = args.port or _free_port()
    command = server_command(config, port)
    print(
        json.dumps({"server_command": command, **runtime}, sort_keys=True),
        flush=True,
    )

    environment = os.environ.copy()
    if config.rollout.tensor_parallel_size == 1:
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_until_ready(process, base_url, args.timeout)
        _request_json(
            "GET",
            f"{base_url}/health_generate",
            timeout=300,
        )
        generation = _complete_generation(base_url, config)
        print(
            json.dumps(
                {
                    "completion_tokens": generation["meta_info"][
                        "completion_tokens"
                    ],
                    "rollout_preflight": "ok",
                    "sampling": {
                        "temperature": config.rollout.temperature,
                        "top_p": config.rollout.top_p,
                    },
                    "text_preview": generation["text"][:120],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        _terminate_process_group(process)


if __name__ == "__main__":
    main()

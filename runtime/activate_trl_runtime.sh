#!/usr/bin/env bash

# Canonical activation contract for the image-baked TRL/vLLM runtime. This file
# must be sourced because Docker and Kubernetes invoke a login shell
# (`bash -lc`), which reconstructs PATH and would otherwise hide the CUDA
# compiler. The VERL-era A100 attempt 4 failed for exactly that reason, so the
# contract is centralized here and exercised during the image build.
#
# CUDA 12.9's compiler closure is installed under /usr/local/cuda-12.9. PyTorch
# and its math libraries come from the official cu129 wheel index; the host only
# supplies the kernel driver through the NVIDIA container runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "activate_trl_runtime.sh must be sourced" >&2
  exit 2
fi

export VIRTUAL_ENV="${TRL_RUNTIME_VIRTUAL_ENV:-/opt/experiment-venv}"

_trl_site=$(echo "${VIRTUAL_ENV}"/lib/python3.*/site-packages)
if [[ ! -d "${_trl_site}" ]]; then
  echo "TRL runtime site-packages is missing: ${_trl_site}" >&2
  return 1
fi
export TRL_RUNTIME_SITE_PACKAGES="${_trl_site}"

export CUDA_HOME="${TRL_RUNTIME_CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_HOME}"
export CC=/usr/bin/cc
export CXX=/usr/bin/c++

# Do not inherit PATH from a login shell. Its content differs between Docker,
# Kubernetes, and an interactive host.
export PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Resolve the official cu129 component wheels by glob so a packaging change
# cannot silently strip a library path from a freshly exec'd process.
_trl_libs="${_trl_site}/torch/lib:${CUDA_HOME}/lib64"
for _trl_dir in "${_trl_site}"/nvidia/*/lib "${_trl_site}"/nvidia/*/lib64; do
  [[ -d "${_trl_dir}" ]] && _trl_libs="${_trl_libs}:${_trl_dir}"
done
export LD_LIBRARY_PATH="${_trl_libs}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _trl_dir _trl_libs

# FlashInfer's generated ninja file links `-lcuda` against
# `$CUDA_HOME/lib64/stubs`. The wheels intentionally ship no driver stub, so
# point the linker at the real driver the container runtime injects. This is a
# link-time search path only; it never precedes the driver on the loader path.
_trl_driver=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ {print $NF; exit}')
if [[ -n "${_trl_driver}" && -e "${_trl_driver}" ]]; then
  export TRL_RUNTIME_DRIVER_DIR="${TRL_RUNTIME_DRIVER_DIR:-/run/jspace-cuda-link}"
  mkdir -p "${TRL_RUNTIME_DRIVER_DIR}"
  ln -sf "${_trl_driver}" "${TRL_RUNTIME_DRIVER_DIR}/libcuda.so"
  export FLASHINFER_EXTRA_LDFLAGS="-L${TRL_RUNTIME_DRIVER_DIR}"
fi
unset _trl_driver

# Restrict every runtime compiler to the deployed architecture. Without this,
# FlashInfer fans out over its full default architecture list.
export FLASHINFER_CUDA_ARCH_LIST="${TRL_RUNTIME_CUDA_ARCH:-8.0}"
export TORCH_CUDA_ARCH_LIST="${TRL_RUNTIME_CUDA_ARCH:-8.0}"

if [[ -n "${TRL_RUNTIME_CACHE_ROOT:-}" ]]; then
  export XDG_CACHE_HOME="${TRL_RUNTIME_CACHE_ROOT}/xdg"
  export TRITON_CACHE_DIR="${TRL_RUNTIME_CACHE_ROOT}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${TRL_RUNTIME_CACHE_ROOT}/inductor"
  export TORCH_EXTENSIONS_DIR="${TRL_RUNTIME_CACHE_ROOT}/torch-extensions"
  export VLLM_CACHE_ROOT="${TRL_RUNTIME_CACHE_ROOT}/vllm"
  export FLASHINFER_WORKSPACE_BASE="${TRL_RUNTIME_CACHE_ROOT}/flashinfer"
  mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${VLLM_CACHE_ROOT}" "${FLASHINFER_WORKSPACE_BASE}"
fi

for runtime_tool in \
  "${VIRTUAL_ENV}/bin/python" \
  "${CUDA_HOME}/bin/nvcc" \
  "${CUDA_HOME}/bin/ptxas" \
  /usr/bin/cc \
  /usr/bin/c++; do
  if [[ ! -x "${runtime_tool}" ]]; then
    echo "TRL runtime executable is missing: ${runtime_tool}" >&2
    return 1
  fi
done
unset runtime_tool
unset _trl_site

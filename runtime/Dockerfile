# Public-source rebuild recipe; not rebuilt in this packaging pass.
# Binary-only CUDA 12.9 TRL/vLLM runtime for Crusoe's R565 A100 fleet.
#
# vLLM 0.19.1 is the first practical release in TRL 1.9.2's supported range
# that combines native Qwen3.5 support with Transformers 5 compatibility. Its
# standard x86_64 wheel is built for CUDA 12.9, so no vLLM source is downloaded
# or compiled in this image. The CUDA compiler packages below are retained only
# because FlashInfer and Triton JIT model-specific kernels at first execution.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/experiment-venv \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CC=/usr/bin/cc \
    CXX=/usr/bin/c++ \
    TORCH_CUDA_ARCH_LIST=8.0 \
    FLASHINFER_CUDA_ARCH_LIST=8.0

# build-essential is a runtime dependency: Triton and FlashInfer compile
# model-specific kernels on first use. The narrow CUDA closure mirrors vLLM's
# official runtime image without installing the full toolkit meta-package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        curl \
        libnuma1 \
        libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

ADD --checksum=sha256:e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d \
    https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    /tmp/cuda-keyring.deb
RUN dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cccl-12-9=12.9.27-1 \
        cuda-crt-12-9=12.9.86-1 \
        cuda-cudart-12-9=12.9.79-1 \
        cuda-cudart-dev-12-9=12.9.79-1 \
        cuda-cuobjdump-12-9=12.9.82-1 \
        cuda-driver-dev-12-9=12.9.79-1 \
        cuda-nvcc-12-9=12.9.86-1 \
        cuda-nvrtc-12-9=12.9.86-1 \
        cuda-nvvm-12-9=12.9.86-1 \
        cuda-toolkit-12-9-config-common=12.9.79-1 \
        cuda-toolkit-12-config-common=12.9.79-1 \
        cuda-toolkit-config-common=13.3.29-1 \
        libcublas-12-9=12.9.2.10-1 \
        libcurand-dev-12-9=10.3.10.19-1 \
    && test "$(readlink -f /usr/local/cuda)" = /usr/local/cuda-12.9 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.10.6

WORKDIR /opt/trl-runtime
COPY infra/trl-runtime/pyproject.toml infra/trl-runtime/uv.lock ./

# This installs only released binary distributions from the committed uv lock,
# including vLLM's official CUDA 12.9 wheel. There is no source-build stage.
RUN uv sync --frozen --no-dev --no-install-project \
    && ! uv pip list --python /opt/experiment-venv/bin/python --format=freeze \
        | grep -Eiq '(^|[-_])cu13([^0-9]|$)'

WORKDIR /opt/experiment
COPY pyproject.toml uv.lock ./
COPY configs ./configs
COPY data ./data
COPY src ./src
COPY infra/activate_trl_runtime.sh ./infra/activate_trl_runtime.sh
COPY infra/verify_trl_runtime.py ./infra/verify_trl_runtime.py
COPY infra/preflight_vllm_engine.py ./infra/preflight_vllm_engine.py
COPY infra/preflight_lesion_equivalence.py ./infra/preflight_lesion_equivalence.py

ENV PATH="/opt/experiment-venv/bin:/usr/local/cuda/bin:${PATH}" \
    PYTHONPATH=/opt/experiment/src

# Exercise the same login-shell activation used by Docker and Kubernetes.
RUN /bin/bash -lc 'source infra/activate_trl_runtime.sh \
    && python infra/verify_trl_runtime.py'

ENTRYPOINT ["/bin/bash", "-lc"]

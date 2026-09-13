# GPU runtime for the J-space lesion recovery experiment.
#
# The reported experiment is plain supervised fine-tuning: a hand-rolled
# training loop over torch.optim.AdamW with the lesion applied by forward
# hooks. There is no RL, no TRL, no vLLM and no inference server, so this
# image installs only Torch, Transformers, jlens and their dependencies from
# the committed uv lock. No CUDA toolchain or source build is needed.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/experiment-venv

# git is required to resolve the pinned jlens source dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.10.6

WORKDIR /opt/experiment
COPY pyproject.toml uv.lock ./
COPY src ./src

# Released binary wheels only, straight from the lock.
RUN uv sync --frozen --no-dev

COPY data ./data
COPY scripts ./scripts

ENV PATH="/opt/experiment-venv/bin:${PATH}"

# Fail the build if the environment cannot import what the experiment needs.
RUN python -c "import torch, transformers, jlens, numpy, yaml; \
print('torch', torch.__version__); \
print('transformers', transformers.__version__)"

ENTRYPOINT ["/bin/bash", "-lc"]

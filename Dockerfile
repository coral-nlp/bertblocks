FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel@sha256:48af19ebb88034e0325decc2b6142e5b9bfc276eaa5197ab6d94582f1d78ea4c
# This base image has pre-installed:
# - python 3.12.3
# - CUDA 13.0 + dev toolkit
# - torch 2.10.0
WORKDIR /app
# Set env vars
ENV UV_COMPILE_BYTECODE=1 \
    UV_TORCH_BACKEND=auto \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
# Install dependencies; mount in cache/uv files to avoid extra copy layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --all-extras --no-dev
# Activate uv venv
ENV PATH="/app/.venv/bin:$PATH"
# Copy sources
ADD . .
# Entrypoint
CMD ["python3", "main.py"]

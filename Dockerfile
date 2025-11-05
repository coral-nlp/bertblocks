# NVIDIA NGC PyTorch container with pre-configured CUDA/cuDNN for Transformer Engine
FROM nvcr.io/nvidia/pytorch:25.08-py3
# Install uv 0.8.11
COPY --from=ghcr.io/astral-sh/uv@sha256:8101ad825250a114e7bef89eefaa73c31e34e10ffbe5aff01562740bac97553c /uv /bin/uv
# Change the working directory to the `app` directory
WORKDIR /app
# Set UV settings
ENV UV_COMPILE_BYTECODE=1 UV_TORCH_BACKEND=auto UV_LINK_MODE=copy
# Transformer Engine build settings
ENV NVTE_FRAMEWORK=pytorch MAX_JOBS=1
# Install dependencies; mount in cache/uv files to avoid extra copy layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --group flash-attn --group optimizers --group transformer-engine
# Activate uv venv
ENV PATH="/app/.venv/bin:$PATH"
# Copy sources
ADD . .
# Entrypoint
CMD ["python3", "main.py"]

# pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
FROM pytorch/pytorch@sha256:639b8229ccfd8a3aa803cf49c33d6d6fe406750d79aaf723fe8c0eb1060d8cff
# Install uv 0.8.2
COPY --from=ghcr.io/astral-sh/uv@sha256:a7999d42cba0e5af47ef3c06ac310229c7f29c5314e35902f8353e8e170eeed1 /uv /bin/uv
# Change the working directory to the `app` directory
WORKDIR /app
# Flash attention needs is run with no-build-isolation, so this needs to be installed system-wide
RUN conda install -y setuptools wheel ninja packaging && \
    conda clean -ya
# Set UV settings for faster startup times
ENV UV_COMPILE_BYTECODE=1
# Install dependencies; mount in cache/uv files to avoid extra copy layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project
# Activate uv venv
ENV PATH="/app/.venv/bin:$PATH"
# Copy sources
ADD . .
# Entrypoint
CMD ["python3", "main.py"]

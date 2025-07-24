# pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
FROM pytorch/pytorch@sha256:639b8229ccfd8a3aa803cf49c33d6d6fe406750d79aaf723fe8c0eb1060d8cff
# Install uv 0.8.2
COPY --from=ghcr.io/astral-sh/uv@sha256:a7999d42cba0e5af47ef3c06ac310229c7f29c5314e35902f8353e8e170eeed1 /uv /bin/uv
# Change the working directory to the `app` directory
WORKDIR /app
# Set UV settings for faster startup times, and install to system python
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/usr/local
# Install dependencies; mount in cache/uv files to avoid extra copy layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project \
# Copy sources
ADD . .
# Entrypoint
CMD ["python3", "main.py"]

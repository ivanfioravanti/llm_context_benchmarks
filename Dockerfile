# Web UI + CLI benchmarks against remote/OpenAI-compatible endpoints.
# MLX-local engines require Apple Silicon and are disabled inside the
# container automatically (webui_common.is_apple_silicon()).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    # tells the web UI to show container-specific hints (host.docker.internal)
    CONTEXT_BENCH_CONTAINER=1

WORKDIR /app

# Dependency layer — deliberately NOT pyproject.toml: the container only
# talks to inference servers over HTTP, so the heavy local-inference stack
# (mlx, torch, transformers, datasets) stays out of the image.
COPY requirements-docker.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements-docker.txt

# Application code (flat modules + webui_static + sample context files)
COPY . .

EXPOSE 8321

# Run the module directly: /app comes first on sys.path, so a bind mount of
# the project overrides the baked-in copy without reinstalling anything.
CMD ["python", "webui.py", "--host", "0.0.0.0", "--port", "8321", "--no-open"]

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs/docker.yaml ./configs/docker.yaml

RUN mkdir -p /input /output /logs
RUN uv sync --frozen --no-dev

ENTRYPOINT ["/bin/sh", "-c", "mkdir -p /output /logs && .venv/bin/dabench run-benchmark --config configs/docker.yaml >/logs/runtime.log 2>&1"]

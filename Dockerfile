FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    DABENCH_SUBMISSION_CONFIG=configs/docker.yaml \
    DABENCH_LOG_ROOT=/logs \
    DABENCH_RETRY_SLEEP_SECONDS=60

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs/docker.yaml ./configs/docker.yaml

RUN mkdir -p /input /output /logs
RUN uv sync --frozen --no-dev

ENTRYPOINT ["/bin/bash", "-lc", "set -o pipefail; mkdir -p /output \"${DABENCH_LOG_ROOT}\"; i=1; status=0; while [ \"$i\" -le 2 ]; do .venv/bin/dabench run-benchmark --skip-completed --config \"${DABENCH_SUBMISSION_CONFIG}\" 2>&1 | tee -a \"${DABENCH_LOG_ROOT}/runtime.log\"; status=$?; echo \"[entrypoint] run ${i}/2 exited status=${status}\" | tee -a \"${DABENCH_LOG_ROOT}/runtime.log\"; i=$((i+1)); if [ \"$i\" -le 2 ]; then echo \"[entrypoint] sleeping ${DABENCH_RETRY_SLEEP_SECONDS}s before next run...\" | tee -a \"${DABENCH_LOG_ROOT}/runtime.log\"; sleep \"${DABENCH_RETRY_SLEEP_SECONDS}\"; fi; done; exit \"$status\""]

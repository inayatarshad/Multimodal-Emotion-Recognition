# Multi-stage: the build stage carries uv and the toolchain, the runtime stage does not.
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so a source edit does not invalidate the (large, torch-heavy) layer.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra serve --no-install-project --no-dev

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra serve --no-dev


FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 wfb
WORKDIR /app

COPY --from=builder --chown=wfb:wfb /app/.venv /app/.venv
COPY --chown=wfb:wfb src/ ./src/
COPY --chown=wfb:wfb configs/ ./configs/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WFB_DATASET=mosi \
    WFB_CHECKPOINTS=/app/outputs \
    WFB_RESULTS=/app/experiments/results

USER wfb
EXPOSE 8000

# The registry loads models at startup, so readiness lags liveness by a few seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "wfb.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

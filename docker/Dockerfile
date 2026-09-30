# syntax=docker/dockerfile:1
#
# VeltronLM container.
#
# CPU image by default; the GPU variants are built from the same source with a different
# base image tag. DirectML is used for AMD/Intel GPUs on Windows and Linux, so a GPU image
# is simply the CPU image plus torch-directml.
#
#   docker build -t veltronlm:cpu .
#   docker build -t veltronlm:directml --build-arg INSTALL_DIRECTML=1 .
FROM python:3.10-slim AS base

ARG INSTALL_DIRECTML=0
ARG INSTALL_PDF=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/app/.cache \
    VELTRON_DEVICE=auto \
    VELTRON_LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install torch first from the CPU index so the CUDA wheels are not pulled in.
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
RUN pip install \
        "tokenizers>=0.15" "safetensors>=0.4" "numpy>=1.24" \
        "pydantic>=2.0" "fastapi>=0.110" "uvicorn>=0.27" "python-multipart>=0.0.9"

RUN if [ "$INSTALL_DIRECTML" = "1" ]; then pip install torch-directml==0.2.5.dev240914 --no-deps; fi \
    && if [ "$INSTALL_PDF" = "1" ]; then pip install pypdf; fi

COPY veltron/ ./veltron/
COPY knowledge_base/ ./knowledge_base/
COPY configs/ ./configs/

# Non-root by default. The process needs no write access outside /app/.cache.
RUN useradd --create-home --uid 10001 veltron \
    && mkdir -p /app/.cache /app/checkpoints /app/reports \
    && chown -R veltron:veltron /app
USER veltron

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/v1/health || exit 1

# Bind to 0.0.0.0 inside the container; publish only to loopback on the host unless you
# have put authentication in front of it (VeltronLM ships none).
CMD ["python", "-m", "veltron.serve", "--host", "0.0.0.0", "--port", "8000"]
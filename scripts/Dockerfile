# syntax=docker/dockerfile:1

ARG VLLM_IMAGE=docker.io/vllm/vllm-openai:v0.28.0

FROM python:3.12-slim AS quantizer

ARG LLM_COMPRESSOR_VERSION=0.13.0
WORKDIR /workspace

RUN python -m pip install --no-cache-dir \
    "llmcompressor==${LLM_COMPRESSOR_VERSION}"

COPY scripts/quantize.py /workspace/quantize.py
ENTRYPOINT ["python", "/workspace/quantize.py"]

FROM ${VLLM_IMAGE} AS serving

LABEL org.opencontainers.image.source="https://github.com/red-hat-ai-dev/vLLM-zero-to-hero-pt2" \
      org.opencontainers.image.description="A ready-to-run vLLM optimization walkthrough"

EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen3-4B", "--host", "0.0.0.0", "--port", "8000", "--served-model-name", "qwen3-4b", "--max-model-len", "8192"]

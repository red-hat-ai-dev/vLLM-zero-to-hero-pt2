#!/bin/sh
set -eu

repo_root="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
image="vllm-zero-to-hero-pt2-quantizer:local"
llm_compressor_version="0.13.0"
hf_home="${HF_HOME:-${HOME:-}/.cache/huggingface}"
venv="$repo_root/.venv-pt2-quantizer"
engine=""

error() {
  echo "Error: $*" >&2
}

select_engine() {
  if [ -n "${ENGINE:-}" ]; then
    if ! command -v "$ENGINE" >/dev/null 2>&1; then
      error "ENGINE is set to '$ENGINE', but that command was not found."
      exit 1
    fi
    if ! "$ENGINE" info >/dev/null 2>&1; then
      error "$ENGINE is installed but is not running."
      exit 1
    fi
    engine="$ENGINE"
  elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    engine="podman"
  elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    engine="docker"
  fi
}

if [ -z "$hf_home" ] || [ "$hf_home" = "/" ]; then
  error "HOME or HF_HOME does not point to a safe cache directory."
  exit 1
fi

mkdir -p "$repo_root/artifacts" "$hf_home"
cd "$repo_root"
select_engine

if [ -n "$engine" ]; then
  echo "Using $engine for the quantizer dependencies."
  echo "Building or reusing the local quantizer image..."
  "$engine" build \
    --target quantizer \
    --build-arg "LLM_COMPRESSOR_VERSION=$llm_compressor_version" \
    --tag "$image" \
    "$repo_root"

  volume_suffix=""
  if [ "$engine" = "podman" ] && [ "$(uname -s)" = "Linux" ]; then
    volume_suffix=":Z"
  fi

  echo "Starting the CPU quantization pipeline."
  "$engine" run --rm -it \
    -v "$hf_home:/root/.cache/huggingface$volume_suffix" \
    -v "$repo_root/artifacts:/workspace/artifacts$volume_suffix" \
    "$image"
  exit 0
fi

echo "No running Docker or Podman was found."
echo "Using an isolated local Python environment instead."
if [ ! -x "$venv/bin/python" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    error "python3 is required when Docker and Podman are unavailable."
    exit 1
  fi
  python3 -m venv "$venv"
  "$venv/bin/python" -m pip install --upgrade pip
  "$venv/bin/python" -m pip install "llmcompressor==$llm_compressor_version"
fi

"$venv/bin/python" quantize.py

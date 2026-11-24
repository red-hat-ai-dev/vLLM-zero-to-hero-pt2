#!/bin/sh
set -eu

repo_root="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
image="vllm-zero-to-hero-pt2-quantizer:local"
llm_compressor_version="0.13.0"
hf_home="${HF_HOME:-${HOME:-}/.cache/huggingface}"
engine=""
output_dir="${QUANTIZE_OUTPUT:-$repo_root/artifacts/Qwen3-4B-FP8-Dynamic}"
device="${QUANTIZE_DEVICE:-cpu}"
case "$device" in cpu|cuda) ;; *) echo "Error: QUANTIZE_DEVICE must be cpu or cuda." >&2; exit 1 ;; esac
case "${QUANTIZER_BACKEND:-auto}" in auto|local) ;; *) echo "Error: QUANTIZER_BACKEND must be auto or local." >&2; exit 1 ;; esac
if [ "$#" -ne 0 ]; then
  echo "Use QUANTIZE_OUTPUT, QUANTIZE_DEVICE, and QUANTIZER_BACKEND to configure this wrapper." >&2
  exit 1
fi

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

case "$output_dir" in /*) ;; *) output_dir="$repo_root/$output_dir" ;; esac
mkdir -p "$(dirname "$output_dir")"
output_parent="$(CDPATH='' cd "$(dirname "$output_dir")" && pwd)"
output_name="$(basename "$output_dir")"
cd "$repo_root"
if [ "${QUANTIZER_BACKEND:-auto}" != local ]; then select_engine; fi

if [ -n "$engine" ]; then
  echo "Using $engine for the quantizer dependencies."
  echo "Building or reusing the local quantizer image..."
  "$engine" build \
    --file "$repo_root/scripts/Dockerfile" \
    --target quantizer \
    --build-arg "LLM_COMPRESSOR_VERSION=$llm_compressor_version" \
    --tag "$image" \
    "$repo_root"

  volume_suffix=""
  if [ "$(uname -s)" = "Linux" ]; then
    # Share the serving volume so a rootful engine never writes to the learner's cache.
    # shellcheck source=scripts/engine.sh
    . "$repo_root/scripts/engine.sh"
    ensure_model_volume
    cache_source="$volume"
    if [ "${engine##*/}" = "podman" ]; then volume_suffix=":Z"; fi
  else
    mkdir -p "$hf_home"
    cache_source="$hf_home"
  fi

  set --
  if [ "$device" = cuda ]; then
    if [ "${engine##*/}" = podman ]; then
      set -- --device nvidia.com/gpu=all --security-opt=label=disable
    else
      set -- --gpus all
    fi
  fi
  echo "Starting the $device quantization pipeline."
  # The file's owner inside the bind mount accounts for rootless ID mapping.
  owner_reference="$(mktemp "$output_parent/.pt2-quantizer-owner.XXXXXXXXXX")"
  trap 'rm -f -- "$owner_reference"' EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$engine" run --rm "$@" \
    -v "$cache_source:/root/.cache/huggingface" \
    -v "$output_parent:/workspace/artifacts$volume_suffix" \
    "$image" --output-dir "/workspace/artifacts/$output_name" --device "$device" \
    --owner-reference "/workspace/artifacts/${owner_reference##*/}"
  exit 0
fi

echo "Using an isolated local Python environment for quantization."
mkdir -p "$hf_home"
python_command=""
if [ -n "${PYTHON:-}" ]; then
  set -- "$PYTHON"
else
  set -- python3.13 python3.12 python3.11 python3.10 python3
fi
for candidate in "$@"; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))' 2>/dev/null; then
    python_command="$candidate"
    break
  fi
done

if [ -z "$python_command" ]; then
  error "Python 3.10 through 3.13 is required when Docker and Podman are unavailable."
  error "Install a supported Python version or start Docker/Podman, then rerun this command."
  exit 1
fi

python_version="$("$python_command" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
venv="$repo_root/.venv-pt2-quantizer-$python_version"

if [ ! -x "$venv/bin/python" ]; then
  "$python_command" -m venv "$venv"
fi

# A failed installation leaves the venv behind; retry until the pinned package imports.
if ! "$venv/bin/python" -c 'import llmcompressor, importlib.metadata as m; assert m.version("llmcompressor") == "0.13.0"' >/dev/null 2>&1; then
  "$venv/bin/python" -m pip install --upgrade pip
  "$venv/bin/python" -m pip install "llmcompressor==$llm_compressor_version"
fi

"$venv/bin/python" "$repo_root/scripts/quantize.py" --output-dir "$output_dir" --device "$device"

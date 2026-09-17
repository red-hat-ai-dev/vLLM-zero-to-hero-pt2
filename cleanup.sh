#!/bin/sh
set -eu

name="vllm-zero-to-hero-pt2"
volume="vllm-zero-to-hero-pt2-models"
state_dir="${TMPDIR:-/tmp}/vllm-zero-to-hero-pt2"
user_home="${HOME:-}"
hf_home="${HF_HOME:-$user_home/.cache/huggingface}"
metal_venv="$user_home/.venv-vllm-metal"
metal_model_cache="$hf_home/hub/models--mlx-community--Qwen3-4B-4bit"
metal_model_locks="$hf_home/hub/.locks/models--mlx-community--Qwen3-4B-4bit"
assume_yes="false"
engine=""

usage() {
  echo "Usage: ./cleanup.sh [--yes]"
  echo "Stop vLLM and remove files managed by this tutorial."
  echo "  --yes  Skip the confirmation prompt."
}

if [ -z "$user_home" ] || [ "$user_home" = "/" ] || [ -z "$hf_home" ] || [ "$hf_home" = "/" ]; then
  echo "Error: HOME or HF_HOME is not a safe user directory. Nothing was removed." >&2
  exit 1
fi

case "${1:-}" in
  "") ;;
  --yes) assume_yes="true" ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Error: unknown option: $1" >&2; usage >&2; exit 1 ;;
esac
if [ "$#" -gt 1 ]; then echo "Error: too many options." >&2; usage >&2; exit 1; fi

select_engine() {
  if [ -n "${ENGINE:-}" ]; then
    if ! command -v "$ENGINE" >/dev/null 2>&1; then
      echo "Error: ENGINE is set to '$ENGINE', but that command was not found." >&2
      exit 1
    fi
    if ! "$ENGINE" info >/dev/null 2>&1; then
      echo "Error: $ENGINE is installed but is not running." >&2
      exit 1
    fi
    engine="$ENGINE"
  elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    engine="podman"
  elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    engine="docker"
  fi
}

describe() {
  label="$1"
  path="$2"
  if [ -e "$path" ] || [ -L "$path" ]; then echo "  - $label: $path"; else echo "  - $label: already removed"; fi
}

confirm() {
  if [ "$assume_yes" = "true" ]; then return; fi
  printf "Continue with complete cleanup? [y/N] "
  answer=""
  read -r answer || true
  case "$answer" in y|Y|yes|Yes|YES) ;; *) echo "Cleanup cancelled. Nothing was removed."; exit 0 ;; esac
}

remove_exact() {
  label="$1"
  path="$2"
  expected="$3"
  if [ -z "$path" ] || [ "$path" = "/" ] || [ "$path" = "$user_home" ] || [ "$path" != "$expected" ]; then
    echo "Error: refusing to remove unexpected path for $label: $path" >&2
    exit 1
  fi
  if [ ! -e "$path" ] && [ ! -L "$path" ]; then return; fi
  echo "Removing $label..."
  rm -rf "$path"
}

if [ "$(uname -s)" = "Darwin" ]; then
  echo "This will stop vLLM and permanently remove:"
  describe "Metal environment" "$metal_venv"
  describe "Part 2 Metal model" "$metal_model_cache"
  describe "Part 2 model locks" "$metal_model_locks"
  describe "temporary state" "$state_dir"
else
  select_engine
  echo "This will stop vLLM and permanently remove:"
  echo "  - container: $name"
  echo "  - downloaded model volume: $volume"
  describe "temporary state" "$state_dir"
fi
echo "  - local quantizer environment and artifacts/: kept"
echo
echo "Other Hugging Face models and container images will not be removed."
confirm

if [ -n "$engine" ]; then ENGINE="$engine" ./stop.sh; else ./stop.sh; fi

if [ "$(uname -s)" = "Darwin" ]; then
  remove_exact "Metal environment" "$metal_venv" "$metal_venv"
  remove_exact "Part 2 Metal model" "$metal_model_cache" "$metal_model_cache"
  remove_exact "Part 2 model locks" "$metal_model_locks" "$metal_model_locks"
else
  if [ -n "$engine" ] && "$engine" volume inspect "$volume" >/dev/null 2>&1; then
    echo "Removing container model volume..."
    "$engine" volume rm "$volume" >/dev/null
  fi
fi
remove_exact "temporary state" "$state_dir" "$state_dir"
echo "Cleanup complete. Your .venv and artifacts/ were kept."

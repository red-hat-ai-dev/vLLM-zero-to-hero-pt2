#!/bin/sh
set -eu

script_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/engine.sh
. "$script_dir/scripts/engine.sh"

state_dir="${TMPDIR:-/tmp}/vllm-zero-to-hero-pt2"
user_home="${HOME:-}"
assume_yes="false"
engine=""

usage() {
  echo "Usage: ./cleanup.sh [--yes]"
  echo "Stop vLLM and remove files managed by this tutorial."
  echo "  --yes  Skip the confirmation prompt."
}

if [ -z "$user_home" ] || [ "$user_home" = "/" ]; then
  echo "Error: HOME does not point to a safe user directory. Nothing was removed." >&2
  exit 1
fi

case "${1:-}" in
  "") ;;
  --yes) assume_yes="true" ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Error: unknown option: $1" >&2; usage >&2; exit 1 ;;
esac
if [ "$#" -gt 1 ]; then echo "Error: too many options." >&2; usage >&2; exit 1; fi

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

if [ "$(uname -s)" = "Darwin" ]; then
  echo "This will stop vLLM and permanently remove:"
  describe "temporary state" "$state_dir"
else
  resolve_managed_engine "$name" "$managed_label" "$volume"
  echo "This will stop vLLM and permanently remove:"
  echo "  - container: $name"
  echo "  - downloaded model volume: $volume"
fi
echo "  - local quantizer environment and artifacts/: kept"
echo
echo "The shared Hugging Face cache and container images are kept."
echo "The shared vLLM Metal environment is kept."
confirm

if [ "$(uname -s)" = Darwin ]; then
  "$script_dir/scripts/metal.sh" cleanup "$state_dir"
else
  stop_lesson_container
  if [ -n "$engine" ]; then remove_model_volume; fi
fi
echo "Cleanup complete. Your .venv and artifacts/ were kept."

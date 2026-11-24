#!/bin/sh
set -eu

script_dir="$(CDPATH='' cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/engine.sh
. "$script_dir/scripts/engine.sh"

port="${PORT:-8000}"
repo_root="$script_dir"
state_dir="${TMPDIR:-/tmp}/vllm-zero-to-hero-pt2"
log_file="$state_dir/metal.log"
metal_venv="${VLLM_METAL_VENV:-$HOME/.venv-vllm-metal}"
metal_model="${METAL_MODEL:-mlx-community/Qwen3-4B-4bit}"
metal_draft_model="${METAL_DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
baseline_model="${BASELINE_MODEL:-Qwen/Qwen3-4B}"
optimized_default="$repo_root/artifacts/Qwen3-4B-FP8-Dynamic"
served_model="${MODEL_NAME:-qwen3-4b}"
max_model_len="8192"
speculator_model="${SPECULATOR_MODEL:-RedHatAI/Qwen3-4B-speculator.dspark}"
num_speculative_tokens="${NUM_SPECULATIVE_TOKENS:-3}"
backend=""
metal_token=""
engine=""
model_arg=""
local_model_mount="false"
mode=""
requested=""
accelerator=""

usage() {
  echo "Usage: ./run.sh [baseline|optimized|speculative] [auto|nvidia|amd|metal|cpu]" >&2
  echo >&2
  echo "Examples:" >&2
  echo "  ./run.sh baseline" >&2
  echo "  ./run.sh optimized nvidia" >&2
  echo "  OPTIMIZED_MODEL=RedHatAI/Qwen3-4B-FP8-dynamic ./run.sh speculative" >&2
}

error() {
  echo "Error: $*" >&2
}

require_curl() {
  if ! command -v curl >/dev/null 2>&1; then
    error "curl is required but was not found."
    exit 1
  fi
}

show_recent_logs() {
  echo >&2
  echo "Recent vLLM logs:" >&2
  if [ "$backend" = "metal" ]; then
    tail -n 40 "$log_file" >&2 2>/dev/null || true
  elif [ -n "${engine:-}" ]; then
    "$engine" logs --tail 40 "$container_id" >&2 2>/dev/null || true
  fi
}

cleanup_started_backend() {
  if [ "$backend" = "metal" ]; then
    "$script_dir/scripts/metal.sh" stop "$state_dir" "$metal_token"
  elif [ "$backend" = "container" ]; then
    remove_launched_container || true
  fi
}

cancel_start() {
  echo >&2
  error "Startup was cancelled. Cleaning up so you can try again."
  cleanup_started_backend
  exit 130
}

backend_is_running() {
  if [ "$backend" = "metal" ]; then
    "$script_dir/scripts/metal.sh" status "$state_dir" "$metal_token" >/dev/null
  else
    [ "$("$engine" inspect --format '{{.State.Running}}' "$container_id" 2>/dev/null || true)" = "true" ]
  fi
}

wait_until_ready() {
  deadline=$(( $(date +%s) + 1800 ))
  printf "Waiting for vLLM"
  until curl --connect-timeout 2 --max-time 5 --fail --silent "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do

    if ! backend_is_running; then
      echo
      error "vLLM stopped before it became ready."
      show_recent_logs
      cleanup_started_backend
      return 1
    fi

    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo
      error "vLLM did not become ready within 30 minutes."
      show_recent_logs
      cleanup_started_backend
      return 1
    fi

    printf "."
    sleep 5
  done

  if ! backend_is_running; then
    error "vLLM stopped before it became ready."
    show_recent_logs
    cleanup_started_backend
    return 1
  fi
  echo
  trap - INT TERM HUP
  echo "vLLM is ready at http://127.0.0.1:$port/v1"
}

engine_is_running() {
  command -v "$1" >/dev/null 2>&1 && "$1" info >/dev/null 2>&1
}

engine_has_nvidia_support() {
  candidate="$1"
  case "$candidate" in
    podman|*/podman)
      if "$candidate" info 2>/dev/null | grep -Fq "nvidia.com/gpu=all"; then
        return 0
      fi
      command -v nvidia-ctk >/dev/null 2>&1 &&
        nvidia-ctk cdi list 2>/dev/null | grep -Fq "nvidia.com/gpu=all"
      ;;
    docker|*/docker)
      "$candidate" info --format '{{json .Runtimes}}' 2>/dev/null |
        grep -Fq '"nvidia"'
      ;;
    *) return 1 ;;
  esac
}

show_nvidia_setup_error() {
  selected_engine="${1:-}"
  if [ -n "$selected_engine" ]; then
    error "ENGINE is set to '$selected_engine', but it cannot access the NVIDIA GPU."
  else
    error "NVIDIA was detected, but no running container engine can access it."
  fi
  echo "Install NVIDIA Container Toolkit, then try again:" >&2
  echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  echo >&2
  echo "Podman must list nvidia.com/gpu=all in: nvidia-ctk cdi list" >&2
  echo "Docker must list an nvidia runtime in: docker info" >&2
}

select_engine() {
  if [ -n "${ENGINE:-}" ]; then
    if ! command -v "$ENGINE" >/dev/null 2>&1; then
      error "ENGINE is set to '$ENGINE', but that command was not found."
      exit 1
    fi
    if ! "$ENGINE" info >/dev/null 2>&1; then
      error "$ENGINE is installed but is not running."
      echo "Start $ENGINE, then run ./run.sh again." >&2
      exit 1
    fi
    if [ "$accelerator" = "nvidia" ] && ! engine_has_nvidia_support "$ENGINE"; then
      show_nvidia_setup_error "$ENGINE"
      exit 1
    fi
    engine="$ENGINE"
    return
  fi

  podman_running="false"
  docker_running="false"
  if engine_is_running podman; then podman_running="true"; fi
  if engine_is_running docker; then docker_running="true"; fi

  if [ "$podman_running" = "false" ] && [ "$docker_running" = "false" ]; then
    error "No running container engine was found."
    echo "Install and start Docker or Podman, then run ./run.sh again." >&2
    exit 1
  fi

  if [ "$accelerator" != "nvidia" ]; then
    if [ "$podman_running" = "true" ]; then engine="podman"; else engine="docker"; fi
    return
  fi

  if [ "$podman_running" = "true" ] && engine_has_nvidia_support podman; then
    engine="podman"
  elif [ "$docker_running" = "true" ] && engine_has_nvidia_support docker; then
    engine="docker"
  else
    show_nvidia_setup_error
    exit 1
  fi
}

detect_accelerator() {
  if [ "$requested" != "auto" ]; then
    accelerator="$requested"
  elif command -v nvidia-smi >/dev/null 2>&1 || [ -e /dev/nvidiactl ]; then
    accelerator="nvidia"
  elif [ -e /dev/kfd ]; then
    accelerator="amd"
  else
    accelerator=""
  fi

  case "$accelerator" in
    nvidia|amd|cpu) ;;
    *)
      error "No supported NVIDIA or AMD accelerator was detected."
      echo "Use NVIDIA or AMD, choose cpu explicitly, or run on an Apple Silicon Mac." >&2
      exit 1
      ;;
  esac
}

prepare_optimized_model() {
  model_source="${OPTIMIZED_MODEL:-$optimized_default}"
  if [ ! -d "$model_source" ] && [ -d "$repo_root/$model_source" ]; then
    model_source="$repo_root/$model_source"
  fi

  if [ -d "$model_source" ]; then
    model_arg="/models/optimized"
    local_model_mount="true"
    optimized_mount_source="$model_source"
    return
  fi

  if [ -n "${OPTIMIZED_MODEL:-}" ]; then
    model_arg="$OPTIMIZED_MODEL"
    local_model_mount="false"
    return
  fi

  error "No local optimized checkpoint was found at $model_source."
  echo "Run the quantization step first, or use a published checkpoint:" >&2
  echo "OPTIMIZED_MODEL=RedHatAI/Qwen3-4B-FP8-dynamic ./run.sh optimized" >&2
  exit 1
}

check_port_free() {
  if curl --connect-timeout 2 --max-time 5 --silent --output /dev/null "http://127.0.0.1:$port/v1/models"; then
    error "An HTTP server already answers on port $port. Stop it or choose PORT."
    exit 1
  fi
}

run_linux() {
  check_port_free
  detect_accelerator
  echo "Detected Linux with $accelerator acceleration."
  select_engine
  echo "Using $engine."

  if "$engine" container inspect "$name" >/dev/null 2>&1; then
    error "$name already exists."
    echo "Run ./stop.sh, then try again." >&2
    exit 1
  fi

  case "$accelerator" in
    nvidia)
      image="${IMAGE:-docker.io/vllm/vllm-openai:v0.28.0}"
      if [ "${engine##*/}" = "podman" ]; then
        set -- --device nvidia.com/gpu=all --security-opt=label=disable
      else
        set -- --gpus all
      fi
      ;;
    amd)
      image="${IMAGE:-docker.io/vllm/vllm-openai-rocm:v0.28.0}"
      set -- --device /dev/kfd --device /dev/dri --group-add video \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined
      ;;
    cpu)
      if [ "$(uname -m)" != "x86_64" ]; then
        error "The CPU example currently requires an x86-64 Linux environment."
        exit 1
      fi
      if [ "$mode" != "baseline" ]; then
        error "The CPU example currently supports baseline serving only."
        exit 1
      fi
      image="docker.io/vllm/vllm-openai-cpu:v0.28.0-x86_64"
      max_model_len="4096"
      set -- -e VLLM_CPU_KVCACHE_SPACE=1 -e VLLM_CPU_NUM_OF_RESERVED_CPU=1
      ;;
  esac

  case "$mode" in
    baseline)
      model_arg="$baseline_model"
      ;;
    optimized|speculative)
      prepare_optimized_model
      ;;
    *)
      error "Unknown mode: $mode"
      usage
      exit 1
      ;;
  esac

  set -- "$@" --ipc=host \
    -p "127.0.0.1:$port:8000" \
    -v "$volume:/root/.cache/huggingface" \
    --label "$managed_label=true"

  if [ "$local_model_mount" = "true" ]; then
    set -- "$@" -v "$optimized_mount_source:/models/optimized:ro,Z"
    echo "Using local optimized checkpoint: $optimized_mount_source"
  fi

  set -- "$@" "$image" \
    "$model_arg" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "$served_model" \
    --max-model-len "$max_model_len"

  if [ "$accelerator" = "cpu" ]; then
    set -- "$@" --dtype bfloat16 --max-num-seqs 4 --enforce-eager
  fi

  if [ "$mode" = "speculative" ]; then
    speculative_config="{\"method\":\"dspark\",\"model\":\"$speculator_model\",\"num_speculative_tokens\":$num_speculative_tokens}"
    set -- "$@" --speculative-config "$speculative_config"
    echo "Using draft model: $speculator_model"
  fi

  echo "Starting $mode vLLM server. The first run may download model files."
  ensure_model_volume
  backend="container"
  container_id=""
  launch_id=""
  trap cancel_start INT TERM HUP
  if ! start_lesson_container "$@"; then
    error "$engine could not start the vLLM container."
    exit 1
  fi

  wait_until_ready
}

install_metal() {
  if [ -x "$metal_venv/bin/vllm" ]; then return; fi

  if [ -n "${VLLM_METAL_VENV:-}" ]; then
    error "No vLLM executable was found in VLLM_METAL_VENV: $metal_venv"
    exit 1
  fi

  echo "Installing the official stable vLLM Metal environment in $metal_venv."
  installer="$(mktemp "${TMPDIR:-/tmp}/vllm-metal-install.XXXXXX")"
  if ! curl --fail --silent --show-error --location \
    https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh \
    --output "$installer"; then
    rm -f "$installer"
    error "Could not download the official vLLM Metal installer."
    exit 1
  fi
  if ! bash "$installer" --stable; then
    rm -f "$installer"
    error "vLLM Metal installation failed."
    exit 1
  fi
  rm -f "$installer"
}

run_macos() {
  case "$requested" in
    auto|metal) ;;
    nvidia|amd|cpu)
      error "$requested acceleration through this launcher requires Linux."
      exit 1
      ;;
    *) error "Unknown accelerator: $requested"; usage; exit 1 ;;
  esac

  if [ "$mode" = "optimized" ]; then
    error "The optimized path is not advertised for vLLM Metal yet."
    echo "Run the CPU quantization step on this Mac if desired, then serve the optimized checkpoint on Linux." >&2
    exit 1
  fi
  if [ "$(uname -m)" != "arm64" ]; then
    error "vLLM Metal requires an Apple Silicon Mac."
    exit 1
  fi
  macos_major="$(sw_vers -productVersion | cut -d. -f1)"
  if [ "$macos_major" -lt 15 ]; then
    error "vLLM Metal requires macOS 15 or newer."
    exit 1
  fi

  echo "Detected Apple Silicon."
  install_metal
  echo "Starting vLLM Metal with $metal_model."
  set -- "$metal_venv/bin/vllm" serve "$metal_model" \
    --host 127.0.0.1 --port "$port" \
    --served-model-name "$served_model" --max-model-len 8192
  if [ "$mode" = "speculative" ]; then
    echo "Using draft model: $metal_draft_model"
    set -- "$@" --speculative-config "{\"method\":\"draft_model\",\"model\":\"$metal_draft_model\",\"num_speculative_tokens\":$num_speculative_tokens}"
  fi
  launch="$("$script_dir/scripts/metal.sh" start "$state_dir" "$mode" "$port" "$@")"
  metal_token="${launch#* }"
  if [ "${launch%% *}" = existing ]; then
    if curl --connect-timeout 2 --max-time 5 --fail --silent "http://127.0.0.1:$port/v1/models" >/dev/null; then
      echo "vLLM is already ready at http://127.0.0.1:$port/v1"
      return
    fi
    error "The requested Metal server is already starting. See $log_file"
    return 1
  fi
  backend="metal"
  trap cancel_start INT TERM HUP
  wait_until_ready
}

if [ "$#" -gt 2 ]; then
  usage
  exit 1
fi

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

mode="${1:-baseline}"
requested="${2:-auto}"
require_curl

case "$mode" in
  baseline|optimized|speculative) ;;
  *) error "Unknown mode: $mode"; usage; exit 1 ;;
esac

case "$(uname -s)" in
  Darwin) run_macos ;;
  Linux) run_linux ;;
  *) error "This operating system is not supported: $(uname -s)"; exit 1 ;;
esac

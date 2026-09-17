#!/bin/sh
set -eu

port="${PORT:-8000}"
model="${MODEL_NAME:-qwen3-4b}"
prompt="${*:-Explain quantization in three sentences.}"

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required." >&2
  exit 1
fi

payload="$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.argv[2]}]}))' "$model" "$prompt")"

curl --fail --show-error --silent \
  "http://127.0.0.1:$port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$payload"
echo

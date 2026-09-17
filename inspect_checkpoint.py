#!/usr/bin/env python3
"""Show the evidence that a model-free PTQ run created a compressed checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a compressed-tensors checkpoint.")
    parser.add_argument(
        "--path",
        default="artifacts/Qwen3-4B-FP8-Dynamic",
        help="Checkpoint directory to inspect.",
    )
    args = parser.parse_args()
    checkpoint = Path(args.path).expanduser()
    config_path = checkpoint / "config.json"

    if not config_path.is_file():
        print(f"Error: could not find {config_path}", file=sys.stderr)
        print("Run quantize.py first, or pass --path to a checkpoint directory.", file=sys.stderr)
        return 1

    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Error: could not read {config_path}: {error}", file=sys.stderr)
        return 1

    quantization = config.get("quantization_config")
    safetensors = sorted(checkpoint.glob("*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in safetensors)

    print(f"Checkpoint: {checkpoint}")
    print(f"Safetensors files: {len(safetensors)}")
    print(f"Safetensors size: {total_bytes / (1024**3):.2f} GiB")
    if quantization:
        print("quantization_config: present")
        print(f"quant_method: {quantization.get('quant_method', 'not listed')}")
        print(f"quantization_status: {quantization.get('quantization_status', 'not listed')}")
    else:
        print("quantization_config: not found")
        return 1

    print("This directory contains a new compressed checkpoint that vLLM can load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

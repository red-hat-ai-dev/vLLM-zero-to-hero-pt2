#!/usr/bin/env python3
"""Create a model-free FP8 checkpoint for the Part 2 walkthrough."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a Safetensors model with LLM Compressor model-free PTQ."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-4B",
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/Qwen3-4B-FP8-Dynamic",
        help="Directory where the compressed checkpoint will be written.",
    )
    parser.add_argument(
        "--scheme",
        default="FP8_DYNAMIC",
        help="LLM Compressor scheme. The tutorial default is FP8_DYNAMIC.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Quantization device, for example cpu, cuda:0, or auto.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of checkpoint worker threads (default: 4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()

    if args.max_workers < 1:
        print("Error: --max-workers must be at least 1.", file=sys.stderr)
        return 2

    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"Error: output directory already exists and is not empty: {output_dir}",
            file=sys.stderr,
        )
        print("Choose a new output path so an existing checkpoint is not overwritten.", file=sys.stderr)
        return 2

    try:
        from llmcompressor import model_free_ptq
    except ImportError:
        print(
            "Error: LLM Compressor is not installed. Run "
            "'python -m pip install -r requirements.txt' first.",
            file=sys.stderr,
        )
        return 1

    device = None if args.device == "auto" else args.device
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source model: {args.model_id}")
    print(f"Scheme: {args.scheme}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("This may take a while. The original checkpoint is never modified.")

    model_free_ptq(
        model_stub=args.model_id,
        save_directory=output_dir,
        scheme=args.scheme,
        ignore=["lm_head", "model.embed_tokens"],
        max_workers=args.max_workers,
        device=device,
    )

    print(f"\nCompressed checkpoint written to {output_dir}")
    print("Start it with: ./run.sh optimized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

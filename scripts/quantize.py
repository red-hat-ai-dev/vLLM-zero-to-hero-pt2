#!/usr/bin/env python3
"""Create and validate the compressed checkpoint used by Part 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

MODEL_ID = "Qwen/Qwen3-4B"
SAVE_DIR = "artifacts/Qwen3-4B-FP8-Dynamic"


def validate_checkpoint(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    quantization = config.get("quantization_config", {})
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("quantization_status") != "compressed"
    ):
        raise ValueError("Checkpoint is missing compressed-tensors/compressed metadata")
    shards = sorted(directory.glob("*.safetensors"))
    if not shards or any(path.stat().st_size == 0 for path in shards):
        raise ValueError("Checkpoint has missing or empty Safetensors weights")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weights = json.loads(index.read_text()).get("weight_map", {})
        if not weights or any(
            Path(name).name != name or not (directory / name).is_file()
            for name in weights.values()
        ):
            raise ValueError("Safetensors index references missing or invalid shards")
    elif len(shards) != 1:
        raise ValueError("Sharded checkpoint is missing its Safetensors index")
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError("Checkpoint is missing tokenizer_config.json")
    return {"files": len(shards), "bytes": sum(path.stat().st_size for path in shards)}


def quantize_checkpoint(model_id: str, output_dir: Path, device: str = "cpu") -> dict:
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(
            f"Output already exists at {output_dir}. Choose a new output directory."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the same filesystem; a failed conversion never replaces a good result.
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent
        )
    )
    try:
        from llmcompressor import model_free_ptq

        print(f"Quantizing {model_id} on {device} with FP8_DYNAMIC...", flush=True)
        model_free_ptq(
            model_stub=model_id,
            save_directory=stage,
            scheme="FP8_DYNAMIC",
            ignore=["lm_head", "model.embed_tokens"],
            max_workers=4,
            device=device,
        )
        summary = validate_checkpoint(stage)
        os.rename(stage, output_dir)
    except BaseException:
        print(
            f"Conversion did not complete. Diagnostic files remain at {stage}",
            flush=True,
        )
        raise
    print(f"Compressed checkpoint created: {output_dir}")
    print(
        f"Files: {summary['files']}; weight size: {summary['bytes'] / 1024**3:.2f} GiB"
    )
    print(
        "File validation passed. Serve the checkpoint to validate runtime loading and inference."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    quantize_checkpoint(args.model, args.output_dir, args.device)


if __name__ == "__main__":
    main()

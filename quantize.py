#!/usr/bin/env python3
"""Create the compressed checkpoint used by the Part 2 walkthrough."""

from __future__ import annotations

import json
from pathlib import Path

from llmcompressor import model_free_ptq


MODEL_ID = "Qwen/Qwen3-4B"
SAVE_DIR = Path("artifacts/Qwen3-4B-FP8-Dynamic")

if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()):
    raise SystemExit(
        f"Output already exists at {SAVE_DIR}. Choose a new directory or remove the old one."
    )

print(f"Quantizing {MODEL_ID} on the CPU with FP8_DYNAMIC...")
print("The first run downloads the model and may take a while.")

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", "model.embed_tokens"],
    max_workers=4,
    device="cpu",
)

safetensors = sorted(SAVE_DIR.rglob("*.safetensors"))
total_bytes = sum(path.stat().st_size for path in safetensors)
config = json.loads((SAVE_DIR / "config.json").read_text())
quantization = config.get("quantization_config", {})

print("\nCompressed checkpoint created.")
print(f"Files: {len(safetensors)} Safetensors files")
print(f"Size: {total_bytes / (1024**3):.2f} GiB")
print(f"Format: {quantization.get('quant_method', 'not listed')}")
print(f"Status: {quantization.get('quantization_status', 'not listed')}")
print(f"Location: {SAVE_DIR}")

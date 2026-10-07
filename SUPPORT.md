# Tested support: Part 2

This is a tutorial validation record, not a vendor support guarantee. Status is
specific to the versions and models below. Last reviewed: 2026-10-07.

## Platform record

| Path | Configuration | Evidence |
| --- | --- | --- |
| Linux / NVIDIA | RTX 3090 24 GiB, driver 610.43.02, Debian, Podman 5.4.2 + CDI, vLLM 0.28.0 | Fresh baseline, locally converted FP8, and DSpark serving passed 2026-10-07 |
| Local CPU quantization | macOS 26.7.1 arm64, Python 3.13, LLM Compressor 0.13.0, PyTorch 2.13.0 | Fresh conversion and file validation passed 2026-10-07; 3 shards, 4.11 GiB |
| Linux CPU and CUDA quantization | LLM Compressor 0.13.0 quantizer container, RTX 3090 for CUDA | Both conversion paths passed 2026-10-07; separate 4.11 GiB outputs |
| Apple Silicon serving | Apple M4 Pro 48 GiB, vLLM Metal 0.29.0, MLX 0.32.1, MLX LM 0.32.0 | Fresh baseline/speculative greedy and sampled chat, completions, and stop passed 2026-10-07 |
| AMD / Intel | vLLM 0.28.0 ROCm / XPU launch paths | Exact baseline/FP8/DSpark combinations unverified |

The fresh CUDA run used `Qwen/Qwen3-4B` as baseline, an FP8_DYNAMIC checkpoint
created by `./quantize.sh`, and `RedHatAI/Qwen3-4B-speculator.dspark` with three
draft tokens. All modes passed greedy and sampled chat requests plus a
completions request. Cleanup also passed from outside the repository without an
ENGINE override and preserved both generated checkpoints. Speculative metrics increased to 57 drafts, 171 drafted
tokens, and 71 accepted tokens. These small requests prove functionality, not a
controlled speedup or accuracy result. The RTX 3090 uses a weight-only FP8 path;
it is not evidence of native FP8 performance on newer hardware.

The [fresh Metal hardware record](validation/2026-10-07-metal.json) used
`mlx-community/Qwen3-4B-4bit` with `Qwen/Qwen3-0.6B` and `draft_model`, with three
draft tokens. Counters recorded 39 drafts, 117 draft tokens, and 49 accepted
tokens. Both serving modes passed the short API checks; this is not a controlled
speculative speedup measurement.

The historical Metal run on 2026-09-25 used `mlx-community/Qwen3-4B-4bit` and
`Qwen/Qwen3-0.6B` with `draft_model`, not DSpark. Its local manual note reported
acceptance and working sampled requests. It is separate from the fresh checks.
The stable Metal path uses speculation for greedy requests; sampled requests
may run without speculation. Metal `optimized` mode rejects the Linux FP8
checkpoint deliberately. An MLX 4-bit model and a Linux FP8 model are different
optimization paths.

## Reproducibility and recovery

Linux serving pins vLLM 0.28.0; quantization pins LLM Compressor 0.13.0. Transitive
Python dependencies are not fully locked. The Metal installer requests the
current stable release and reuses an existing environment; record the actual
installed versions instead of assuming it matches a past test.

`QUANTIZE_OUTPUT` chooses a fresh checkpoint path; `QUANTIZE_DEVICE=cuda`
selects CUDA conversion. CPU remains the default. A failed conversion keeps
its staging directory and never publishes a partial checkpoint at the final
path. File checks cannot replace an actual loading/inference test.

Podman uses CDI device arguments; Docker uses `--gpus all`. `ENGINE` is honored
by start, stop, and cleanup. Ambiguous ownership fails explicitly. Cleanup can
be invoked from another directory and keeps generated artifacts, the quantizer
environment, and the shared Metal environment.

See the [sanitized hardware record](validation/2026-10-07.json) for model revisions,
image identity, and request scope.

## Reproduce the checks

Run `python3 -m unittest discover -s tests -v` for the committed regression and
documentation checks. The pull-request workflow runs these on Linux and macOS
with Python 3.10 and 3.13. These tests use fake engines; they do not establish GPU
support. Successful CI, a successful image build, and successful inference are
separate kinds of evidence.

For a hardware result, record the commit, date, OS/architecture, GPU and driver,
engine/version, image digest or Python packages, exact model revision, command,
request payload, response, and logs. For speculative decoding include accepted
and drafted token counters. Record failed steps too. Do not include credentials
or model weights in an issue or PR.

## Remaining validation

AMD ROCm, Intel XPU, and WSL2 have not been exercised by this change. Do not infer
support for the tutorial's exact model and quantization from an upstream backend
support table. Maintainers should attach equivalent hardware evidence before
marking those paths tested. No image publication or GPU CI is performed by the
pull-request checks.

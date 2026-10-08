# Behind the scenes

The [walkthrough](../README.md) has two pieces: create a compressed model, then
serve a model through vLLM. Quantization produces files on disk. Serving loads
a checkpoint and starts the local API. You can do those steps on different
computers.

## What is in this repository?

| File or folder | What it does |
| --- | --- |
| [`quantize.sh`](../quantize.sh) | Prepares the quantizer's dependencies and creates a checkpoint. |
| [`run.sh`](../run.sh) | Starts the baseline, optimized, or speculative server. |
| [`stop.sh`](../stop.sh) | Stops the server and keeps downloaded models. |
| [`cleanup.sh`](../cleanup.sh) | Asks before removing this lesson's serving cache and temporary state. |
| [`scripts/`](../scripts) | Holds the Python quantizer, container build file, and container ownership helper. |
| [`docs/assets/`](assets) | Holds the diagrams used in the walkthrough. |

Running the lesson also creates local files such as `artifacts/` and a
quantizer environment. These are generated on your computer, not downloaded
as part of the repository.

## What `quantize.sh` does

The wrapper first looks for a running Podman or Docker engine. It builds the
`quantizer` stage in [`scripts/Dockerfile`](../scripts/Dockerfile), which
installs LLM Compressor 0.13.0 and copies in
[`scripts/quantize.py`](../scripts/quantize.py). The model cache and output
directory are mounted into that container.

If neither engine is running, it creates a local Python environment using
Python 3.10 through 3.13. The environment is named
`.venv-pt2-quantizer-<python-version>` and is reused on later runs.
`QUANTIZER_BACKEND=local` selects this path explicitly; `ENGINE` selects a
container engine explicitly.

Both paths run the same Python quantizer. It calls `model_free_ptq` with:

- `Qwen/Qwen3-4B` as the source checkpoint;
- the `FP8_DYNAMIC` scheme;
- `lm_head` and `model.embed_tokens` left out of quantization;
- four workers and CPU conversion by default.

`QUANTIZE_DEVICE=cuda` selects CUDA conversion. Converting a checkpoint on CPU
does not imply that the resulting FP8 model will serve efficiently on CPU.
The CPU serving option in this lesson runs the baseline only.

The default output is `artifacts/Qwen3-4B-FP8-Dynamic`. Conversion first writes
to a temporary sibling directory. The script checks the compressed-tensors
metadata, weight files, shard index when needed, and tokenizer configuration
before moving the completed checkpoint into place. A nonempty existing output
is refused. Failed conversions leave their temporary directory for diagnosis.

These checks establish that the expected files exist. Loading the model and
measuring its answers are separate checks, covered by serving and Part 3.

## What `run.sh` does

The launcher chooses a backend, starts vLLM, and waits for `/v1/models` to
answer. The default API is `http://127.0.0.1:8000/v1`, and the model is exposed
as `qwen3-4b` so the request in the walkthrough works across the paths.

| Mode | Linux GPU path | Apple Silicon path |
| --- | --- | --- |
| `baseline` | `Qwen/Qwen3-4B` | `mlx-community/Qwen3-4B-4bit` through vLLM Metal |
| `optimized` | Your local FP8 checkpoint, or the model selected by `OPTIMIZED_MODEL` | This mode is not enabled in Part 2. |
| `speculative` | The optimized target with `RedHatAI/Qwen3-4B-speculator.dspark` | The Metal target with `Qwen/Qwen3-0.6B` as a draft model |

Part 2's Mac baseline is already a four-bit MLX checkpoint. Part 3 uses a
different Mac comparison: BF16 versus four-bit MLX. The FP8 checkpoint created
here is for the Linux GPU path.

### Linux containers and CPU

NVIDIA uses the vLLM CUDA image; AMD uses the ROCm image. The launcher prefers
Podman when it can access the accelerator, otherwise it can use Docker.
An explicit `ENGINE` must also be running and meet the NVIDIA integration
checks when applicable.

The container port is published only on the host's loopback address. Downloaded
models are stored in the `vllm-zero-to-hero-pt2-models` volume. A local optimized
checkpoint is mounted read-only at `/models/optimized`.

`./run.sh baseline cpu` explicitly chooses the official x86-64 CPU image. It
uses BF16, eager execution, a 4,096-token context, a 1 GiB KV cache, and at most
four concurrent requests. CPU is never selected automatically, and the
optimized and speculative modes are not enabled for it.

### Apple Silicon and speculation

The Mac path uses the shared vLLM Metal environment at `~/.venv-vllm-metal`,
installing the official stable release if it is missing. Set `VLLM_METAL_VENV`
to use an existing environment instead. The server runs in the background;
its PID and log are in `${TMPDIR:-/tmp}/vllm-zero-to-hero-pt2/`.

Linux speculation uses the `dspark` method. Metal uses `draft_model` with a
separate small draft model. The default proposal length is three tokens,
controlled by `NUM_SPECULATIVE_TOKENS`. The draft can be selected with
`SPECULATOR_MODEL` on Linux or `METAL_DRAFT_MODEL` on Mac.

Use `"temperature": 0` for the Metal example: its draft path applies to greedy
requests. Adding a draft introduces extra work as well as potential savings;
measure the result on your hardware before treating it as a speed improvement.

## Stopping, cleanup, and failures

The launcher allows up to 30 minutes for startup. It checks whether the backend
has exited while waiting, reports recent logs on failure, and cleans up the
backend it started so you can retry.

`stop.sh` stops the native Metal process or removes the managed serving
container. It keeps model downloads for the next start.

[`scripts/engine.sh`](../scripts/engine.sh) helps `stop.sh` and `cleanup.sh`
find the container engine that owns this lesson's resources. If separate
resources exist in both engines, it asks you to select `ENGINE`. It also checks
the container's project label before allowing removal.

After confirmation, `cleanup.sh` removes the serving container and model
volume on Linux, or the default Part 2 Metal model cache on Mac, plus temporary
state. It keeps the shared Metal environment, the local quantizer environment,
container images, and `artifacts/`. The draft model and custom model caches are
also kept on Mac. Cleanup stops if it cannot stop the server.

## Building the images

The quantizer image is built automatically by `quantize.sh`. The same
Dockerfile has a separate `serving` stage used by the image publication
workflow. To build that stage from the repository root:

```bash
docker build -f scripts/Dockerfile --target serving -t vllm-zero-to-hero-pt2:cuda .
```

The launcher uses upstream vLLM images by default; `IMAGE` can select a custom
GPU serving image. Model weights are downloaded at runtime or mounted from a
local checkpoint.

## Tested support

The [Part 2 RFC](https://github.com/red-hat-ai-dev/vLLM-zero-to-hero-pt2/issues/2)
holds validation results, exact environments, and remaining platform checks.
The presence of a backend in the launcher does not establish that every mode
has been tested on that hardware.

[Back to the walkthrough](../README.md)

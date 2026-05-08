# ggml

[Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

Tensor library for machine learning

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***

## This fork

This is a Qwen3-TTS-specific fork of [`ggml-org/ggml`](https://github.com/ggml-org/ggml) used as the backing tensor library for [`dbrain/qwen3-tts.cpp`](https://github.com/dbrain/qwen3-tts.cpp). 13 commits ahead of upstream `master`.

What it adds, by area:

**Vocoder kernels (CUDA):**
- `GGML_OP_SNAKE` (CPU + CUDA, F32/F16) — fused `α·sin(βx)² + γx` for the WavTokenizer-class vocoder activation. Replaces a `pow(sin(αx), 2) / α + β·x` broadcast chain with ~10× tensor-broadcast overhead.
- `GGML_OP_CONV_1D_DIRECT` — smem-tiled CUDA kernel with a tensor-core `wmma` variant (F16 weights). Single biggest win on the vocoder side of the qwen3-tts fork (see [`dbrain/qwen3-tts.cpp/docs/ARCHITECTURE.md`](https://github.com/dbrain/qwen3-tts.cpp/blob/main/docs/ARCHITECTURE.md)).
- smem-tiled F16 wmma `conv_transpose_1d` kernel.
- F16 in/out paths through `conv_1d_direct`, `conv_transpose_1d`, `snake`, `acc` + `ggml_*_to` dst-type API variants — keeps the cascade in F16 instead of F32 and halves the scheduler arena.
- `concat` F16 + I32 paths (was F32-only via assert).

**Talker / megakernel infrastructure (CUDA):**
- `mul_mat` dispatcher hook (`ggml_cuda_set_mul_mat_hook`) — lets external code install shape-specialized matmul kernels at runtime, falling through to ggml's generic path on miss.
- `graph_compute_begin` hook + cgraph-aware variant — per-cgraph plan rebuild trigger; receives the cgraph so external code can scan for fusion opportunities before dispatch.
- Generic per-op hook — for sub-op fusion (currently unused in production; the fusions that needed it didn't pay).
- `Q4_K` `get_rows` CUDA kernel (was missing — Q4_K_M models couldn't get_rows on GPU and fell back to CPU).
- I32 `row_indices` in the `ROPE+VIEW+SET_ROWS` fusion path — unblocks talker streaming KV writes.

**Multi-backend / latency-sensitive workloads:**
- `ggml_backend_cuda_init_with_priority` + `params="priority=low|high"` parsing in `device_init_backend`. Lets a process pin specific backends (e.g. talker = HIGH, vocoder = default) so high-priority work preempts low-priority work for SM time. (No-op on consumer Ampere where `cudaDeviceGetStreamPriorityRange` returns `[-5, 0]` and least == default; useful on devices with a real priority range below default.)
- Sched fixes: `hash_set` 2× `graph_size`; `reserve_n` failure check; null-check on tensor-copy malloc.
- Kernel-launch error checks in `conv_1d_direct`, `snake`, `acc`.
- `supports_op` tightening for `CONV_1D_DIRECT`, `SNAKE`, `ACC`.

These are mostly Qwen3-TTS-specific layouts (`wmma` shapes, F16 conv variants tuned around the cascade arena budget, the snake activation) or invasive enough that they don't merge upstream as-is. The fork exists so the qwen3-tts.cpp build can pin a known-working ggml without waiting on PR cycles for project-specific kernels.

For the project this powers, see [`dbrain/qwen3-tts.cpp`](https://github.com/dbrain/qwen3-tts.cpp). Performance numbers, methodology, and the lever-by-lever architecture story live in that repo's README + `docs/`.

## Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime

## Build

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

# install python dependencies in a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

## GPT inference (example)

```bash
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

For more information, checkout the corresponding programs in the [examples](examples) folder.

## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

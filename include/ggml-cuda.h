#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
// Return unused physical pages held by CUDA VMM pools while keeping active
// allocations valid. Intended for explicit request/window boundaries.
GGML_BACKEND_API void ggml_backend_cuda_trim_memory(ggml_backend_t backend);

// Free cuDNN's cached reordered 3D-convolution weights. These allocations are
// external to ggml's VMM pool and are keyed by temporary staged weight pointers,
// so a video-window boundary must release them after those staging buffers go
// away. No-op when ggml-cuda was built without cuDNN.
GGML_BACKEND_API void ggml_backend_cuda_release_cudnn_conv3d_weights(void);

// Register a per-tensor NVFP4 weight global scale (ModelOpt weight_scale_2) of an
// UNFOLDED import, keyed by the ggml tensor name of the weight it belongs to. The FP4
// cuBLASLt GEMM folds it into the matmul alpha, which is free, instead of the graph
// paying a full-size elementwise multiply per Linear. Names that were never registered
// multiply by 1.0 (legacy FOLDED gguf path stays byte-identical) — which means an
// unfolded weight whose scalar was NOT registered produces silently wrong output, so
// callers must gate any graph-level simplification on ggml_cuda_nvfp4_weight_global_folded().
GGML_BACKEND_API void ggml_cuda_nvfp4_register_weight_global(const char * name, float g);

// Drop every registered weight global. Must be called before re-registering for a
// hot-swapped diffusion model: the registry is process-global, so stale entries would
// either double-scale a folded gguf or mis-scale a different unfolded one.
GGML_BACKEND_API void ggml_cuda_nvfp4_clear_weight_globals(void);

// True iff a mul_mat of the NVFP4 weight named `name` on `backend` is GUARANTEED to be
// served by the cuBLASLt FP4 path AND that path will apply the registered weight global
// through the GEMM alpha. Only then may a caller elide its own compensating multiply.
// False whenever anything is unproven (non-CUDA backend, FP4 cuBLASLt disabled,
// pre-Blackwell device, name not registered) — the caller must then keep its multiply.
GGML_BACKEND_API bool ggml_cuda_nvfp4_weight_global_folded(ggml_backend_t backend, const char * name);

// True iff `backend` can run an NVFP4 mul_mat with an F16 activation and an F16 destination
// (the cuBLASLt FP4 GEMM: E2M1 activation quant that reads `half`, F32 accumulate, F16 store).
// This is the ONLY such route, so a graph builder must consult this before choosing to run a
// residual stream in F16 -- where it is false, every F16 mul_mat fails supports_op and
// ggml_backend_sched drops those nodes to the CPU backend, which is catastrophic, not slow.
// False for a non-CUDA backend, with the FP4 cuBLASLt env gate off, or pre-Blackwell.
// NOTE: this answers the DEVICE question only. Per-node shape constraints (2D, contiguous,
// K % 64 == 0) are still enforced independently by ggml_backend_supports_op().
GGML_BACKEND_API bool ggml_cuda_nvfp4_f16_dst_available(ggml_backend_t backend);

// One LoRA module's contribution to one weight, as f32 host arrays.
// `down` is [rank, in] row-major (== a ggml lora_down with ne = {in, rank}) and `up` is
// [rows, rank] row-major (== a ggml lora_up with ne = {rank, rows}). row_begin/rows exist
// because a LoRA may target a PACKED projection in segments, each owning a slice of the
// output rows, which the runtime path expresses as a concat along dim 0.
struct ggml_cuda_lora_module {
    const float * down;
    const float * up;
    int64_t rank;
    int64_t row_begin;
    int64_t rows;
    float   scale;
};

// Merge `mods` into an NVFP4 weight IN PLACE, on the GPU. `blocks` is the host-side ggml
// NVFP4 block array ([out, in/64] blocks of 36 bytes); it is uploaded, folded and written
// back. Every UE4M3 block scale is preserved EXACTLY -- the grid is frozen, so `.wglobal`
// never changes and the weight-global registry above stays valid. `inv_wglobal` converts
// the true-unit delta into the scaled domain the nibbles live in.
//
// Rounding is STOCHASTIC, which is not a detail: the LoRA delta is smaller than the NVFP4
// quantisation step, so round-to-nearest discards most of it (measured projection 0.06 vs
// 0.94). It is also DETERMINISTIC -- the stream is a counter-based hash of (seed, row,
// element) -- so two folds of the same inputs produce identical bytes.
//
// Returns false without touching `blocks` if the shapes are unsupported or CUDA/cuBLAS
// fails; the caller is expected to fall back to the CPU fold.
GGML_BACKEND_API bool ggml_cuda_lora_fold_nvfp4(void * blocks, int64_t in, int64_t out,
                                                float inv_wglobal,
                                                const struct ggml_cuda_lora_module * mods,
                                                int n_mods, uint64_t seed);

// Free the fold's device scratch and cuBLAS handle. Safe to call when no fold is running.
GGML_BACKEND_API void ggml_cuda_lora_fold_release(void);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif

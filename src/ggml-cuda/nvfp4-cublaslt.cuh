#pragma once
#include "common.cuh"

// Phase-1 fast FP4 GEMM: route an NVFP4 weight mul_mat through cuBLASLt's
// CUDA_R_4F_E2M1 / VEC16_UE4M3 blockscaled GEMM (Blackwell FP4 tensor cores,
// ~3.3x the ggml MMQ-FP4 kernel). Gated by env GGML_NVFP4_CUBLASLT=1.
//
// Returns true if it handled the op; false (no side effects) to fall back to
// the normal dispatch (MMQ / dequant-cublas).

bool ggml_cuda_nvfp4_cublaslt_enabled();
bool ggml_cuda_nvfp4_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src0,
                                      const ggml_tensor * src1,
                                      ggml_tensor * dst);

// TASK F: drop every entry from the pointer-keyed weight-repack cache. The in-place
// repack overwrites src0->data, so if an NVFP4 weight buffer is freed and a later alloc
// reuses the same address a stale "already repacked" entry would be served for the new
// (un-repacked) weight -> silent corruption. Call this whenever NVFP4 weight buffers are
// freed (e.g. the flux2 warm worker's per-request DiT free). Safe to call with cuBLASLt
// disabled / cache empty (no-op). Thread-safe (takes the cache mutex). Also exported as
// the C-callable ggml_cuda_nvfp4_clear_repack_cache() in ggml-cuda.h for host callers.
void ggml_cuda_nvfp4_cublaslt_clear_repack_cache();

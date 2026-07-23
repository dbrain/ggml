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

// True iff a per-tensor NVFP4 weight global (ModelOpt weight_scale_2) was registered for
// `name` via ggml_cuda_nvfp4_register_weight_global (declared in ggml-cuda.h). Used by
// ggml_cuda_nvfp4_weight_global_folded() to prove to the graph builder that this weight's
// scalar really is folded into the GEMM alpha, so the redundant full-size ggml_mul can be
// dropped. Never guess: an unregistered name silently multiplies by 1.0.
bool ggml_cuda_nvfp4_weight_global_registered(const char * name);

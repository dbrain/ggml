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

// FP8 (e4m3) FFN path: promote an NVFP4 FFN weight to e4m3 + quantize the activation
// to e4m3 (both per-tensor SCALAR scale) and run a cuBLASLt FP8xFP8 GEMM. 8-bit
// activations kill the FP4-activation "worm"/stipple in flat regions (the Q4_K-clean
// recipe = 4-bit weight + 8-bit act) while running ~2x FP4 cost (not BF16's ~8x).
// Env-gated (GGML_FP8_FFN=1) + name filter (GGML_FP8_LAYERS, default "ff.net"); default
// off => prod byte-identical. Returns true if it handled the op, false to fall back.
bool ggml_cuda_fp8_ffn_enabled();
bool ggml_cuda_fp8_ffn_name_match(const char * name);
bool ggml_cuda_fp8_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor * src0,
                                    const ggml_tensor * src1,
                                    ggml_tensor * dst);

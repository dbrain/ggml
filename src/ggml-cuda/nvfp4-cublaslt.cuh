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

// FP8 (e4m3) FFN path: promote an NVFP4 weight to e4m3 + quantize the activation to e4m3
// (both per-tensor SCALAR scale) and run a cuBLASLt FP8xFP8 GEMM. 8-bit activations kill the
// FP4-activation "worm"/stipple on flat regions under motion (the Q4_K-clean recipe = 4-bit
// weight + 8-bit act) at ~2x FP4 cost instead of BF16's ~8x. The weight stays FP4-STORED --
// the e4m3 copy is transient pool scratch -- so steady-state VRAM is unchanged.
//
// Env-gated: GGML_FP8_FFN=1 plus a name substring filter GGML_FP8_LAYERS (default "ff.net";
// prod ships "transformer_blocks", i.e. every DiT Linear). Default off => byte-identical to
// the FP4-only behaviour. Returns true if it handled the op, false (dst untouched) to fall
// back to the FP4 / MMQ / dequant path.
//
// The registered NVFP4 weight global (weight_scale_2) is applied INSIDE the e4m3 promotion,
// so the graph-level elision proven by ggml_cuda_nvfp4_weight_global_folded() stays correct
// for the Linears this path steals from the FP4 GEMM.
bool ggml_cuda_fp8_ffn_enabled();
bool ggml_cuda_fp8_ffn_name_match(const char * name);
bool ggml_cuda_fp8_cublaslt_mul_mat(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor * src0,
                                    const ggml_tensor * src1,
                                    ggml_tensor * dst);

// Invalidate the FP8 AND NVFP4 activation-quant reuse caches (one counter each, bumped
// together; GGML_NVFP4_ACT_QUANT_CACHE gates the FP4 one and is default OFF).
// MUST be called once per graph compute,
// before any node of that graph runs: the cache reuses an e4m3 activation across the q/k/v
// Linears that share one src1, and node/data ADDRESSES are recycled by gallocr between
// computes, so identity alone cannot tell "same activation" from "same address, new value".
// The generation counter is what makes a cross-compute hit impossible.
// Called from ggml_backend_cuda_graph_compute() (ggml-cuda.cu) — the backend's own graph
// entry point, so it covers every host path and every ggml_backend_sched split. Cheap relaxed
// atomic; a no-op when GGML_FP8_FFN is off.
void ggml_cuda_fp8_act_cache_new_generation(void);

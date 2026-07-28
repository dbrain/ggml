#pragma once

#include "common.cuh"

// cuDNN fused SDPA flash-attention variant (Blackwell sm_120+).
//
// Env-gated alternative to ggml's flash_attn_ext_f16 MMA kernel. Built only when
// the ggml-cuda backend is configured with -DGGML_CUDNN=ON (links libcudnn + nvrtc,
// includes cudnn-frontend headers). When the macro is undefined the function is a
// no-op stub and the selection guard in fattn.cu never routes to it.
//
// Shape contract (flux2-klein DiT self-attn): Q/K/V ne = [D, L, H, N] (BHSD memory
// layout, contiguous), mask == nullptr, max_bias == 0, logit_softcap == 0,
// gqa_ratio == 1, D in {64,128}, K/V == F16. Output dst is fresh contiguous F32 with
// ne = [D, H, L, N] (BSHD memory layout). See attn_golden.cu for the reference graph.

void ggml_cuda_flash_attn_ext_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_FLASH_ATTN_EXT_LSE: same SDPA, plus cuDNN's softmax Stats (the per-query
// natural-log log-sum-exp) written after the output in the packed F32 dst. Asking for
// Stats builds a SEPARATE cuDNN graph, so the inference plan cache is untouched.
void ggml_cuda_flash_attn_ext_lse_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// True when the TU was compiled with cuDNN support (GGML_CUDNN defined at build).
bool ggml_cuda_cudnn_available();

// Explicit lifecycle hooks for long-lived worker/request boundaries. The
// execution plans own cuDNN-side resources and must be released before the
// backing CUDA context is torn down.
void ggml_cuda_cudnn_sdpa_release_plans();
void ggml_cuda_cudnn_sdpa_release_handle();

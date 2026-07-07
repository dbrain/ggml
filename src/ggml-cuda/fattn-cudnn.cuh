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

// True when the TU was compiled with cuDNN support (GGML_CUDNN defined at build).
bool ggml_cuda_cudnn_available();

// Destroy every cached cuDNN SDPA execution plan (the file-scope, shape-keyed
// plan cache). Each cached fe::graph::Graph holds cuDNN-backend device memory
// that is NEVER returned to the driver — a plan built for one shape (e.g. a
// longer-token base pass) squats for the rest of the process and taxes every
// later phase's reserve-time high-water. ggml_backend_cuda_trim_pools cannot
// reach it (it lives outside the ggml VMM pool). Call ONLY at a boundary where
// no attention op is in flight (issues no sync itself). Plans rebuild lazily on
// the next attention op of that shape (one-time build cost, not per-step).
// No-op stub when the TU was built without GGML_CUDNN.
void ggml_cuda_cudnn_sdpa_release_plans();

#pragma once

#include "common.cuh"

// Custom FP8 (e4m3) flash-attention for consumer Blackwell (sm120), v2.
//
// Streaming / online-softmax flash kernel that runs QK^T on the sm120 FP8 tensor
// cores (new e4m3 m16n8k32 MMA in mma.cuh) and P*V in F16. Env-gated
// (GGML_FP8_ATTN=1) and OFF by default; the default cuDNN F16 SDPA path is
// byte-untouched. Generic + reusable: handles head_dim D in {64,128}, mask-free,
// self- OR cross-attention (Lq may != Lkv), any batch N / head count H; anything
// else falls through to cuDNN F16 unchanged.
//
// Shape contract (same as the cuDNN borrow): Q ne = [D, Lq, H, N] (BHSD memory,
// contiguous), K/V ne = [D, Lkv, H, N], mask == nullptr, softmax scale in
// dst->op_params[0]. dst is fresh contiguous F32 or F16 with ne = [D, H, Lq, N]
// (BSHD memory). See fattn-cudnn.cu for the reference dispatch/permute conventions
// and spike_cutlass_fp4/attn_ltx_golden.cu for the F32 attention reference.

void ggml_cuda_flash_attn_ext_fp8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

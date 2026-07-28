#pragma once

#include "common.cuh"

// Experimental SageAttention3 route. The implementation accepts only the
// contiguous, mask-free LTX self-attention contract and returns false for all
// other graphs so the caller keeps the cuDNN path.
bool ggml_cuda_flash_attn_ext_sa3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_FLASH_ATTN_EXT_LSE on SA3. Same kernel, plus the per-query natural-log
// log-sum-exp written after the output in the packed F32 dst, and Lq != Lkv allowed (the
// relay's segment merge attends the full query set against one slice of the keys).
//
// Gated by GGML_LTX_SA3=1 AND GGML_LTX_SA3_LSE=1, both default off; returns false when the
// gate is closed or the shape does not fit, and the caller keeps the cuDNN path.
bool ggml_cuda_flash_attn_ext_lse_sa3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#pragma once

#include "common.cuh"

// Experimental SageAttention3 route. The implementation accepts only the
// contiguous, mask-free LTX self-attention contract and returns false for all
// other graphs so the caller keeps the cuDNN path.
bool ggml_cuda_flash_attn_ext_sa3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

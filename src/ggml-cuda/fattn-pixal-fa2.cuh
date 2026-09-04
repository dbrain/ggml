#pragma once

#include "common.cuh"

// Source-built FlashAttention v2 bridge for Pixal's exact BF16 self-attention
// case (D=128, B=1, no mask/bias/dropout). It is separately opt-in at build
// and graph construction time; generic ggml attention keeps its normal path.
bool ggml_cuda_flash_attn_ext_pixal_fa2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

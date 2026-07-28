#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// GGML_OP_FLASH_ATTN_EXT_LSE -- flash attention that also writes the per-query softmax
// log-sum-exp. cuDNN SDPA only; supported() is false everywhere else, so a caller can gate
// on ggml_backend_supports_op and keep its old path.
void ggml_cuda_flash_attn_ext_lse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_lse_supported(int device, const ggml_tensor * dst);

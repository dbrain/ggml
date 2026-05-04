#include "common.cuh"

// Explicit op (dbrain fork): GGML_OP_SNAKE — raw alpha/beta, applies exp().
void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fusion entry point (ggml-org upstream). Caller supplies x/a/inv_b explicitly.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * x,
                              const ggml_tensor * a,
                              const ggml_tensor * inv_b,
                              ggml_tensor *       dst);

#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_norm_fused_add(ggml_backend_cuda_context & ctx,
                                 ggml_tensor *               dst,
                                 ggml_tensor *               mul_tensor,
                                 ggml_tensor *               add_tensor);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

// Pre-norm bias fold: ADD(x, prebias) immediately followed by RMS_NORM (LTX DiT q/k_norm).
// `dst` = RMS_NORM node, `add_node` = preceding ADD, `x` = matmul-side ADD operand,
// `prebias` = the 1D F32 bias operand. See norm.cu for the bit-exactness argument.
void ggml_cuda_op_rms_norm_fused_prebias(ggml_backend_cuda_context & ctx,
                                         ggml_tensor *               dst,
                                         ggml_tensor *               add_node,
                                         ggml_tensor *               x,
                                         ggml_tensor *               prebias);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// Fused AdaLN (comfy adaln.cu equivalent): out = rms_norm(x)*(1+scale)+shift as ONE op.
// dst->src[0]=x (F16|F32), dst->src[1]=scale, dst->src[2]=shift (scale/shift may be a
// different type than x — read with no cast; the +1 is intrinsic to the kernel).
void ggml_cuda_op_rms_modulate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#pragma once

#include "common.cuh"

// cuDNN implicit-GEMM 2D convolution (Blackwell sm_120+), env-gated alternative to
// ggml's direct conv2d kernel (GGML_OP_CONV_2D). Built only when ggml-cuda is configured
// with -DGGML_CUDNN=ON. When the macro is undefined this returns false and the caller
// keeps using the direct kernel.
//
// Lifts the proven cudnn-frontend conv-fprop graph from
// flux2.cpp/spike_cutlass_fp4/conv_golden.cu (3x3 s1 p1, NHWC fp16, cosine 1.0 on the
// flux2-klein VAE decoder shapes). The im2col(materialize)+GEMM path the flux2 VAE
// currently uses pays an IC*KH*KW HBM blowup; cuDNN's implicit GEMM skips it.
//
// Contract (ggml GGML_OP_CONV_2D / ggml_conv_2d_direct):
//   kernel = dst->src[0]  ne=[KW,KH,IC,OC]  (f16 or f32), contiguous
//   input  = dst->src[1]  ne=[IW,IH,IC,N]   f32, NCHW memory
//   dst                    ne=[OW,OH,OC,N]   f32, NCHW memory
//   op_params = {ST_X,ST_Y,PD_X,PD_Y,DL_X,DL_Y,cwhn==0}
//
// Returns true if it handled the op (ran cuDNN), false to fall back to the direct kernel.

// Gate: env GGML_CUDNN_CONV=1 AND compiled with GGML_CUDNN AND supported shape.
bool ggml_cuda_op_conv2d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// True when this TU was compiled with cuDNN support.
bool ggml_cuda_conv2d_cudnn_available();

// Drop cached conv2d cuDNN frontend plans and destroy the thread-local handle.
// Must be called from the CUDA worker thread that created the handle.
void ggml_cuda_cudnn_conv2d_release_handle();

// Free the raw-cudaMalloc'd reordered conv-weight buffers (g_weight_cache) and clear the
// cache. Twin of ggml_cuda_cudnn_conv3d_release_weights.
//
// The cache is keyed by STABLE IDENTITY (tensor name + buffer + shape + type + device, see
// cudnn-weight-key.cuh), NOT by the weight's device address, so re-offloading or otherwise
// moving the weights does NOT invalidate it and this need not be called for that. What it
// is for: the weight CONTENT changing under an unchanged tensor name (a LoRA epoch change,
// a different checkpoint loaded into the same names), and freeing the buffers when the
// model goes away. Call with no conv2d op in flight. No-op stub without GGML_CUDNN.
void ggml_cuda_cudnn_conv2d_release_weights();

// True when the reorder cache holds anything. Lets a caller skip the device synchronize
// that must precede a release. No-op stub (false) without GGML_CUDNN.
bool ggml_cuda_cudnn_conv2d_has_cached_weights();

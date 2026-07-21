#pragma once

#include "common.cuh"

// cuDNN implicit-GEMM 3D convolution (Blackwell sm_120+), env-gated alternative to
// ggml's im2col_3d + GEMM decomposition. Built only when ggml-cuda is configured with
// -DGGML_CUDNN=ON. When the macro is undefined this returns false and the caller keeps
// using the im2col_3d path.
//
// Lifts the proven cudnn-frontend 3D conv-fprop graph from
// flux2.cpp/spike_cutlass_fp4/conv3d_golden.cu (3x3x3 s1, NDHWC fp16, depth-pad0 /
// spatial-pad1, cosine 1.0 on the LTX-2.3 22B VideoVAE decoder shapes). LTX decodes its
// VAE via im2col_3d (materialize the IC*KD*KH*KW=IC*27 column blowup to HBM) + GEMM;
// cuDNN's implicit 3D GEMM skips the materialization entirely (the 27x blowup is worse
// than 2D's 9x, so the win is bigger here).
//
// Contract (ggml GGML_OP_CONV_3D / ggml_conv_3d_direct):
//   kernel = dst->src[0]  ne=[KW,KH,KD,c*oc]  (f16 or f32), contiguous  (KCRS-3d:
//                          memory ((((oc*c+ic)*KD+kd)*KH+kh)*KW+kw))
//   input  = dst->src[1]  ne=[W,H,D,c*n]      f32, NCDHW memory (per n: [c][d][h][w])
//   dst                    ne=[OW,OH,OD,oc*n]  f32, NCDHW memory (per n: [oc][od][oh][ow])
//   op_params = {s0,s1,s2, p0,p1,p2, d0,d1,d2, c, n, oc, hi_prec}   (axis 0=W, 1=H, 2=D)
//              hi_prec=1 -> F32-IO plan: cuDNN writes fp32 Y (not fp16) into an F32 dst, so the
//              output isn't fp16-quantized (WAN_VAE_HEAD_F32 head.2 unpatchify-grid fix).
//
// Returns true if it handled the op (ran cuDNN), false to fall back (caller aborts /
// CPU). Activations transposed/cast NCDHW-f32 <-> NDHWC-f16 around the call with the
// same 32x32 tiled shared-mem transpose the conv2d borrow uses (DHW collapsed to one
// spatial dim). Plan cached by shape; weight reordered KCRS-3d->KRSC-3d f16 once per ptr.

// Gate: env GGML_CUDNN_CONV3D (or GGML_CUDNN_CONV) =1 AND compiled with GGML_CUDNN AND
// a supported shape AND workspace within the cap. Otherwise returns false.
bool ggml_cuda_op_conv3d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// True when this TU was compiled with cuDNN support.
bool ggml_cuda_conv3d_cudnn_available();

// Destroy every cached cuDNN 3D-conv execution plan (the file-scope, shape-keyed
// plan cache). Like the SDPA plan cache, each cached fe::graph::Graph pins
// cuDNN-backend device memory that is never returned to the driver until the plan
// is destroyed — the big VAE-decoder conv plans built during the first segment's
// decode then tax every subsequent phase's reserve-time high-water. Only the plan
// cache is cleared; the per-weight reorder caches (raw cudaMalloc, keyed by the
// persistent weight ptr) are kept, so no leak and no re-reorder. Call only when no
// conv op is in flight. No-op stub when built without GGML_CUDNN.
void ggml_cuda_cudnn_conv3d_release_plans();

// Clear cached conv3d plans and destroy this thread's cuDNN handle. Use this when
// plan-cache clearing alone does not return cuDNN-internal per-handle device
// reservations to the driver. Must be called from the CUDA worker thread that
// created the handle, at a boundary with no in-flight cuDNN op.
void ggml_cuda_cudnn_conv3d_release_handle();

// Free the raw-cudaMalloc'd reordered conv-weight buffers (g_weight3d_cache /
// g_weight3d_f32_cache) and clear the caches. Required on a continuation because
// LTXAV_VAE_LAZY re-offloads the VAE params each segment -> weights get a fresh device
// address -> old reorder buffers orphaned (~1.4 GB/segment leak). Call at a segment
// boundary after the VAE params are released, with no conv3d op in flight. No-op stub
// when built without GGML_CUDNN.
void ggml_cuda_cudnn_conv3d_release_weights();

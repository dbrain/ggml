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

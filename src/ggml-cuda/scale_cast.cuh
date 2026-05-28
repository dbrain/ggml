#pragma once

#include "common.cuh"

// LongCat lap-28.5: fused SCALE -> CAST(F16) for the kv_scale prescale pattern
// used by the avatar's flash-attn wrapper and the lap-28.2 cond-cache consume
// path. The unfused chain bounces ~153 MB F32 -> 153 MB F32 (scale) + 76 MB F16
// (cast) per noise k/v of 9360 tokens × 128 head_dim × 32 heads — fusing
// collapses the round-trip to a single read F32 / write F16 (~229 MB saved per
// pair). Bit-exact (one MUL rounding + F32→F16 cast, same as the unfused chain).
void ggml_cuda_op_scale_cast_f16(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor *         src_f32,
                                 ggml_tensor *               dst_f16,
                                 float                       scale,
                                 float                       bias);

#pragma once

#include "common.cuh"

// LongCat lap-28.3 (gate_add fusion):
// Fused dst = x + y * gate where y and x share the same shape and `gate`
// broadcasts on ONE dim of the MUL view. Specifically tuned for the avatar's
// gate_add(x, y, gate) pattern (RESHAPE_4d -> MUL -> RESHAPE_3d -> ADD with x).
// All operands are contiguous F32. Bit-exact equivalent to the unfused
// MUL+ADD chain (uses __fmul_rn + __fadd_rn to avoid FMA contraction).
void ggml_cuda_op_mul_add_bcast(ggml_backend_cuda_context & ctx,
                                ggml_tensor *               mul_n,   // the MUL node (output ne == y/x shape, post-bcast)
                                ggml_tensor *               add_n,   // the ADD node (final dst; output buffer)
                                const ggml_tensor *         x,       // ADD's other src (residual)
                                const ggml_tensor *         y_view,  // MUL's main src in y-shape (4D view)
                                const ggml_tensor *         gate,    // MUL's broadcast src
                                const ggml_tensor *         shift = nullptr); // flux AdaLN: optional trailing +shift (same bcast layout as gate)

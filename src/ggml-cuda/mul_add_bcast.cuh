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

// Same-shape (no-broadcast) fused multiply-add: dst = x + y*g (+ shift). All
// operands contiguous F32 of identical element count. Covers NAVA's per-token
// AdaLN modulation (x + x*scale + shift, batch N=1) which the broadcast path
// above cannot match. Bit-exact (__fmul_rn + __fadd_rn).
void ggml_cuda_op_fused_madd_same(ggml_backend_cuda_context & ctx,
                                  ggml_tensor *               add_n,
                                  const ggml_tensor *         x,
                                  const ggml_tensor *         y,
                                  const ggml_tensor *         g,
                                  const ggml_tensor *         shift = nullptr);

// Fused bias-add + GELU: dst = gelu(x + bias), bias broadcast on dims 1..3
// ([d0,1,1,1] over [d0,tokens]). Covers the FFN w1 path (ggml_ext_linear's
// add_inplace(bias) immediately followed by ggml_gelu_inplace) — the bias-add
// otherwise runs as a full-width op_add<half,float,half> over the 4x-wide
// [inner_dim, tokens] intermediate. Bit-exact: rounds to BIG after the add
// (matching the half store) then GELU (matching ggml's op_gelu single-precision
// tanh approximation) and rounds to BIG on store.
void ggml_cuda_op_bias_gelu(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               gelu_n,  // final dst (output buffer)
                            const ggml_tensor *         x,       // matmul output (ADD's src0)
                            const ggml_tensor *         bias);   // 1D bias (ADD's src1, broadcast)

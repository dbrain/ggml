#pragma once

#include "common.cuh"

// GGML_CUDNN_ATTN_LSE_SELFCHECK -- prove a flash-attention backend's per-query log-sum-exp
// is what the segment merge assumes it is.
//
// Backend-agnostic on purpose: cuDNN's "Stats" and SageAttention3's softmax row state are
// produced by completely different machinery, and both feed the SAME recombination
// (ltxv.hpp ltx_segmented_self_attention), so both are checked against one oracle.
//
// `tag`   -- printed in the log line, e.g. "cudnn" or "sa3".
// `Q`/`K` -- the op's own contiguous BHSD inputs; Q may be F32 or F16, K must be F16.
// `lse`   -- device F32 [Lq, H] with the query innermost, i.e. lse[h*Lq + row].
// `tol`   -- absolute residual, in nats, below which the natural-log reading is accepted.
//            cuDNN's exact F16 SDPA lands at ~1e-3; an FP4 backend cannot, so it passes a
//            looser bound. Both plausible failure modes stay far outside either: a missing
//            or doubled log2(1/(448*6)) offset is ~7.9 nats, a log2-base reading is ~30% of
//            |LSE|. The raw residuals for both hypotheses are printed regardless.
//
// No-op unless GGML_CUDNN_ATTN_LSE_SELFCHECK is set; prints at most a few times per process.
void ggml_cuda_attn_lse_selfcheck(const char * tag,
                                  const ggml_tensor * Q, const ggml_tensor * K, const float * lse,
                                  int64_t H, int64_t Lq, int64_t Lkv, int64_t D,
                                  float scale, float tol, cudaStream_t stream);
